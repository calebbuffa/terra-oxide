//! Async tile load scheduler.
//!
//! [`LoadScheduler`] owns the async task dispatch and result draining that was
//! previously embedded in `ContentManager`.  It:
//!
//! * Holds the [`TileStore`] with **unique ownership** (not `Arc`).
//! * Dispatches load tasks on the background thread pool, providing each
//!   task a snapshot of the store fields it needs (no `Arc<TileStore>`).
//! * Drains completed [`TileLoadResult`]s and normalises them into
//!   [`LoadEvent`]s, applying all `TileStore` mutations inline - no `unsafe`.
//! * Handles `ExternalTileset` and `StoreInit` completely internally, so
//!   callers only see `Loaded`, `Empty`, `Failed`, `RetryLater`.
//! * Tracks camera velocity for (foveated time-delay).
//!   (cull-requests-while-moving) gating.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::{CancellationToken, Task};
use terra::Ellipsoid;

use crate::async_runtime::AsyncRuntime;
use crate::camera_motion::CameraMotionTracker;
use crate::dispatch_gate::{DispatchContext, DispatchGate};
use crate::frame_decision::{LoadEvent, LoadRequest};
use crate::loader::{
    ContentLoader, TileContentKind, TileLoadInput, TileLoadResult, TileLoadResultState,
};
use crate::options::SelectionOptions;
use crate::selection_state::TileStates;
use crate::tile_store::{TileId, TileStore};
use crate::view::ViewState;

/// Async tile load scheduler.
///
/// Uniquely owns the [`TileStore`]; background tasks receive a snapshot of
/// the specific store fields they need at dispatch time.
pub(crate) struct LoadScheduler {
    /// Uniquely owned tile tree - safe to mutate on the main thread without
    /// any `unsafe` - no background tasks hold a reference to this.
    pub store: TileStore,
    pub loaders: Vec<Arc<dyn ContentLoader>>,
    accessor: Arc<dyn AssetAccessor>,
    headers: Arc<[(String, String)]>,
    pub(crate) ellipsoid: Arc<Ellipsoid>,
    runtime: AsyncRuntime,
    /// Active background tasks, keyed by `NodeIndex` for O(1) duplicate
    /// detection.  The `Task<()>` is kept alive so it doesn't get dropped;
    /// the `CancellationToken` lets us abort a tile's load if it is evicted
    /// before the background thread finishes.
    in_flight: HashMap<TileId, (Task<()>, CancellationToken)>,
    /// Completed raw load results delivered from background threads.
    result_rx: orkester::Receiver<(TileId, TileLoadResult)>,
    result_tx: orkester::Sender<(TileId, TileLoadResult)>,
    /// Per-frame camera motion state used by dispatch gates.
    camera_motion: CameraMotionTracker,
}

impl LoadScheduler {
    pub fn new(
        store: TileStore,
        loaders: Vec<Arc<dyn ContentLoader>>,
        accessor: Arc<dyn AssetAccessor>,
        headers: Vec<(String, String)>,
        ellipsoid: Ellipsoid,
        runtime: AsyncRuntime,
        channel_capacity: usize,
    ) -> Self {
        let (result_tx, result_rx) = orkester::mpsc(channel_capacity);
        Self {
            store,
            loaders,
            accessor,
            headers: headers.into(),
            ellipsoid: Arc::new(ellipsoid),
            runtime,
            in_flight: HashMap::new(),
            result_rx,
            result_tx,
            camera_motion: CameraMotionTracker::default(),
        }
    }

    /// Register an additional `ContentLoader` (e.g. from an external tileset).
    pub fn push_loader(&mut self, loader: Arc<dyn ContentLoader>) {
        self.loaders.push(loader);
    }

    /// Number of tiles currently in-flight.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Cancel the in-flight load for `tile`, if any.
    ///
    /// The background task may already have completed by the time this is
    /// called - that is fine.  Cancellation is cooperative and best-effort;
    /// the result channel may still receive a (now-stale) result which will
    /// be silently discarded during `drain()` because the tile is no longer
    /// in `in_flight`.
    pub fn cancel_tile(&mut self, tile: TileId) {
        if let Some((_, token)) = self.in_flight.remove(&tile) {
            token.cancel();
        }
    }

    /// Cancel all in-flight loads and clear the in-flight table.
    ///
    /// Call this when the entire tileset is being unloaded so that background
    /// threads stop sending stale results.
    pub fn cancel_all(&mut self) {
        for (_, (_, token)) in self.in_flight.drain() {
            token.cancel();
        }
    }

    /// Update camera-velocity state and dispatch load requests.
    ///
    /// Mirrors `ContentManager::load_tiles` + the camera-state update that was
    /// in `update_view_group`.
    pub fn dispatch(
        &mut self,
        requests: &[LoadRequest],
        states: &TileStates,
        views: &[ViewState],
        delta_time: f32,
        opts: &SelectionOptions,
        maximum_screen_space_error: f64,
        gates: &[Arc<dyn DispatchGate>],
    ) {
        // Update camera-motion state.
        if let Some(first_view) = views.first() {
            self.camera_motion.update(
                first_view.position,
                delta_time,
                opts.streaming.camera_movement_threshold,
            );
        }

        let max_loads = opts.loading.max_simultaneous_loads;

        // Cancel in-flight loads whose tiles are no longer requested.
        //
        // The load queue is recomputed every frame and contains every tile the
        // current frame would like loaded. An in-flight entry missing from the
        // queue is therefore outdated (camera moved, LOD changed, tile kicked,
        // tileset replaced, ...). Cancelling releases its [`CancellationToken`]
        // so the background thread can bail out of expensive decode work and
        // frees a dispatch slot for higher-priority tiles the user actually
        // needs.
        if !self.in_flight.is_empty() {
            let mut requested: HashSet<TileId> = HashSet::with_capacity(requests.len());
            for req in requests {
                requested.insert(req.tile);
            }
            self.in_flight.retain(|tile, (_, token)| {
                if requested.contains(tile) {
                    true
                } else {
                    token.cancel();
                    false
                }
            });
        }

        let current_in_flight = self.in_flight.len();
        let slots = max_loads.saturating_sub(current_in_flight);
        if slots == 0 {
            return;
        }

        // Build gate context once (shared across all requests this frame).
        let gate_ctx = DispatchContext {
            time_since_camera_stopped_secs: self.camera_motion.time_since_stopped_secs(),
            camera_delta_magnitude: self.camera_motion.delta_magnitude(),
        };

        // Sort descending by priority so Urgent fires first.
        let mut loads = requests.to_vec();
        loads.sort_unstable_by(|a, b| {
            b.priority
                .group
                .partial_cmp(&a.priority.group)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.priority
                        .score
                        .partial_cmp(&a.priority.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut dispatched = 0usize;
        'outer: for req in &loads {
            if dispatched >= slots {
                break;
            }

            // Run each gate; skip this request if any gate vetoes.
            for gate in gates {
                if !gate.should_dispatch(req, &gate_ctx) {
                    continue 'outer;
                }
            }

            let before = self.in_flight.len();
            self.dispatch_one(req.tile, states, maximum_screen_space_error);
            if self.in_flight.len() > before {
                dispatched += 1;
            }
        }
    }

    /// Drain completed raw results and normalise into [`LoadEvent`]s.
    ///
    /// All `TileStore` mutations (gltf up-axis, updated bounds, external
    /// tileset children, `tile_initializer` callbacks) are applied inline
    /// with safe `&mut self.store` - no `unsafe` needed.
    ///
    /// `ExternalTileset` results are handled entirely here; callers only see
    /// `Loaded`, `Empty`, `Failed`, or `RetryLater`.
    pub fn drain(&mut self) -> Vec<LoadEvent> {
        let results: Vec<(TileId, TileLoadResult)> = self.result_rx.try_iter().collect();
        let completed: HashSet<TileId> = results.iter().map(|(n, _)| *n).collect();
        self.in_flight.retain(|tile, _| !completed.contains(tile));

        let mut events = Vec::with_capacity(results.len());
        for (tile, result) in results {
            if let Some(event) = self.normalise(tile, result) {
                events.push(event);
            }
        }
        events
    }

    fn dispatch_one(&mut self, tile: TileId, states: &TileStates, maximum_screen_space_error: f64) {
        use crate::selection_state::TileLoadState;
        let load_state = states.get(tile).load_state;

        // Skip if already loading/ready.
        if matches!(load_state, TileLoadState::Loading | TileLoadState::Ready) {
            return;
        }
        // Respect retry backoff.
        if load_state == TileLoadState::RetryScheduled {
            if states.frame_index < states.get(tile).next_retry_frame {
                return;
            }
        }
        // Skip if already in-flight (deduplication).
        if self.in_flight.contains_key(&tile) {
            return;
        }

        let loader = match self.loader_for(tile) {
            Some(l) => Arc::clone(l),
            None => return,
        };

        // Snapshot store fields at dispatch time - background task gets a
        // plain struct, no Arc<TileStore> reference needed.
        let input = TileLoadInput {
            tile,
            content_keys: self.store.content_keys(tile).to_vec(),
            world_transform: self.store.world_transform(tile),
            refinement: self.store.refinement(tile),
            accessor: Arc::clone(&self.accessor),
            headers: Arc::clone(&self.headers),
            runtime: self.runtime.clone(),
            ellipsoid: Arc::clone(&self.ellipsoid),
            maximum_screen_space_error,
        };

        let tx = self.result_tx.clone();
        let tx_err = tx.clone();
        let bg_ctx = self.runtime.background();
        let token = CancellationToken::new();

        let task: Task<()> = bg_ctx
            .run(move || loader.load_tile(input))
            .with_cancellation(&token)
            .then(&bg_ctx, move |load_result| {
                let _ = tx.send((tile, load_result));
            })
            .catch(&bg_ctx, move |err| {
                // Silently discard cancellations - the tile was evicted before
                // the load completed.  Any other error counts as a real failure.
                if !err.is_cancelled() {
                    let _ = tx_err.send((tile, TileLoadResult::failed()));
                }
            });

        self.in_flight.insert(tile, (task, token));
    }

    fn normalise(&mut self, tile: TileId, mut result: TileLoadResult) -> Option<LoadEvent> {
        let source_url = result.source_url.take();

        match result.state {
            TileLoadResultState::Failed => {
                return Some(LoadEvent::Failed {
                    tile,
                    url: source_url,
                    message: "Tile load failed".into(),
                });
            }
            TileLoadResultState::RetryLater => {
                return Some(LoadEvent::RetryLater { tile });
            }
            TileLoadResultState::Success => {}
        }

        Self::apply_post_load_mutations(&mut self.store, tile, &mut result);

        match result.content {
            TileContentKind::Empty => Some(LoadEvent::Empty { tile }),
            TileContentKind::Gltf(model) => Some(LoadEvent::Loaded { tile, model }),
            TileContentKind::Custom(content) => Some(LoadEvent::Custom { tile, content }),
            TileContentKind::External {
                root_descriptor,
                child_loaders,
            } => Some(self.absorb_external_tileset(tile, root_descriptor, child_loaders)),
        }
    }

    /// Apply every post-load side-effect onto the owned store.
    ///
    /// Split out from [`Self::normalise`] so the classification of the load
    /// result (success/failure/retry) is separate from the mechanical work of
    /// updating the tile store - a pure `&mut TileStore` transformation
    /// driven by the loader's returned fields.
    fn apply_post_load_mutations(store: &mut TileStore, tile: TileId, result: &mut TileLoadResult) {
        // Apply gltf up-axis correction (unique store ownership - no unsafe).
        if result.gltf_up_axis != zukei::Axis::Z {
            let correction = *zukei::get_up_axis_transform(result.gltf_up_axis, zukei::Axis::Z);
            let current = store.world_transform(tile);
            store.set_world_transform(tile, correction * current);
        }

        // Apply updated bounds inline.
        if let Some(bounds) = result.updated_bounds.take() {
            store.set_bounds(tile, bounds);
        }

        // Apply tile_initializer inline (unique store ownership - no unsafe).
        if let Some(f) = result.tile_initializer.take() {
            f(store, tile);
        }
    }

    /// Absorb a just-loaded external-tileset shell into this scheduler:
    /// splice the child descriptor onto `tile`, register any new loaders, and
    /// report the shell as an empty-geometry tile so the selection engine can
    /// keep traversing.
    fn absorb_external_tileset(
        &mut self,
        tile: TileId,
        root_descriptor: crate::tile_store::TileDescriptor,
        child_loaders: Vec<Arc<dyn ContentLoader>>,
    ) -> LoadEvent {
        self.store.insert_children(tile, &[root_descriptor]);
        for loader in child_loaders {
            self.push_loader(loader);
        }
        LoadEvent::Empty { tile }
    }

    fn loader_for(&self, tile: TileId) -> Option<&Arc<dyn ContentLoader>> {
        let idx = self
            .store
            .effective_loader_index(tile)
            .map(|li| li.index() as usize)
            .unwrap_or(0);
        self.loaders.get(idx)
    }
}

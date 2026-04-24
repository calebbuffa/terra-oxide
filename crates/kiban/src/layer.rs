//! [`Layer`] - the main entry point for streaming a 3D Tiles dataset.
//!
//! Drives tile selection, load scheduling, and raster overlay fetching.
//! Register listeners on the public event fields; they fire synchronously
//! during each [`tick`](Layer::tick) call.
//!
//! # Callback Thread Safety
//!
//! Callbacks run in different threads depending on the event type:
//! - **Background thread**: `tile_loaded`, `custom_tile_loaded` (safe for expensive work)
//! - **Main frame thread**: `tile_evicted`, `overlay_ready`, `overlay_detached`, `tile_failed` (keep brief)
//!
//! See [`callback_safety.md`](callback_safety.md) for detailed thread context, safety guidelines, and examples.
//!
//! # Quick start
//!
//! ```rust,ignore
//! let mut layer = Layer::new(root, Box::new(loader), vec![], externals, options);
//! layer.overlays_mut().add(Basemap::Osm.into_overlay());
//!
//! // Register listeners once - they fire synchronously during tick().
//! let handle = layer.handle();
//! let store  = Arc::clone(&my_content_store);
//! let _h1 = layer.tile_loaded.subscribe({
//!     let bg     = pool.context();
//!     let store  = Arc::clone(&store);
//!     let handle = handle.clone();
//!     move |args: &TileLoadedArgs| {
//!         let model  = Arc::clone(&args.model);
//!         let store  = Arc::clone(&store);
//!         let handle = handle.clone();
//!         bg.spawn(move || {
//!             let mesh = encode_glb(&model);
//!             store.lock().unwrap().insert(args.tile, mesh);
//!             handle.mark_tile_ready(args.tile);   // ← ack: overlays can now attach
//!         });
//!     }
//! });
//!
//! let _h2 = layer.tile_evicted.subscribe({
//!     let store = Arc::clone(&store);
//!     move |tile: &TileId| { store.lock().unwrap().remove(tile); }
//! });
//!
//! let _h3 = layer.overlay_ready.subscribe(|args: &OverlayReadyArgs| {
//!     renderer.apply_overlay(args.tile, args.overlay_id, &args.overlay);
//! });
//!
//! // per-frame:
//! wq.flush();                          // drain bg->main continuations
//! layer.tick(&mut vg, &views, dt);     // fires events, drives selection
//! render(layer.selected_tiles());      // draw using your content store
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use courtier::AssetAccessor;
use glam::DVec3;
use orkester::Task;
use sovra::{OverlayEngine, OverlayEvent, OverlayViewInfo};
use terra::Ellipsoid;

use crate::async_runtime::AsyncRuntime;
use crate::dispatch_gate::{CullWhileMovingGate, DispatchGate, FoveatedTimeDelayGate};
use crate::events::{CustomTileLoadedArgs, OverlayReadyArgs, TileFailedArgs, TileLoadedArgs};
use crate::eviction::{EvictionPolicy, MaxAgeEvictionPolicy};
use crate::frame_decision::{ExpandResult, LoadEvent};
use crate::hooks::FrameHook;
use crate::load_scheduler::LoadScheduler;
use crate::loader::{ContentLoader, TileChildrenResult, TileExcluder};
use crate::loaders::cesium::{TilesetInitResult, TilesetJsonError, TilesetJsonLoader};
use crate::memory_budget::MemoryBudget;
use crate::options::SelectionOptions;
use crate::scorer::{LoadPriorityScorer, WeightedComponentScorer};
use crate::selected_tile::SelectedTile;
use crate::selection_state::TileStates;
use crate::strategy::TraversalStrategy;
use crate::tile_store::{TileDescriptor, TileId, TileStore};
use crate::traversal::{SelectionContext, select_tiles};
use crate::view::{Projection, ViewGroup, ViewState};
use orkester::Event;
use orkester::EventListener;

/// Details passed to [`LayerOptions::on_load_error`].
#[derive(Debug, thiserror::Error)]
#[error("{message}")]
pub struct LoadError {
    pub url: Option<String>,
    pub message: String,
}

/// Configuration for a [`ContentManager`].
#[derive(Clone)]
pub struct LayerOptions {
    pub maximum_screen_space_error: f64,
    pub selection: SelectionOptions,
    pub on_load_error: Option<Arc<dyn Fn(LoadError) + Send + Sync>>,
    pub tile_cache_unload_time_limit_secs: f32,
    pub tile_excluders: Vec<Arc<dyn TileExcluder>>,
    /// Pluggable load priority scorer.
    ///
    /// Defaults to [`WeightedComponentScorer`] (4-component weighted composite).
    /// Override to implement custom per-frame load ordering without touching the traversal.
    pub scorer: Arc<dyn LoadPriorityScorer>,

    /// Pluggable dispatch gates (evaluated in order before each load request).
    ///
    /// Defaults to the two built-in gates:
    /// 1. [`FoveatedTimeDelayGate`] - defers non-central tiles until the camera stops.
    /// 2. [`CullWhileMovingGate`] - suppresses tiles whose bounding radius is small
    ///    relative to the per-frame camera displacement.
    ///
    /// Set to an empty `Vec` to disable all gating, or provide your own chain.
    pub gates: Vec<Arc<dyn DispatchGate>>,

    /// Pluggable traversal strategy.
    ///
    /// Defaults to [`DefaultTraversalStrategy`].  Override with
    /// [`SkipLodTraversalStrategy`] or a custom implementation to change
    /// how the tile tree is walked each frame.
    pub strategy: Arc<dyn TraversalStrategy>,

    /// Observer hooks called at key tick phase boundaries.
    ///
    /// Hooks are called in registration order.  Default: empty (no hooks).
    pub hooks: Vec<Arc<dyn FrameHook>>,

    /// Pluggable eviction policy: determines whether a `Renderable` tile's
    /// content is stale and should be re-fetched, and adjusts the effective
    /// SSE threshold under memory pressure.
    ///
    /// Defaults to `BudgetEvictionPolicy { inner: MaxAgeEvictionPolicy }`,
    /// which combines age-based expiry with memory-pressure SSE ramp-up.
    /// Replace with a custom [`EvictionPolicy`] implementation to override
    /// either or both behaviours.
    pub eviction_policy: Arc<dyn EvictionPolicy>,

    /// Optional renderer-provided occlusion proxy.
    ///
    /// When set and `SelectionOptions::culling::delay_refinement_for_occlusion`
    /// is `true`, the traversal will skip refining any tile the renderer
    /// reports as [`Occluded`](crate::occlusion::TileOcclusionState::Occluded).
    pub occlusion_proxy: Option<Arc<dyn crate::occlusion::TileOcclusionProxy>>,
}

impl Default for LayerOptions {
    fn default() -> Self {
        let opts = SelectionOptions::default();
        let gates = Self::default_gates(&opts);
        let strategy = Self::default_strategy(&opts);
        Self {
            maximum_screen_space_error: 16.0,
            selection: opts,
            on_load_error: None,
            tile_cache_unload_time_limit_secs: 0.0,
            tile_excluders: Vec::new(),
            scorer: Arc::new(WeightedComponentScorer::default()),
            gates,
            strategy,
            hooks: Vec::new(),
            eviction_policy: Arc::new(crate::eviction::BudgetEvictionPolicy::new(
                MaxAgeEvictionPolicy,
            )),
            occlusion_proxy: None,
        }
    }
}

impl LayerOptions {
    fn default_gates(opts: &SelectionOptions) -> Vec<Arc<dyn DispatchGate>> {
        let mut gates: Vec<Arc<dyn DispatchGate>> = Vec::new();
        gates.push(Arc::new(FoveatedTimeDelayGate {
            time_delay: opts.lod.foveated_time_delay as f64,
        }));
        if opts.streaming.cull_requests_while_moving {
            gates.push(Arc::new(CullWhileMovingGate {
                multiplier: opts.streaming.cull_requests_while_moving_multiplier,
            }));
        }
        gates
    }

    fn default_strategy(opts: &SelectionOptions) -> Arc<dyn TraversalStrategy> {
        if opts.lod.skip_level_of_detail {
            Arc::new(crate::strategy::SkipLodTraversalStrategy)
        } else {
            Arc::new(crate::strategy::DefaultTraversalStrategy)
        }
    }
}

/// External dependencies required by a [`ContentManager`].
#[derive(Clone)]
pub struct LayerExternals {
    pub accessor: Arc<dyn AssetAccessor>,
    pub runtime: AsyncRuntime,
    pub ellipsoid: Ellipsoid,
}

enum State {
    Loading {
        task: Task<Result<TilesetInitResult, TilesetJsonError>>,
    },
    Ready {
        scheduler: LoadScheduler,
        states: TileStates,
        tile_excluders: Vec<Arc<dyn TileExcluder>>,
    },
    Failed,
}

/// A lightweight handle to a [`Layer`] safe to send across threads.
///
/// Obtained via [`Layer::handle`]. Exposes only
/// [`mark_tile_ready`](LayerHandle::mark_tile_ready) so background prepare
/// closures can signal readiness without holding a mutable reference to the
/// layer.
#[derive(Clone)]
pub struct LayerHandle {
    ack_tx: orkester::Sender<TileId>,
}

impl LayerHandle {
    /// Signal that your GPU resources for `tile` are ready.
    ///
    /// Call this from your background prepare closure once geometry has been
    /// uploaded to the GPU. kiban will allow raster overlays to attach to
    /// `tile` starting from the next [`tick`](Layer::tick).
    ///
    /// Non-blocking - sends to an internal channel (`try_send`). Logs a
    /// warning if the channel is full (capacity 4096); silently ignores a
    /// closed channel.
    pub fn mark_tile_ready(&self, tile: TileId) {
        match self.ack_tx.try_send(tile) {
            Ok(()) => {}
            Err(orkester::TrySendError::Full(_)) => {
                log::warn!(
                    "kiban: ack channel full - tile {:?} readiness delayed one frame",
                    tile
                );
            }
            Err(orkester::TrySendError::Closed(_)) => {}
        }
    }
}

/// Private bundle of fire-side event handles.
///
/// Stored on [`Layer`] and never exposed directly - external code receives
/// [`EventListener`]s via the public fields.
struct LayerEvents {
    tile_loaded: Event<TileLoadedArgs>,
    custom_tile_loaded: Event<CustomTileLoadedArgs>,
    tile_evicted: Event<TileId>,
    overlay_ready: Event<OverlayReadyArgs>,
    overlay_detached: Event<(TileId, sovra::OverlayId)>,
    load_progress: Event<(usize, usize)>,
    all_tiles_loaded: Event<()>,
    tile_failed: Event<TileFailedArgs>,
}

impl LayerEvents {
    fn new() -> Self {
        Self {
            tile_loaded: Event::new(),
            custom_tile_loaded: Event::new(),
            tile_evicted: Event::new(),
            overlay_ready: Event::new(),
            overlay_detached: Event::new(),
            load_progress: Event::new(),
            all_tiles_loaded: Event::new(),
            tile_failed: Event::new(),
        }
    }
}

/// A streaming 3D Tiles content manager.
///
/// # Quick start
///
/// ```rust,ignore
/// let mut layer = Layer::new(root, Box::new(loader), vec![], externals, options);
/// layer.overlays_mut().add(Basemap::Osm.into_overlay());
///
/// // Register listeners once - they fire synchronously during tick().
/// let _h = layer.tile_loaded.subscribe({
///     let handle = layer.handle();
///     let bg = pool.context();
///     let store = Arc::clone(&my_store);
///     move |args: &TileLoadedArgs| {
///         let model = Arc::clone(&args.model);
///         let handle = handle.clone();
///         let store = Arc::clone(&store);
///         bg.spawn(move || {
///             let mesh = encode_glb(&model);
///             store.lock().unwrap().insert(args.tile, mesh);
///             handle.mark_tile_ready(args.tile);
///         });
///     }
/// });
///
/// let _h2 = layer.tile_evicted.subscribe({
///     let store = Arc::clone(&my_store);
///     move |tile: &TileId| { store.lock().unwrap().remove(tile); }
/// });
///
/// // per-frame:
/// wq.flush();                          // drain bg->main continuations
/// layer.tick(&mut vg, &views, dt);     // fires events, drives selection
/// render(layer.selected_tiles());
/// ```
pub struct Layer {
    state: State,
    externals: LayerExternals,
    options: LayerOptions,
    overlays: OverlayEngine<TileId>,
    attribution: Option<Arc<str>>,

    /// Tracks which tiles have been acknowledged as GPU-ready via ack_rx.
    selected_set: HashSet<TileId>,

    /// Send side of the ack channel - cloned into [`LayerHandle`].
    ack_tx: orkester::Sender<TileId>,
    /// Receive side of the ack channel - drained each tick before overlays.
    ack_rx: orkester::Receiver<TileId>,

    /// Cached render set from the last `tick()` call.
    selected: Vec<SelectedTile>,
    /// Memory-pressure-adjusted SSE threshold.
    memory_adjusted_sse: f64,

    /// Performance metrics from the last `tick()` call.
    last_frame_metrics: crate::metrics::FrameMetrics,

    /// Memory budget tracker for tile cache eviction and pressure monitoring.
    memory_budget: Arc<crate::memory_budget::MemoryBudget>,

    /// Whether the layer is currently visible.
    ///
    /// When `false` and `LoadingOptions::preload_when_hidden` is also `false`,
    /// `tick()` skips selection and returns immediately with no tiles selected.
    visible: bool,

    /// Speculative camera position hint for preloading flight destinations.
    ///
    /// Set via [`Self::set_preload_hint`].  When
    /// `LoadingOptions::preload_flight_destinations` is `true` and this is
    /// `Some`, callers may run an additional `tick()` pass with this position
    /// as the camera to load destination tiles at background priority.
    preload_hint: Option<DVec3>,

    /// Private fire-side event handles.
    events: LayerEvents,

    /// Fired when a tile's glTF geometry has been loaded from the network.
    ///
    /// Dispatch background work (GLB encoding, collision mesh, etc.) in your
    /// listener, then call [`LayerHandle::mark_tile_ready`] when GPU resources
    /// are ready. kiban keeps the parent tile visible until the ack arrives.
    pub tile_loaded: EventListener<TileLoadedArgs>,

    /// Fired when a tile with custom (non-glTF) content finishes loading.
    pub custom_tile_loaded: EventListener<CustomTileLoadedArgs>,

    /// Fired when a tile was evicted from the streaming cache.
    ///
    /// Free any renderer resources associated with `tile`. The tile may be
    /// reloaded later, in which case [`tile_loaded`](Self::tile_loaded) fires
    /// again.
    pub tile_evicted: EventListener<TileId>,

    /// Fired when a raster overlay tile is ready to be uploaded and attached.
    ///
    /// Only fires after [`LayerHandle::mark_tile_ready`] has been called for
    /// `tile`, so the tile's GPU geometry is guaranteed to exist.
    pub overlay_ready: EventListener<OverlayReadyArgs>,

    /// Fired when a raster overlay was detached from a tile.
    ///
    /// Free any overlay renderer resources for `(tile, overlay_id)`.
    pub overlay_detached: EventListener<(TileId, sovra::OverlayId)>,

    /// Fired every [`tick`](Self::tick) with `(pending_requests, processing)`.
    ///
    /// Use this to drive render-loop redraws - mirrors
    /// `Cesium3DTileset.loadProgress`.
    pub load_progress: EventListener<(usize, usize)>,

    /// Fired when all tiles meeting the current screen-space error are loaded.
    ///
    /// Raised every frame that the load queue is empty. Mirrors
    /// `Cesium3DTileset.allTilesLoaded`.
    pub all_tiles_loaded: EventListener<()>,

    /// Fired when a tile fails to load permanently.
    pub tile_failed: EventListener<TileFailedArgs>,
}

impl Layer {
    /// Create a `ContentManager` from a pre-built loader root.
    ///
    /// Use this for procedural tilesets (e.g. `EllipsoidTilesetLoader`) or
    /// any source that doesn't need an async JSON fetch first.
    pub fn new(
        root: TileDescriptor,
        loader: Arc<dyn ContentLoader>,
        child_loaders: Vec<Arc<dyn ContentLoader>>,
        externals: LayerExternals,
        options: LayerOptions,
    ) -> Self {
        let bg = externals.runtime.background();
        let ov_acc = Arc::clone(&externals.accessor);
        let overlays = OverlayEngine::new(ov_acc, bg);
        let state = Self::build_ready(root, loader, child_loaders, &externals, &options);
        let initial_sse = options.maximum_screen_space_error;
        let (ack_tx, ack_rx) = orkester::mpsc::<TileId>(4096);
        let events = LayerEvents::new();

        let memory_budget = Arc::new(MemoryBudget::new(
            options.selection.loading.max_cached_bytes,
        ));

        Self {
            state,
            externals,
            options,
            overlays,
            attribution: None,
            selected_set: HashSet::new(),
            ack_tx,
            ack_rx,
            selected: Vec::new(),
            memory_adjusted_sse: initial_sse,
            last_frame_metrics: Default::default(),
            memory_budget,
            visible: true,
            preload_hint: None,
            tile_loaded: events.tile_loaded.listener(),
            custom_tile_loaded: events.custom_tile_loaded.listener(),
            tile_evicted: events.tile_evicted.listener(),
            overlay_ready: events.overlay_ready.listener(),
            overlay_detached: events.overlay_detached.listener(),
            load_progress: events.load_progress.listener(),
            all_tiles_loaded: events.all_tiles_loaded.listener(),
            tile_failed: events.tile_failed.listener(),
            events,
        }
    }

    /// Begin loading a tileset from a `tileset.json` URL.
    ///
    /// Emits no events until the root descriptor arrives, then switches to
    /// full streaming automatically.
    pub fn from_url(url: &str, externals: LayerExternals, options: LayerOptions) -> Self {
        let bg = externals.runtime.background();
        let ov_acc = Arc::clone(&externals.accessor);
        let overlays = OverlayEngine::new(ov_acc, bg.clone());
        let task =
            TilesetJsonLoader::create_loader(url, vec![], Arc::clone(&externals.accessor), bg);
        let initial_sse = options.maximum_screen_space_error;
        let (ack_tx, ack_rx) = orkester::mpsc::<TileId>(4096);
        let events = LayerEvents::new();

        let memory_budget = Arc::new(MemoryBudget::new(
            options.selection.loading.max_cached_bytes,
        ));

        Self {
            state: State::Loading { task },
            externals,
            options,
            overlays,
            attribution: None,
            selected_set: HashSet::new(),
            ack_tx,
            ack_rx,
            selected: Vec::new(),
            memory_adjusted_sse: initial_sse,
            last_frame_metrics: Default::default(),
            memory_budget,
            visible: true,
            preload_hint: None,
            tile_loaded: events.tile_loaded.listener(),
            custom_tile_loaded: events.custom_tile_loaded.listener(),
            tile_evicted: events.tile_evicted.listener(),
            overlay_ready: events.overlay_ready.listener(),
            overlay_detached: events.overlay_detached.listener(),
            load_progress: events.load_progress.listener(),
            all_tiles_loaded: events.all_tiles_loaded.listener(),
            tile_failed: events.tile_failed.listener(),
            events,
        }
    }

    pub fn externals(&self) -> &LayerExternals {
        &self.externals
    }
    pub fn externals_mut(&mut self) -> &mut LayerExternals {
        &mut self.externals
    }
    pub fn options(&self) -> &LayerOptions {
        &self.options
    }
    pub fn options_mut(&mut self) -> &mut LayerOptions {
        &mut self.options
    }
    pub fn memory_budget(&self) -> &Arc<MemoryBudget> {
        &self.memory_budget
    }
    pub fn overlays(&self) -> &OverlayEngine<TileId> {
        &self.overlays
    }
    pub fn overlays_mut(&mut self) -> &mut OverlayEngine<TileId> {
        &mut self.overlays
    }
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }
    pub fn is_ready(&self) -> bool {
        matches!(self.state, State::Ready { .. })
    }
    pub fn is_failed(&self) -> bool {
        matches!(self.state, State::Failed)
    }

    /// Set layer visibility.
    ///
    /// When `false` and [`LoadingOptions::preload_when_hidden`] is also
    /// `false`, [`tick`](Self::tick) skips tile selection entirely.
    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    /// Returns `true` if the layer is currently visible.
    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Set a speculative camera position to preload tiles for.
    ///
    /// When [`LoadingOptions::preload_flight_destinations`] is `true` and this
    /// is `Some`, callers can run an additional `tick()` with a synthesised
    /// [`ViewState`](crate::ViewState) at the hint position to load destination
    /// tiles at background priority ahead of a camera flight.
    ///
    /// Pass `None` to clear the hint.
    pub fn set_preload_hint(&mut self, position: Option<DVec3>) {
        self.preload_hint = position;
    }

    /// Returns the current speculative preload position, if any.
    pub fn preload_hint(&self) -> Option<DVec3> {
        self.preload_hint
    }

    pub fn store(&self) -> Option<&TileStore> {
        match &self.state {
            State::Ready { scheduler, .. } => Some(&scheduler.store),
            _ => None,
        }
    }

    pub fn compute_load_progress(&self) -> f32 {
        match &self.state {
            State::Ready { states, .. } => states.compute_load_progress(),
            State::Loading { .. } => 0.0,
            State::Failed => 100.0,
        }
    }

    pub fn unload_all(&mut self) {
        if let State::Ready {
            scheduler, states, ..
        } = &mut self.state
        {
            scheduler.cancel_all();
            states.mark_all_evicted();
        }
        self.selected_set.clear();
    }

    /// Returns a [`LayerHandle`] that can be sent across threads.
    ///
    /// Handles are cheap to clone. Use them in background closures to call
    /// [`LayerHandle::mark_tile_ready`] after GPU upload completes.
    pub fn handle(&self) -> LayerHandle {
        LayerHandle {
            ack_tx: self.ack_tx.clone(),
        }
    }

    /// Signal that your GPU resources for `tile` are ready.
    ///
    /// Convenience wrapper around [`LayerHandle::mark_tile_ready`] for
    /// callers on the main thread that don't need a separate handle.
    pub fn mark_tile_ready(&self, tile: TileId) {
        self.handle().mark_tile_ready(tile);
    }

    /// Retrieve performance metrics from the last [`tick`](Self::tick) call.
    ///
    /// Use this to observe tile selection performance, load queue depth, and
    /// memory usage. Metrics are updated each frame after [`tick`](Self::tick) returns.
    pub fn last_frame_metrics(&self) -> &crate::metrics::FrameMetrics {
        &self.last_frame_metrics
    }

    /// Iterate the render set produced by the last [`tick`](Self::tick) call.
    ///
    /// No content is included - look up your own store with `tile` as the key.
    pub fn selected_tiles(&self) -> impl Iterator<Item = &SelectedTile> + '_ {
        self.selected.iter()
    }

    /// Advance one frame.
    ///
    /// Fires lifecycle events synchronously on the calling thread:
    /// [`tile_loaded`](Self::tile_loaded), [`tile_evicted`](Self::tile_evicted),
    /// [`overlay_ready`](Self::overlay_ready), etc.
    ///
    /// Register listeners before calling `tick()`. After `tick()` returns,
    /// call [`selected_tiles`](Self::selected_tiles) to get the current render
    /// set.
    pub fn tick(&mut self, vg: &mut ViewGroup, views: &[ViewState], delta_time: f32) {
        self.try_transition();

        // Borrow the private event bundle separately from self.state so the
        // borrow checker is happy inside the `if let State::Ready { ... }` arm.
        let ev = &self.events;

        if let State::Ready {
            scheduler,
            states,
            tile_excluders,
        } = &mut self.state
        {
            // Skip selection entirely if hidden and preloading while hidden is disabled.
            if !self.visible && !self.options.selection.loading.preload_when_hidden {
                return;
            }

            // 1. Drain network completions and fire tile_loaded / custom_tile_loaded.
            let load_events = scheduler.drain();
            let tiles_loaded_this_frame = load_events
                .iter()
                .filter(|e| matches!(e, crate::frame_decision::LoadEvent::Loaded { .. }))
                .count();
            let tiles_empty_this_frame = load_events
                .iter()
                .filter(|e| matches!(e, crate::frame_decision::LoadEvent::Empty { .. }))
                .count();
            for hook in &self.options.hooks {
                hook.after_drain(&load_events);
            }
            #[cfg(not(target_arch = "wasm32"))]
            let t_load = std::time::Instant::now();
            apply_load_events(
                load_events,
                states,
                &scheduler.store,
                &ev.tile_loaded,
                &ev.custom_tile_loaded,
                &ev.tile_failed,
            );
            #[cfg(not(target_arch = "wasm32"))]
            let load_processing_time_ms = t_load.elapsed().as_secs_f64() * 1000.0;
            #[cfg(target_arch = "wasm32")]
            let load_processing_time_ms = 0.0_f64;

            // 2. Advance frame + absorb acks. Must run before traversal so
            //    newly-ready tiles enter the render set this frame, and before
            //    overlay update so overlay_ready never fires before geometry
            //    is GPU-ready.
            states.advance_frame();
            while let Ok(tile) = self.ack_rx.try_recv() {
                states.mark_ready(tile);
                self.selected_set.insert(tile);
            }

            // 3. Selection traversal (with memory-pressure SSE adjustment).
            let frame_time_ms = outil::time_now_ms();
            self.memory_adjusted_sse = adjust_memory_sse(
                self.memory_adjusted_sse,
                self.options.maximum_screen_space_error,
                states,
                self.options.eviction_policy.as_ref(),
                &self.options.selection.loading,
            );

            let effective_sse = self.memory_adjusted_sse;
            let mut pending_children: Vec<(TileId, Vec<TileDescriptor>)> = Vec::new();

            #[cfg(not(target_arch = "wasm32"))]
            let t_select = std::time::Instant::now();
            let output = {
                let store_ref = &scheduler.store;
                let loaders_ref = &scheduler.loaders;
                let ellipsoid_ref = &*scheduler.ellipsoid;

                let mut expand_fn = |tile: TileId| -> ExpandResult {
                    let idx = store_ref
                        .effective_loader_index(tile)
                        .map(|li| li.index() as usize)
                        .unwrap_or(0);
                    let loader = match loaders_ref.get(idx) {
                        Some(l) => Arc::clone(l),
                        None => return ExpandResult::None,
                    };
                    match loader.create_children(tile, store_ref, ellipsoid_ref) {
                        TileChildrenResult::Children(descs) => {
                            let to_return = descs.clone();
                            pending_children.push((tile, descs));
                            ExpandResult::Children(to_return)
                        }
                        TileChildrenResult::RetryLater => ExpandResult::RetryLater,
                        TileChildrenResult::None => ExpandResult::None,
                    }
                };

                // Create culling volumes for this frame (can't use buffers.culling_volumes
                // pre-allocation due to borrow conflicts with select_tiles &mut buffers)
                let culling_volumes: Vec<zukei::CullingVolume> = views
                    .iter()
                    .map(crate::traversal::build_culling_volume)
                    .collect();

                let selection_ctx = SelectionContext {
                    store: store_ref,
                    options: &self.options.selection,
                    views,
                    culling_volumes: &culling_volumes,
                    maximum_screen_space_error: effective_sse,
                    excluders: tile_excluders.as_slice(),
                    scorer: self.options.scorer.as_ref(),
                    strategy: self.options.strategy.as_ref(),
                    eviction_policy: self.options.eviction_policy.as_ref(),
                    frame_time_ms,
                    occlusion_proxy: self.options.occlusion_proxy.clone(),
                };

                select_tiles(&selection_ctx, states, &mut vg.buffers, &mut expand_fn)
            };
            #[cfg(not(target_arch = "wasm32"))]
            let selection_time_ms = t_select.elapsed().as_secs_f64() * 1000.0;
            #[cfg(target_arch = "wasm32")]
            let selection_time_ms = 0.0_f64;

            for hook in &self.options.hooks {
                hook.after_select(&output);
            }

            // Capture traversal stats before `output` is consumed by commit_result.
            let nodes_visited = output.nodes_visited;
            let nodes_culled = output.nodes_culled;
            let nodes_refined = output.nodes_refined;

            // 4. Apply pending children + overlay update.
            for (tile, descs) in pending_children {
                scheduler.store.insert_children(tile, &descs);
            }
            let frame_number = states.frame_index;

            // Reuse pre-allocated buffer for ready tiles
            vg.buffers.ready_tiles.clear();
            vg.buffers.ready_tiles.extend(
                output
                    .selected
                    .iter()
                    .map(|&n| (n, scheduler.store.geometric_error(n))),
            );
            // Propagate camera state to the overlay engine so it can pick
            // the correct zoom level for each geometry tile.
            if let Some(view) = views.first() {
                let sse_denominator = match view.projection {
                    Projection::Perspective { fov_y, .. } => 2.0 * (fov_y / 2.0).tan(),
                    Projection::Orthographic { half_height, .. } => 2.0 * half_height,
                };
                self.overlays.set_view_info(OverlayViewInfo {
                    viewport_height: view.viewport_px[1] as f64,
                    sse_denominator,
                    maximum_screen_space_error: effective_sse,
                });
            }
            self.overlays
                .update(&vg.buffers.ready_tiles, &scheduler.store);
            translate_overlay_events(
                &mut self.overlays,
                &self.selected_set,
                &ev.overlay_ready,
                &ev.overlay_detached,
            );

            // 5. Commit traversal, dispatch loads, evict stale tiles.
            vg.commit_result(
                output,
                frame_number,
                delta_time,
                &self.options.selection.streaming,
                states,
                self.memory_adjusted_sse,
            );

            let before_dispatch = scheduler.in_flight_count();
            #[cfg(not(target_arch = "wasm32"))]
            let t_dispatch = std::time::Instant::now();
            scheduler.dispatch(
                &vg.load_queue,
                states,
                views,
                delta_time,
                &self.options.selection,
                effective_sse,
                &self.options.gates,
            );
            #[cfg(not(target_arch = "wasm32"))]
            let load_dispatch_time_ms = t_dispatch.elapsed().as_secs_f64() * 1000.0;
            #[cfg(target_arch = "wasm32")]
            let load_dispatch_time_ms = 0.0_f64;
            let dispatched = scheduler.in_flight_count().saturating_sub(before_dispatch);
            for hook in &self.options.hooks {
                hook.after_dispatch(dispatched);
            }

            let tiles_evicted_this_frame = evict_stale_tiles(
                states,
                &mut self.selected_set,
                self.options.tile_cache_unload_time_limit_secs,
                &ev.tile_evicted,
            );

            // 6. Rebuild render set.
            rebuild_ready_set(&mut self.selected, vg, &scheduler.store);

            // 7. Fire load_progress / all_tiles_loaded.
            let (loading, processing, failed, resident_bytes) = states.load_stats();

            // Update memory budget with current resident bytes (for pressure tracking)
            self.memory_budget.update_usage(resident_bytes);

            ev.load_progress
                .raise((loading as usize, processing as usize));
            if loading == 0 && processing == 0 {
                ev.all_tiles_loaded.raise(());
            }

            // 8. Update frame metrics.
            self.last_frame_metrics = crate::metrics::FrameMetrics {
                selection_time_ms,
                load_processing_time_ms,
                load_dispatch_time_ms,
                load_queue_size: vg.load_queue.len(),
                tiles_loaded_this_frame,
                tiles_evicted_this_frame,
                selected_tile_count: self.selected.len(),
                estimated_memory_bytes: resident_bytes as u64,
                tiles_visited: nodes_visited,
                tiles_culled: nodes_culled,
                tiles_failed: failed as usize,
                tiles_empty: tiles_empty_this_frame,
                tiles_in_flight: scheduler.in_flight_count(),
                tiles_refined: nodes_refined,
            };
        }

        for hook in &self.options.hooks {
            hook.after_tick();
        }
    }

    fn try_transition(&mut self) {
        let loading = match &mut self.state {
            State::Loading { task } => match task.poll_ready() {
                Some(result) => result,
                None => return,
            },
            _ => return,
        };

        self.state = match loading {
            Ok(Ok(init)) => {
                self.attribution = init.attribution.clone();
                Self::build_ready_from_init(init, &self.externals, &self.options)
            }
            _ => State::Failed,
        };
    }

    fn build_ready(
        root: TileDescriptor,
        loader: Arc<dyn ContentLoader>,
        child_loaders: Vec<Arc<dyn ContentLoader>>,
        externals: &LayerExternals,
        options: &LayerOptions,
    ) -> State {
        let store = TileStore::from_descriptor(root);
        let mut loaders: Vec<Arc<dyn ContentLoader>> = vec![loader];
        for cl in child_loaders {
            loaders.push(cl);
        }
        let scheduler = LoadScheduler::new(
            store,
            loaders,
            Arc::clone(&externals.accessor),
            vec![],
            externals.ellipsoid.clone(),
            externals.runtime.clone(),
            options.selection.loading.result_channel_capacity,
        );
        State::Ready {
            scheduler,
            states: TileStates::new(),
            tile_excluders: options.tile_excluders.clone(),
        }
    }

    fn build_ready_from_init(
        init: TilesetInitResult,
        externals: &LayerExternals,
        options: &LayerOptions,
    ) -> State {
        Self::build_ready(
            init.root,
            init.loader,
            init.child_loaders,
            externals,
            options,
        )
    }
}

/// Apply scheduler load events to selection state and translate the content-
/// carrying variants into [`TileEvent`]s. Records each loaded tile's resident
/// byte cost so the eviction policy sees real numbers.
fn apply_load_events(
    load_events: Vec<LoadEvent>,
    states: &mut TileStates,
    store: &TileStore,
    tile_loaded: &Event<TileLoadedArgs>,
    custom_tile_loaded: &Event<CustomTileLoadedArgs>,
    tile_failed: &Event<TileFailedArgs>,
) {
    for event in &load_events {
        match event {
            LoadEvent::Loaded { .. } => {} // handled below so we can consume model
            other => states.apply(other),
        }
    }
    for event in load_events {
        match event {
            LoadEvent::Loaded { tile, model } => {
                states.get_mut(tile).content_byte_size =
                    model.resident_byte_size().try_into().unwrap_or(u32::MAX);
                let transform = store.world_transform(tile);
                tile_loaded.raise(TileLoadedArgs {
                    tile,
                    model: std::sync::Arc::new(model),
                    transform,
                });
            }
            LoadEvent::Custom { tile, content } => {
                let transform = store.world_transform(tile);
                custom_tile_loaded.raise(CustomTileLoadedArgs {
                    tile,
                    content,
                    transform,
                });
            }
            LoadEvent::Failed { tile, url, message } => {
                tile_failed.raise(TileFailedArgs {
                    tile,
                    url: url.map(|s| s.into()),
                    message: message.into(),
                });
            }
            _ => {}
        }
    }
}

/// Ratchet the memory-pressure SSE threshold based on current resident bytes.
/// Delegated to the eviction policy so callers can override or disable it.
fn adjust_memory_sse(
    current: f64,
    nominal: f64,
    states: &TileStates,
    policy: &dyn EvictionPolicy,
    loading: &crate::options::LoadingOptions,
) -> f64 {
    let (_, _, _, resident_bytes) = states.load_stats();
    policy
        .adjust_sse(
            current,
            resident_bytes,
            loading.max_cached_bytes,
            loading.maximum_cache_overflow_bytes,
        )
        .max(nominal)
}

/// Drain overlay attach/detach events and translate them into [`TileEvent`]s.
/// `OverlayLoaded` is gated on `selected_set` so it never fires before the
/// tile's content has been acknowledged via [`Layer::mark_tile_ready`].
fn translate_overlay_events(
    overlays: &mut OverlayEngine<TileId>,
    selected_set: &HashSet<TileId>,
    overlay_ready: &Event<OverlayReadyArgs>,
    overlay_detached: &Event<(TileId, sovra::OverlayId)>,
) {
    for ov_event in overlays.drain_events() {
        match ov_event {
            OverlayEvent::Attached {
                tile_id: tile,
                overlay_id,
                uv_index,
                tile: raster_tile,
            } => {
                if selected_set.contains(&tile) {
                    overlay_ready.raise(OverlayReadyArgs {
                        tile,
                        overlay_id,
                        uv_index,
                        overlay: raster_tile,
                    });
                }
            }
            OverlayEvent::Detached {
                tile_id: tile,
                overlay_id,
            } => {
                overlay_detached.raise((tile, overlay_id));
            }
            _ => {}
        }
    }
}

/// Evict renderable tiles that haven't been touched for `time_limit_secs`.
/// Does nothing when `time_limit_secs <= 0.0`. Returns the number of evictions.
fn evict_stale_tiles(
    states: &mut TileStates,
    selected_set: &mut HashSet<TileId>,
    time_limit_secs: f32,
    tile_evicted: &Event<TileId>,
) -> usize {
    if time_limit_secs <= 0.0 {
        return 0;
    }
    let current_secs = outil::time_now_secs();
    let grace = time_limit_secs as u64;
    let stale: Vec<TileId> = states.stale(current_secs, grace).collect();
    let count = stale.len();
    for tile in stale {
        states.mark_evicted(tile);
        selected_set.remove(&tile);
        tile_evicted.raise(tile);
    }
    count
}

/// Rebuild the ready set from the view-group result, applying fade alphas
/// from the fade strategy.
fn rebuild_ready_set(selected: &mut Vec<SelectedTile>, vg: &ViewGroup, store: &TileStore) {
    let result = vg.view_update_result();
    let fading_set: HashSet<TileId> = result.tiles_fading_out.iter().copied().collect();

    selected.clear();

    for &tile in result.tiles_fading_out.iter() {
        let pct = result
            .tile_fade_percentages
            .get(&tile)
            .copied()
            .unwrap_or(0.0);
        let alpha = vg.fade_strategy().fade_out_alpha(pct);
        selected.push(SelectedTile {
            tile,
            transform: store.world_transform(tile),
            alpha,
            selection_depth: 0,
            final_resolution: true,
        });
    }

    for (i, &tile) in result.selected_tiles.iter().enumerate() {
        if fading_set.contains(&tile) {
            continue;
        }
        let alpha = result
            .tile_fade_percentages
            .get(&tile)
            .copied()
            .map(|pct| vg.fade_strategy().fade_in_alpha(pct))
            .unwrap_or(1.0);
        selected.push(SelectedTile {
            tile,
            transform: store.world_transform(tile),
            alpha,
            selection_depth: result.tile_selection_depths.get(i).copied().unwrap_or(0),
            final_resolution: result
                .tile_final_resolutions
                .get(i)
                .copied()
                .unwrap_or(true),
        });
    }
}

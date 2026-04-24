//! Standalone raster overlay engine.
//!
//! [`OverlayEngine`] manages raster overlay lifecycle independently from
//! any tile selection engine. Each frame the caller passes a set of visible
//! tile ids and a hierarchy reference; the engine fetches, composites, and
//! caches overlay tiles for those tiles.
//!
//! - Each (geometry-tile, overlay) pair has at most one **ready** raster tile
//!   currently attached and optionally one **loading** higher-resolution tile.
//! - When the loading tile finishes, it replaces the ready tile (detach old,
//!   attach new).
//! - Tile overlay state is **not** destroyed the instant a geometry tile leaves
//!   the render set; it survives for one additional frame so that brief
//!   flickering in the selection engine doesn't cause jarring attach/detach
//!   cycles.

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::Task;

use crate::credit::Credit;
use crate::event::OverlayEvent;
use crate::hierarchy::OverlayHierarchy;
use crate::overlay::{
    OverlayCollection, OverlayId, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
};
use orkester::{Event, EventListener};

/// Default texel density (texels per radian) used when choosing overlay zoom levels.
pub const DEFAULT_TARGET_TEXELS_PER_RADIAN: f64 = 256.0 / (std::f64::consts::PI / 4.0);

/// Options for constructing an [`OverlayEngine`].
pub struct OverlayEngineOptions {
    /// Maximum concurrent tile fetch tasks per engine. Default: 20.
    pub max_simultaneous_requests: usize,
    /// Maximum composited output texture size in either dimension. Default: 2048.
    pub maximum_texture_size: u32,
    /// LRU sub-tile cache capacity (number of individual sub-tiles). Default: 256.
    pub tile_cache_capacity: usize,
}

impl Default for OverlayEngineOptions {
    fn default() -> Self {
        Self {
            max_simultaneous_requests: 20,
            maximum_texture_size: 2048,
            tile_cache_capacity: 256,
        }
    }
}

/// Size-bounded LRU cache for overlay sub-tiles.
///
/// Keyed by `(overlay_id, level, x, y)`.  On capacity overflow the
/// least-recently-used entry is evicted.
struct SubTileCache<K: Eq + Hash> {
    entries: HashMap<K, Arc<RasterOverlayTile>>,
    order: std::collections::VecDeque<K>,
    capacity: usize,
}

impl<K: Eq + Hash + Clone> SubTileCache<K> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: std::collections::VecDeque::new(),
            capacity,
        }
    }

    fn get(&mut self, key: &K) -> Option<Arc<RasterOverlayTile>> {
        if let Some(tile) = self.entries.get(key) {
            // Move to the back (most-recently-used position).
            if let Some(pos) = self.order.iter().position(|k| k == key) {
                self.order.remove(pos);
                self.order.push_back(key.clone());
            }
            Some(Arc::clone(tile))
        } else {
            None
        }
    }

    fn insert(&mut self, key: K, tile: Arc<RasterOverlayTile>) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), tile);
            if let Some(pos) = self.order.iter().position(|k| k == &key) {
                self.order.remove(pos);
                self.order.push_back(key);
            }
            return;
        }
        if self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, tile);
    }
}

/// Per-tile info passed to the overlay engine each frame.
#[derive(Clone, Copy, Debug)]
pub struct OverlayTileInfo<T> {
    pub tile_id: T,
    /// Geometric error of this tile (metres).  Smaller = more detailed.
    pub geometric_error: f64,
}

/// Viewport / projection info needed to compute per-tile overlay resolution.
#[derive(Clone, Copy, Debug)]
pub struct OverlayViewInfo {
    /// Viewport height in pixels.
    pub viewport_height: f64,
    /// SSE denominator = `2 * tan(fov_y / 2)` for perspective.
    /// For orthographic: `2 * half_height`.
    pub sse_denominator: f64,
    /// Maximum screen-space error threshold (pixels).  Default 16.
    pub maximum_screen_space_error: f64,
}

impl Default for OverlayViewInfo {
    fn default() -> Self {
        Self {
            viewport_height: 768.0,
            sse_denominator: 2.0 * (std::f64::consts::FRAC_PI_4).tan(), // 45 degree fov
            maximum_screen_space_error: 16.0,
        }
    }
}

enum ProviderState {
    /// Provider is still being constructed (e.g. WMS `GetCapabilities` in flight).
    Pending(Task<Arc<dyn RasterOverlayTileProvider>>),
    /// Provider is live and can serve tiles.
    Active(Arc<dyn RasterOverlayTileProvider>),
    /// Provider construction failed; this slot is inert until removed.
    ///
    /// Using a dedicated variant (instead of a stub no-op provider sentinel)
    /// means every code path that iterates providers must acknowledge the
    /// failure case explicitly, which keeps failures from being silently
    /// masked by a `.get_tile()` that returns nothing.
    Failed,
}

/// A raster tile currently attached to a geometry tile.
struct AttachedTile {
    tile: RasterOverlayTile,
    /// Zoom level of this tile (so we can detect when an upgrade is needed).
    level: u32,
}

/// An in-flight fetch group whose composite will replace `attached`.
struct LoadingTiles {
    /// Zoom level being loaded (all tasks below are at this level).
    level: u32,
    /// In-flight fetch tasks keyed by `(x, y, level)`.
    tasks: HashMap<(u32, u32, u32), Task<Result<RasterOverlayTile, TileFetchError>>>,
    /// Sub-tiles already satisfied from the LRU cache (no network fetch needed).
    cached_tiles: Vec<RasterOverlayTile>,
}

/// State for ONE overlay on ONE geometry tile.
///
/// The two fields are orthogonal: `attached` tracks what is currently shown
/// (possibly inherited from an ancestor), and `loading` tracks an in-flight
/// upgrade. All four `(None|Some) x (None|Some)` combinations are meaningful:
/// * both `None`: just created, nothing dispatched yet.
/// * `attached=None, loading=Some`: initial fetch in progress.
/// * `attached=Some, loading=Some`: showing a placeholder while upgrading.
/// * `attached=Some, loading=None`: best-available tile attached, done.
#[derive(Default)]
struct MappedOverlay {
    attached: Option<AttachedTile>,
    loading: Option<LoadingTiles>,
}

/// All overlay state for a single geometry tile.
#[derive(Default)]
struct TileOverlayState {
    /// Per-overlay attachment state.
    overlays: HashMap<OverlayId, MappedOverlay>,
}

pub struct OverlayEngine<T: Copy + Eq + Hash + Send + Sync + Debug + 'static> {
    accessor: Arc<dyn AssetAccessor>,
    ctx: orkester::Context,
    collection: OverlayCollection,
    providers: HashMap<OverlayId, ProviderState>,
    tile_state: HashMap<T, TileOverlayState>,
    /// Tiles ready in the *previous* frame.
    prev_ready: HashSet<T>,
    /// Tiles ready two frames ago - used for deferred cleanup.
    prev_prev_ready: HashSet<T>,
    overlay_order: Vec<OverlayId>,
    /// Fast lookup for overlay index in overlay_order (O(1) instead of O(n))
    overlay_index_map: HashMap<OverlayId, u32>,
    target_texels_per_radian: f64,
    view_info: OverlayViewInfo,
    events: Vec<OverlayEvent<T>>,

    /// Semaphore limiting the number of concurrent tile fetch tasks.
    request_semaphore: orkester::Semaphore,
    /// LRU cache for individual overlay sub-tiles.
    tile_cache: SubTileCache<(OverlayId, u32, u32, u32)>,
    /// Maximum output texture dimension when compositing overlay tiles.
    maximum_texture_size: u32,

    // Fire-side handles (private - only OverlayEngine calls raise).
    ev_overlay_attached: Event<(T, OverlayId, u32, RasterOverlayTile)>,
    ev_overlay_detached: Event<(T, OverlayId)>,

    /// Fired when an overlay tile is attached to a geometry tile.
    ///
    /// Args: `(tile_id, overlay_id, uv_index, raster_tile)`.
    pub overlay_attached: EventListener<(T, OverlayId, u32, RasterOverlayTile)>,

    /// Fired when an overlay is detached from a tile (tile left the view or
    /// overlay was removed).
    ///
    /// Args: `(tile_id, overlay_id)`.
    pub overlay_detached: EventListener<(T, OverlayId)>,
}

impl<T: Copy + Eq + Hash + Send + Sync + Debug + 'static> OverlayEngine<T> {
    pub fn new(accessor: Arc<dyn AssetAccessor>, ctx: orkester::Context) -> Self {
        Self::new_with_options(accessor, ctx, OverlayEngineOptions::default())
    }

    /// Construct with explicit engine options.
    pub fn new_with_options(
        accessor: Arc<dyn AssetAccessor>,
        ctx: orkester::Context,
        options: OverlayEngineOptions,
    ) -> Self {
        let ev_overlay_attached: Event<(T, OverlayId, u32, RasterOverlayTile)> = Event::new();
        let ev_overlay_detached: Event<(T, OverlayId)> = Event::new();
        Self {
            accessor,
            ctx,
            collection: OverlayCollection::new(),
            providers: HashMap::new(),
            tile_state: HashMap::new(),
            prev_ready: HashSet::new(),
            prev_prev_ready: HashSet::new(),
            overlay_order: Vec::new(),
            overlay_index_map: HashMap::new(),
            target_texels_per_radian: DEFAULT_TARGET_TEXELS_PER_RADIAN,
            view_info: OverlayViewInfo::default(),
            events: Vec::new(),
            request_semaphore: orkester::Semaphore::new(options.max_simultaneous_requests),
            tile_cache: SubTileCache::new(options.tile_cache_capacity),
            maximum_texture_size: options.maximum_texture_size,
            overlay_attached: ev_overlay_attached.listener(),
            overlay_detached: ev_overlay_detached.listener(),
            ev_overlay_attached,
            ev_overlay_detached,
        }
    }

    /// Collect credits from all active overlay providers.
    pub fn credits(&self) -> Vec<Credit> {
        self.providers
            .values()
            .filter_map(|state| {
                if let ProviderState::Active(p) = state {
                    Some(p.credits())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    pub fn set_target_texels_per_radian(&mut self, v: f64) {
        self.target_texels_per_radian = v;
    }

    pub fn set_view_info(&mut self, info: OverlayViewInfo) {
        self.view_info = info;
    }

    pub fn add(&mut self, overlay: impl crate::overlay::RasterOverlay + 'static) -> OverlayId {
        let id = self.collection.add(overlay);
        if let Some((_, raw)) = self.collection.iter().find(|(oid, _)| *oid == id) {
            let task = raw.create_tile_provider(&self.ctx, &self.accessor);
            self.providers.insert(id, ProviderState::Pending(task));
        }
        let index = self.overlay_order.len() as u32;
        self.overlay_order.push(id);
        self.overlay_index_map.insert(id, index);
        id
    }

    pub fn remove(&mut self, id: OverlayId) {
        self.collection.remove(id);
        self.providers.remove(&id);
        for (&tile_id, state) in &mut self.tile_state {
            if state.overlays.remove(&id).is_some() {
                self.events.push(OverlayEvent::Detached {
                    tile_id,
                    overlay_id: id,
                });
                self.ev_overlay_detached.raise((tile_id, id));
            }
        }
        // Find removal position before modifying the Vec, so we know which
        // indices shifted.  Overlays are few (typically < 10) so this is fine.
        let removed_pos = self.overlay_order.iter().position(|&oid| oid == id);
        self.overlay_order.retain(|&oid| oid != id);
        self.overlay_index_map.remove(&id);
        // Only update entries at positions >= removed_pos; earlier entries are
        // unaffected.  For removal of the last overlay this loop is O(0).
        if let Some(pos) = removed_pos {
            for i in pos..self.overlay_order.len() {
                let oid = self.overlay_order[i];
                self.overlay_index_map.insert(oid, i as u32);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.overlay_order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.overlay_order.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (OverlayId, &dyn crate::overlay::RasterOverlay)> {
        self.collection.iter()
    }

    pub fn for_tile(&self, tile_id: T) -> Vec<(u32, &RasterOverlayTile)> {
        self.tile_state
            .get(&tile_id)
            .map(|state| {
                state
                    .overlays
                    .iter()
                    .filter_map(|(oid, mapped)| {
                        let uv = overlay_order_index(&self.overlay_index_map, *oid) as u32;
                        mapped.attached.as_ref().map(|a| (uv, &a.tile))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns `true` if all active overlays have been attached to this tile.
    ///
    /// When overlays are registered, tiles should not be ready until their
    /// overlays are ready - the parent tile should stay visible instead.
    pub fn is_tile_ready(&self, tile_id: T) -> bool {
        // If no overlays registered, everything is ready.
        if self.overlay_order.is_empty() {
            return true;
        }
        // If no active providers yet, nothing can be ready.
        let active_count = self
            .providers
            .values()
            .filter(|p| matches!(p, ProviderState::Active(_)))
            .count();
        if active_count == 0 {
            return false;
        }
        let state = match self.tile_state.get(&tile_id) {
            Some(s) => s,
            None => return false,
        };
        // Check that every active overlay has an attached tile for this tile.
        for (&overlay_id, provider_state) in &self.providers {
            if !matches!(provider_state, ProviderState::Active(_)) {
                continue;
            }
            match state.overlays.get(&overlay_id) {
                Some(mapped) if mapped.attached.is_some() => {}
                _ => return false,
            }
        }
        true
    }

    /// Run one frame of overlay processing.
    ///
    /// Each entry in `tiles` is `(tile_id, geometric_error)`. The geometric
    /// error is used together with `maximum_screen_space_error` to compute
    /// per-tile overlay texel density.
    pub fn update(&mut self, tiles: &[(T, f64)], hierarchy: &dyn OverlayHierarchy<T>) {
        // 1. Promote pending providers.
        self.drain_pending_providers();

        let current: HashSet<T> = tiles.iter().map(|(id, _)| *id).collect();

        // 2. Deferred cleanup: tiles gone for TWO consecutive frames get purged.
        //    This prevents flicker from one-frame selection jitter.
        let stale: Vec<T> = self
            .tile_state
            .keys()
            .copied()
            .filter(|id| !current.contains(id) && !self.prev_ready.contains(id))
            .collect();
        for tile_id in stale {
            if let Some(state) = self.tile_state.remove(&tile_id) {
                for (&overlay_id, mapped) in &state.overlays {
                    if mapped.attached.is_some() {
                        self.events.push(OverlayEvent::Detached {
                            tile_id,
                            overlay_id,
                        });
                        self.ev_overlay_detached.raise((tile_id, overlay_id));
                    }
                }
            }
        }

        // 3. For visible tiles, dispatch fetches and process completions.
        for &(tile_id, geometric_error) in tiles {
            self.dispatch_for_tile(tile_id, geometric_error, hierarchy);
        }
        self.process_completions();

        // 4. Rotate frame sets.
        self.prev_prev_ready = std::mem::take(&mut self.prev_ready);
        self.prev_ready = current;
    }

    pub fn collection(&self) -> &OverlayCollection {
        &self.collection
    }

    pub fn drain_events(&mut self) -> Vec<OverlayEvent<T>> {
        std::mem::take(&mut self.events)
    }

    fn drain_pending_providers(&mut self) {
        let ready_ids: Vec<OverlayId> = self
            .providers
            .iter()
            .filter_map(|(&id, state)| match state {
                ProviderState::Pending(task) if task.is_ready() => Some(id),
                _ => None,
            })
            .collect();
        for id in ready_ids {
            if let Some(ProviderState::Pending(task)) = self.providers.remove(&id) {
                match task.block() {
                    Ok(provider) => {
                        self.providers.insert(id, ProviderState::Active(provider));
                    }
                    Err(err) => {
                        log::warn!(
                            "overlay provider initialisation failed (overlay={id:?}): {err}"
                        );
                        self.providers.insert(id, ProviderState::Failed);
                    }
                }
            }
        }
    }

    /// For a single geometry tile, ensure each active overlay has fetches in flight.
    /// If the tile has no overlay yet but a parent does, immediately inherit the
    /// parent's overlay (CesiumJS "upsampledFromParent" pattern) so tiles are
    /// never ready without some texture.
    fn dispatch_for_tile(
        &mut self,
        tile_id: T,
        geometric_error: f64,
        hierarchy: &dyn OverlayHierarchy<T>,
    ) {
        let geo_rect = match hierarchy.globe_rectangle(tile_id) {
            Some(r) => r,
            None => return,
        };

        // Before mutating tile_state, collect parent overlay data for any
        // overlay that this tile doesn't have yet.
        let mut parent_overlays: HashMap<OverlayId, RasterOverlayTile> = HashMap::new();
        if !self.tile_state.contains_key(&tile_id)
            || self.tile_state.get(&tile_id).map_or(false, |s| {
                self.overlay_order
                    .iter()
                    .any(|oid| !s.overlays.contains_key(oid))
            })
        {
            // Walk up hierarchy to find closest ancestor with ready overlays.
            let mut ancestor = hierarchy.parent(tile_id);
            while let Some(pid) = ancestor {
                if let Some(parent_state) = self.tile_state.get(&pid) {
                    for oid in &self.overlay_order {
                        if !parent_overlays.contains_key(oid) {
                            if let Some(pm) = parent_state.overlays.get(oid) {
                                if let Some(ref a) = pm.attached {
                                    parent_overlays.insert(*oid, a.tile.clone());
                                }
                            }
                        }
                    }
                    // If we found all overlays, stop walking.
                    if parent_overlays.len() == self.overlay_order.len() {
                        break;
                    }
                }
                ancestor = hierarchy.parent(pid);
            }
        }

        // Compute the per-tile target screen-pixel size for the overlay
        // texture.
        //
        //   diameters = on-ellipsoid size of the projected rectangle (metres)
        //   target_pixels = diameters * sse / geometric_error
        //
        // Larger geometric_error -> tile is far away -> fewer pixels needed.
        // Smaller error (refining) -> more pixels needed -> higher overlay level.
        const EARTH_RADIUS_M: f64 = 6_378_137.0;
        let mid_lat = 0.5 * (geo_rect.south + geo_rect.north);
        let lon_span = if geo_rect.east >= geo_rect.west {
            geo_rect.east - geo_rect.west
        } else {
            geo_rect.east - geo_rect.west + std::f64::consts::TAU
        };
        let lat_span = (geo_rect.north - geo_rect.south).abs();
        let diameter_x_m = lon_span.abs() * mid_lat.cos() * EARTH_RADIUS_M;
        let diameter_y_m = lat_span * EARTH_RADIUS_M;

        let view = self.view_info;
        // Avoid division-by-zero: an effectively-leaf tile (geom_error ~ 0)
        // should request maximum detail; clamp to a small floor.
        let safe_geom_error = geometric_error.max(1.0e-3);
        let scale = view.maximum_screen_space_error / safe_geom_error;
        let target = glam::DVec2::new(diameter_x_m * scale, diameter_y_m * scale);
        let state = self.tile_state.entry(tile_id).or_default();

        for (&overlay_id, provider_state) in &self.providers {
            let provider = match provider_state {
                ProviderState::Active(p) => Arc::clone(p),
                ProviderState::Pending(_) | ProviderState::Failed => continue,
            };

            let is_new = !state.overlays.contains_key(&overlay_id);

            let mapped = state.overlays.entry(overlay_id).or_default();

            // Inherit parent's overlay immediately if this is a new entry.
            if is_new && mapped.attached.is_none() {
                if let Some(parent_tile) = parent_overlays.remove(&overlay_id) {
                    mapped.attached = Some(AttachedTile {
                        tile: parent_tile.clone(),
                        level: 0, // parent level - will be upgraded
                    });

                    let uv_index = overlay_order_index(&self.overlay_index_map, overlay_id);
                    self.events.push(OverlayEvent::Attached {
                        tile_id,
                        overlay_id,
                        uv_index,
                        tile: parent_tile.clone(),
                    });
                    self.ev_overlay_attached
                        .raise((tile_id, overlay_id, uv_index, parent_tile));
                }
            }

            // If already fully attached and no loading in progress, check if
            // higher resolution is available.
            let tile_coords = provider.tiles_for_extent(geo_rect, target);
            if tile_coords.is_empty() {
                continue;
            }
            let target_level = tile_coords[0].2;

            // Already loading this level or better? Skip.
            if let Some(ref loading) = mapped.loading {
                if loading.level >= target_level {
                    continue;
                }
            }
            // Already attached at this level (no pending upgrade)? Skip.
            if mapped.loading.is_none() {
                if let Some(ref a) = mapped.attached {
                    if a.level >= target_level {
                        continue;
                    }
                }
            }

            // Dispatch fetches for all coords at the target level (replaces any
            // prior in-flight set we're superseding).
            log::debug!(
                "overlay tile={:?} ge={:.1} target_tpr={:.1} -> level {} ({} tiles)",
                tile_id,
                geometric_error,
                target,
                target_level,
                tile_coords.len(),
            );
            let mut tasks = HashMap::with_capacity(tile_coords.len());
            let mut cached_tiles = Vec::new();
            for (x, y, level) in tile_coords {
                let cache_key = (overlay_id, level, x, y);
                if let Some(cached) = self.tile_cache.get(&cache_key) {
                    cached_tiles.push((*cached).clone());
                } else {
                    let task = provider
                        .get_tile(x, y, level)
                        .with_semaphore(&self.request_semaphore);
                    tasks.insert((x, y, level), task);
                }
            }
            mapped.loading = Some(LoadingTiles {
                level: target_level,
                tasks,
                cached_tiles,
            });
        }
    }

    /// Check all in-flight fetch groups. When ALL fetches for a (tile, overlay)
    /// are done, composite into one tile and emit an Attached event.
    fn process_completions(&mut self) {
        let overlay_index_map = &self.overlay_index_map;
        let maximum_texture_size = self.maximum_texture_size;

        // Collect (tile, overlay) pairs whose loading group is fully ready.
        let ready_pairs: Vec<(T, OverlayId)> = self
            .tile_state
            .iter()
            .flat_map(|(&tile_id, state)| {
                state
                    .overlays
                    .iter()
                    .filter(|(_, mapped)| {
                        mapped.loading.as_ref().map_or(false, |l| {
                            l.tasks.values().all(|t| t.is_ready())
                                && (!l.tasks.is_empty() || !l.cached_tiles.is_empty())
                        })
                    })
                    .map(move |(&overlay_id, _)| (tile_id, overlay_id))
            })
            .collect();

        for (tile_id, overlay_id) in ready_pairs {
            let state = match self.tile_state.get_mut(&tile_id) {
                Some(s) => s,
                None => continue,
            };
            let mapped = match state.overlays.get_mut(&overlay_id) {
                Some(m) => m,
                None => continue,
            };

            // Drain all loading tasks. Sort by (level, y, x) before compositing
            // so that blit order is deterministic - iterating the HashMap
            // directly produced frame-to-frame flicker when tiles overlap.
            let loading = match mapped.loading.take() {
                Some(l) => l,
                None => continue,
            };
            let mut ordered: Vec<(
                (u32, u32, u32),
                Task<Result<RasterOverlayTile, TileFetchError>>,
            )> = loading.tasks.into_iter().collect();
            ordered.sort_by_key(|((x, y, level), _)| (*level, *y, *x));

            // Collect all task results first for all-or-nothing semantics.
            let results: Vec<_> = ordered
                .into_iter()
                .map(|(coords, task)| (coords, task.block()))
                .collect();

            // If any sub-tile failed, skip compositing and retry next frame.
            // Keep the currently attached (parent/inherited) tile as-is.
            if results.iter().any(|(_, r)| !matches!(r, Ok(Ok(_)))) {
                log::debug!(
                    "overlay composite skipped (tile={tile_id:?}, overlay={overlay_id:?}): \
                     one or more sub-tile fetches failed; will retry next frame"
                );
                continue;
            }

            let mut tiles: Vec<RasterOverlayTile> = Vec::with_capacity(results.len());
            for ((x, y, level), result) in results {
                if let Ok(Ok(tile)) = result {
                    let cache_key = (overlay_id, level, x, y);
                    self.tile_cache.insert(cache_key, Arc::new(tile.clone()));
                    tiles.push(tile);
                }
            }
            // Incorporate tiles that were satisfied from the LRU cache.
            tiles.extend(loading.cached_tiles);

            if tiles.is_empty() {
                continue;
            }

            // Composite into a single tile.
            let composite = if tiles.len() == 1 {
                tiles.into_iter().next().unwrap()
            } else {
                let mut west = f64::MAX;
                let mut south = f64::MAX;
                let mut east = f64::MIN;
                let mut north = f64::MIN;
                for t in &tiles {
                    west = west.min(t.rectangle.west);
                    south = south.min(t.rectangle.south);
                    east = east.max(t.rectangle.east);
                    north = north.max(t.rectangle.north);
                }
                let target_rect = terra::GlobeRectangle::new(west, south, east, north);
                let cols = ((east - west) / (tiles[0].rectangle.east - tiles[0].rectangle.west))
                    .ceil() as u32;
                let rows = ((north - south) / (tiles[0].rectangle.north - tiles[0].rectangle.south))
                    .ceil() as u32;
                let target_w = (cols * tiles[0].width).min(maximum_texture_size);
                let target_h = (rows * tiles[0].height).min(maximum_texture_size);
                crate::compositing::composite_overlay_tiles(&tiles, target_w, target_h, target_rect)
            };

            let uv_index = overlay_order_index(overlay_index_map, overlay_id);

            // Attach the new composite.
            // NOTE: intentionally no Detach event emitted here during upgrades.
            // The renderer replaces the existing texture in-place when it
            // receives the new Attached event, avoiding a one-frame blank gap.
            mapped.attached = Some(AttachedTile {
                tile: composite.clone(),
                level: loading.level,
            });

            self.events.push(OverlayEvent::Attached {
                tile_id,
                overlay_id,
                uv_index,
                tile: composite.clone(),
            });
            self.ev_overlay_attached
                .raise((tile_id, overlay_id, uv_index, composite));
        }
    }
}

fn overlay_order_index(index_map: &HashMap<OverlayId, u32>, id: OverlayId) -> u32 {
    index_map.get(&id).copied().unwrap_or(0)
}

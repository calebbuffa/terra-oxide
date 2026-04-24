//! Iterative depth-first tile selection algorithm.
//!
//! This is the core of `kiban`'s selection engine, equivalent to the traversal
//! logic inside cesium-native's `TilesetContentManager`.
//!
//! Operates directly on [`TileStore`] (SoA layout) and [`SelectionState`] -
//! no intermediate [`NodeDescriptor`] marshaling.  The traversal loop
//! touches only the hot `bounds` and `geometric_errors` arrays during culling
//! and SSE checks, maximising cache efficiency.
//!
//! # Two traversal modes
//!
//! When `options.lod.skip_level_of_detail` is **false** (default), the engine
//! runs the base two-phase DFS identical to cesium-native's base traversal.
//! When it is **true**, it runs the skip-LOD two-phase algorithm from
//! `Cesium3DTilesetSkipTraversal`.
//!
//! Load requests carry four raw components:
//! `raw_distance`, `raw_foveated_factor`, `raw_depth`, `raw_reverse_sse`.
//! After the main DFS loop, `normalize_load_scores` normalises all four
//! components across the frame and produces a composite `[0,1]` score,
//! matching CesiumJS `Tile.updatePriority()`.
//!
//! The *priority holder* is the sibling with the lowest foveated factor.
//! All siblings share the holder's distance and foveated factor for their load
//! score computations, matching CesiumJS `tile._priorityHolder`.
//!
//! Incremental frustum culling
//!
//! `parent_plane_masks` is threaded top-down through the work stack.
//! Bit `i` set = plane `i` still needs testing for this subtree.
//! When all bits are 0 (parent was fully inside all planes) the frustum test
//! is skipped entirely - equivalent to the old `skip_frustum_cull` fast path
//! but generalised to handle partially-inside parents too.

use glam::DVec3;
use smallvec::SmallVec;
use std::sync::Arc;
use zukei::{CullingVolume, Plane, SpatialBounds};

use crate::eviction::EvictionPolicy;
use crate::frame_decision::{ExpandResult, LoadPriority, LoadRequest, SelectionOutput};
use crate::occlusion::{TileOcclusionProxy, TileOcclusionState};
use crate::scorer::LoadPriorityScorer;
use crate::selection_state::{TileLoadState, TileRefinementResult, TileStates};
use crate::strategy::TraversalStrategy;
use crate::tile_store::{RefinementMode, TileFlags, TileId, TileStore, bounding_sphere_radius};

use crate::options::SelectionOptions;
use crate::view::{Projection, ViewState};

/// Per-frame immutable inputs for the selection algorithm.
pub struct SelectionContext<'a> {
    pub store: &'a TileStore,
    pub options: &'a SelectionOptions,
    pub views: &'a [ViewState],
    /// Per-view `CullingVolume`s, parallel to `views`.
    ///
    /// These are pure functions of `views` but rebuilding them per tile is
    /// O(views * tiles). Callers compute once per frame and pass a borrowed
    /// slice so visibility queries can index by view directly.
    pub culling_volumes: &'a [CullingVolume],
    pub maximum_screen_space_error: f64,
    pub excluders: &'a [Arc<dyn crate::loader::TileExcluder>],
    /// Pluggable scorer that converts raw per-tile metrics into a composite
    /// `priority.score` value after each traversal pass.
    pub scorer: &'a dyn LoadPriorityScorer,
    /// Pluggable traversal strategy (base DFS vs skip-LOD vs custom).
    pub strategy: &'a dyn TraversalStrategy,
    /// Pluggable eviction policy: determines whether a Ready tile is stale.
    pub eviction_policy: &'a dyn EvictionPolicy,
    /// Current frame timestamp in milliseconds since Unix epoch.
    pub frame_time_ms: u64,
    /// Optional renderer-provided occlusion proxy.
    ///
    /// When set and `SelectionOptions::culling::delay_refinement_for_occlusion`
    /// is `true`, the traversal skips refining any tile reported as
    /// [`Occluded`](crate::occlusion::TileOcclusionState::Occluded).
    pub occlusion_proxy: Option<Arc<dyn TileOcclusionProxy>>,
}

#[derive(Clone, Copy)]
pub struct FrameLocals {
    frame_index: u64,
    now_secs: u64,
    now_ms: u64,
    lod_active: bool,
}

struct NodeSnapshot<'a> {
    tile: TileId,
    bounds: &'a SpatialBounds,
    geometric_error: f64,
    has_content: bool,
    refinement: RefinementMode,
    unconditionally_refined: bool,
    load_state: TileLoadState,
    last_result: TileRefinementResult,
    children_len: usize,
    children: &'a [TileId],
    children_within_parent: bool,
}

struct LoadDecision {
    urgent: Option<LoadRequest>,
    normal: Option<LoadRequest>,
}

/// data from the sibling with the lowest foveated factor, shared by all
/// siblings for their load priority computations.
/// Mirrors CesiumJS `tile._priorityHolder`.
#[derive(Clone, Copy)]
struct PriorityHolderData {
    distance: f64,
    foveated_factor: f64,
}

const MINIMUM_CAMERA_DISTANCE: f64 = 1.0;

/// Compute screen-space error for a tile given its geometric error and bounds.
#[inline]
pub fn compute_sse(geometric_error: f64, view: &ViewState, bounds: &SpatialBounds) -> f64 {
    let distance = bounds.distance_to(view.position);
    compute_sse_at_distance(geometric_error, view, distance, MINIMUM_CAMERA_DISTANCE)
}

/// Variant that accepts a pre-computed camera-to-bounds distance and minimum distance floor.
///
/// Pass `opts.lod.minimum_camera_distance` (default `1.0`) as `min_distance` to
/// prevent divide-by-zero when the camera is inside a tile's bounding volume.
#[inline]
pub fn compute_sse_at_distance(
    geometric_error: f64,
    view: &ViewState,
    distance: f64,
    min_distance: f64,
) -> f64 {
    let multiplier = view.lod_metric_multiplier as f64;
    match &view.projection {
        Projection::Perspective { fov_y, .. } => {
            let sse_denominator = 2.0 * (fov_y * 0.5).tan();
            let distance = distance.max(min_distance);
            let viewport_height = view.viewport_px[1] as f64;
            (geometric_error * viewport_height * multiplier) / (distance * sse_denominator)
        }
        Projection::Orthographic { half_height, .. } => {
            let viewport_height = view.viewport_px[1] as f64;
            let pixel_world_size = 2.0 * half_height / viewport_height;
            if pixel_world_size <= 0.0 {
                return 0.0;
            }
            (geometric_error / pixel_world_size) * multiplier
        }
    }
}

#[inline]
pub(crate) fn build_culling_volume(view: &ViewState) -> CullingVolume {
    match &view.projection {
        Projection::Perspective { fov_x, fov_y } => {
            CullingVolume::from_fov(view.position, view.direction, view.up, *fov_x, *fov_y)
        }
        Projection::Orthographic {
            half_width,
            half_height,
        } => CullingVolume::from_orthographic(
            view.position,
            view.direction,
            view.up,
            -*half_width,
            *half_width,
            -*half_height,
            *half_height,
            0.0,
        ),
    }
}

#[inline]
fn is_visible(bounds: &SpatialBounds, view: &ViewState) -> bool {
    build_culling_volume(view).visibility_bounds(bounds) != zukei::CullingResult::Outside
}

/// Returns `true` if the tile's content is visible in at least one view.
///
/// When `content_bounds` is absent the tile bounds already passed the frustum
/// test, so the content is assumed visible.  When present, tests the (usually
/// tighter) content bounding volume - a tile whose content lies outside all
/// view frustums should not be added to the selection list even though its tile
/// bounds are inside.
///
/// Mirrors CesiumJS `Cesium3DTile.contentVisibility()`.
#[inline]
fn content_visible(tile: TileId, ctx: &SelectionContext<'_>) -> bool {
    match ctx.store.content_bounds(tile) {
        Some(cb) if !matches!(cb, SpatialBounds::Empty) => {
            ctx.views.iter().any(|v| is_visible(cb, v))
        }
        _ => true,
    }
}

/// Test `bounds` against all views with per-view plane masks.
///
/// Returns `(frustum_culled, child_plane_masks)` where:
/// - `frustum_culled` = ALL views say Outside.
/// - `child_plane_masks[i]` = intersecting-plane bitmask for view i.
fn test_frustum_masked(
    bounds: &SpatialBounds,
    views: &[ViewState],
    culling_volumes: &[CullingVolume],
    parent_plane_masks: &[u32],
) -> (bool, Vec<u32>) {
    let mut all_outside = true;
    let mut child_masks = Vec::with_capacity(views.len());

    for (i, _view) in views.iter().enumerate() {
        let parent_mask = parent_plane_masks.get(i).copied().unwrap_or(u32::MAX);
        if parent_mask == 0 {
            // Parent was fully inside this view's frustum.
            all_outside = false;
            child_masks.push(0u32);
        } else {
            let cv = &culling_volumes[i];
            let (result, child_mask) = cv.visibility_bounds_masked(bounds, parent_mask);
            if result != zukei::CullingResult::Outside {
                all_outside = false;
            }
            child_masks.push(child_mask);
        }
    }

    (all_outside, child_masks)
}

/// Build the "all planes need testing" mask for each view (used for the root tile).
fn all_planes_masks(culling_volumes: &[CullingVolume]) -> Vec<u32> {
    culling_volumes
        .iter()
        .map(|cv| cv.all_planes_mask())
        .collect()
}

struct StampSet {
    stamps: Vec<u64>,
    generation: u64,
}

impl StampSet {
    fn new() -> Self {
        Self {
            stamps: Vec::new(),
            generation: 1,
        }
    }

    #[inline]
    fn clear(&mut self) {
        self.generation += 1;
    }

    #[inline]
    fn insert(&mut self, tile: TileId) -> bool {
        let idx = tile.slot();
        if idx >= self.stamps.len() {
            self.stamps.resize(idx + 1, 0);
        }
        if self.stamps[idx] == self.generation {
            false
        } else {
            self.stamps[idx] = self.generation;
            true
        }
    }

    #[inline]
    fn contains(&self, tile: TileId) -> bool {
        self.stamps.get(tile.slot()).copied().unwrap_or(0) == self.generation
    }
}

#[derive(Clone, Copy, Debug)]
struct TraversalDetails {
    all_ready: bool,
    any_selected_last_frame: bool,
    not_yet_ready_count: usize,
}

impl TraversalDetails {
    fn leaf_ready(selected_last_frame: bool) -> Self {
        Self {
            all_ready: true,
            any_selected_last_frame: selected_last_frame,
            not_yet_ready_count: 0,
        }
    }
    fn leaf_not_ready() -> Self {
        Self {
            all_ready: false,
            any_selected_last_frame: false,
            not_yet_ready_count: 1,
        }
    }
    fn empty() -> Self {
        Self {
            all_ready: true,
            any_selected_last_frame: false,
            not_yet_ready_count: 0,
        }
    }
    fn combine(&mut self, other: TraversalDetails) {
        self.all_ready &= other.all_ready;
        self.any_selected_last_frame |= other.any_selected_last_frame;
        self.not_yet_ready_count += other.not_yet_ready_count;
    }
}

enum CullDecision {
    Skip {
        preload_sibling: bool,
    },
    ForceVisit,
    Pass {
        frustum_culled: bool,
        /// per-view masks of still-intersecting planes for children.
        child_plane_masks: Vec<u32>,
    },
}

struct SseDecision {
    max_sse: f64,
    refines: bool,
    child_ancestor_meets_sse: bool,
    self_selected: bool,
    must_continue_refining: bool,
}

struct VisitDecision {
    tile: TileId,
    selected_start: usize,
    load_start: usize,
    detail_start: usize,
    self_selected: bool,
    self_ready: bool,
    has_content: bool,
    unconditionally_refined: bool,
    ancestor_meets_sse: bool,
    load_state: TileLoadState,
    refinement: RefinementMode,
    refines: bool,
    queued_for_load: bool,
    /// True if the tile's content bounding volume is visible in at least one view.
    /// Used to gate selection list additions. Mirrors CesiumJS `contentVisibility()`.
    content_bounds_visible: bool,
}

enum WorkItem {
    Visit {
        tile: TileId,
        ancestor_selected: bool,
        ancestor_meets_sse: bool,
        depth: u32,
        /// per-view masks of still-intersecting frustum planes.
        /// All-zeros = skip frustum entirely (fast path or parent fully inside).
        parent_plane_masks: Arc<[u32]>,
        /// priority holder data from sibling analysis.
        priority_holder: Option<PriorityHolderData>,
        /// parent's geometric error for `meetsScreenSpaceErrorEarly`.
        parent_geometric_error: Option<f64>,
    },
    Finalize(VisitDecision),
}

struct SkipTraversalBuffers {
    traversal_stack: Vec<TileId>,
    descendant_stack: Vec<TileId>,
    selection_stack: Vec<TileId>,
    /// (tile, stack_length_when_pushed).
    ancestor_stack: Vec<(TileId, usize)>,
    should_select: StampSet,
    refines: StampSet,
    /// Maps parent tile -> priority holder for its children group.
    /// Filled by `skip_update_and_push_children`; consumed by `skip_issue_load`.
    /// Mirrors CesiumJS `tile._priorityHolder` propagation in skip traversal.
    priority_holders: std::collections::HashMap<TileId, PriorityHolderData>,
}

impl SkipTraversalBuffers {
    fn new() -> Self {
        Self {
            traversal_stack: Vec::new(),
            descendant_stack: Vec::new(),
            selection_stack: Vec::new(),
            ancestor_stack: Vec::new(),
            should_select: StampSet::new(),
            refines: StampSet::new(),
            priority_holders: std::collections::HashMap::new(),
        }
    }
    fn clear(&mut self) {
        self.traversal_stack.clear();
        self.descendant_stack.clear();
        self.selection_stack.clear();
        self.ancestor_stack.clear();
        self.should_select.clear();
        self.refines.clear();
        self.priority_holders.clear();
    }
}

/// Per-frame working buffers retained across frames for zero-allocation.
pub struct TraversalBuffers {
    work_stack: Vec<WorkItem>,
    detail_stack: Vec<TraversalDetails>,
    selected: Vec<TileId>,
    selected_sse: Vec<f32>,
    /// Parallel to `selected`: stencil selection depth for each tile.
    selected_depths: Vec<u32>,
    /// Parallel to `selected`: final-resolution flag for each tile.
    /// `true` = leaf/foreground; `false` = background ancestor (skip-LOD only).
    selected_final_resolutions: Vec<bool>,
    selected_set: StampSet,
    load: Vec<LoadRequest>,
    /// Membership index over `load`, queried by `contains_load(tile)` and
    /// maintained by `push_load(...)`. Replaces a linear `load.iter().any()`
    /// scan that otherwise degrades to O(N^2) as the load queue grows.
    load_set: StampSet,
    fading_out: Vec<TileId>,
    fading_out_set: StampSet,
    fading_in: Vec<TileId>,
    /// Scratch buffer reused across `(child, distance)` sorts inside
    /// refine expansion. `std::mem::take`-swap pattern keeps the allocation
    /// alive across frames without requiring interior mutability.
    child_sort_buf: Vec<(TileId, f64)>,
    /// Reusable buffer for culling volumes (one per view).
    pub(crate) culling_volumes: Vec<zukei::CullingVolume>,
    /// Reusable buffer for ready tiles (tile_id, geometric_error) pairs.
    pub(crate) ready_tiles: Vec<(TileId, f64)>,
    skip: SkipTraversalBuffers,
}

impl TraversalBuffers {
    pub fn new() -> Self {
        Self {
            work_stack: Vec::new(),
            detail_stack: Vec::new(),
            selected: Vec::new(),
            selected_sse: Vec::new(),
            selected_depths: Vec::new(),
            selected_final_resolutions: Vec::new(),
            selected_set: StampSet::new(),
            load: Vec::new(),
            load_set: StampSet::new(),
            fading_out: Vec::new(),
            fading_out_set: StampSet::new(),
            fading_in: Vec::new(),
            child_sort_buf: Vec::new(),
            culling_volumes: Vec::new(),
            ready_tiles: Vec::new(),
            skip: SkipTraversalBuffers::new(),
        }
    }

    fn clear(&mut self) {
        self.work_stack.clear();
        self.detail_stack.clear();
        self.selected.clear();
        self.selected_sse.clear();
        self.selected_depths.clear();
        self.selected_final_resolutions.clear();
        self.selected_set.clear();
        self.load.clear();
        self.load_set.clear();
        self.fading_out.clear();
        self.fading_out_set.clear();
        self.fading_in.clear();
        self.child_sort_buf.clear();
        self.culling_volumes.clear();
        self.ready_tiles.clear();
        self.skip.clear();
    }

    /// Push `req` onto the load queue and return whether it was newly added.
    ///
    /// Duplicate requests (same `TileId` already in the queue this frame) are
    /// silently ignored and `false` is returned.
    #[inline]
    fn push_load(&mut self, req: LoadRequest) -> bool {
        if self.load_set.insert(req.tile) {
            self.load.push(req);
            true
        } else {
            false
        }
    }

    /// `true` if `tile` has already been queued for load this frame.
    #[inline]
    fn contains_load(&self, tile: TileId) -> bool {
        self.load_set.contains(tile)
    }
}

impl Default for TraversalBuffers {
    fn default() -> Self {
        Self::new()
    }
}

/// Run the selection algorithm for one frame.
pub fn select_tiles(
    ctx: &SelectionContext<'_>,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    expand: &mut dyn FnMut(TileId) -> ExpandResult,
) -> SelectionOutput {
    assert!(!ctx.views.is_empty(), "select called with no views");

    buffers.clear();
    for ex in ctx.excluders {
        ex.start_new_frame();
    }

    let frame = FrameLocals {
        frame_index: state.frame_index,
        now_secs: ctx.frame_time_ms / 1000,
        now_ms: ctx.frame_time_ms,
        lod_active: ctx.options.streaming.enable_lod_transition,
    };

    // Pre-compute fog density once per view per frame, via the strategy hook.
    let fog_densities = ctx.strategy.fog_densities(ctx);

    // Pre-compute dynamic SSE density once per view per frame, via the strategy hook.
    let dynamic_sse_densities = ctx.strategy.dynamic_sse_densities(ctx);

    // Delegate to the pluggable traversal strategy.
    ctx.strategy.execute(
        ctx,
        &frame,
        state,
        buffers,
        expand,
        &fog_densities,
        &dynamic_sse_densities,
    )
}

/// Default fog-density computation used by [`TraversalStrategy::fog_densities`].
///
/// Public so custom strategies can call it and augment the result rather than
/// reimplementing the WGS-84-height correction from scratch.
pub fn compute_default_fog_densities(ctx: &SelectionContext<'_>) -> Vec<f64> {
    if !ctx.options.culling.enable_fog_culling || ctx.options.culling.fog_density_table.is_empty() {
        return Vec::new();
    }
    ctx.views
        .iter()
        .map(|view| {
            let height = view
                .position_cartographic()
                .map(|c| c.height.max(0.0))
                .unwrap_or_else(|| {
                    // Approximate geodetic height when no ellipsoid is configured.
                    // Subtract WGS-84 semi-major axis (~6378 km) from geocentric radius.
                    (view.position.length() - 6.378_136_6e6).max(0.0)
                });
            interpolate_fog_density(&ctx.options.culling.fog_density_table, height)
        })
        .collect()
}

/// Default dynamic-SSE-density computation used by
/// [`TraversalStrategy::dynamic_sse_densities`].
///
/// Public so custom strategies can call it and augment the result rather than
/// reimplementing the horizon-factor formula from scratch.
pub fn compute_default_dynamic_sse_densities(ctx: &SelectionContext<'_>) -> Vec<f64> {
    if !ctx.options.lod.enable_dynamic_detail_reduction {
        return Vec::new();
    }
    ctx.views
        .iter()
        .map(|view| {
            let up = if view.position.length() > 1e-8 {
                view.position.normalize()
            } else {
                DVec3::Z
            };
            let horizon_factor = 1.0 - view.direction.dot(up).abs();
            ctx.options.lod.dynamic_detail_reduction_density * horizon_factor
        })
        .collect()
}

pub(crate) fn select_base_inner(
    ctx: &SelectionContext<'_>,
    frame: &FrameLocals,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    expand: &mut dyn FnMut(TileId) -> ExpandResult,
    fog_densities: &[f64],
    dynamic_sse_densities: &[f64],
) -> SelectionOutput {
    let root_plane_masks: Arc<[u32]> = Arc::from(all_planes_masks(ctx.culling_volumes));

    buffers.work_stack.push(WorkItem::Visit {
        tile: ctx.store.root(),
        ancestor_selected: false,
        ancestor_meets_sse: false,
        depth: 0,
        parent_plane_masks: root_plane_masks,
        priority_holder: None,
        parent_geometric_error: None,
    });

    let mut visited = 0usize;
    let mut culled = 0usize;
    let mut culled_but_visited = 0usize;
    let mut kicked = 0usize;
    let mut refined = 0usize;
    let mut max_depth = 0u32;

    while let Some(item) = buffers.work_stack.pop() {
        match item {
            WorkItem::Visit {
                tile,
                ancestor_selected,
                ancestor_meets_sse,
                depth,
                parent_plane_masks,
                priority_holder,
                parent_geometric_error,
            } => {
                if depth > max_depth {
                    max_depth = depth;
                }
                visit_node(
                    ctx,
                    frame,
                    tile,
                    ancestor_selected,
                    ancestor_meets_sse,
                    depth,
                    &parent_plane_masks,
                    priority_holder,
                    parent_geometric_error,
                    fog_densities,
                    dynamic_sse_densities,
                    state,
                    buffers,
                    &mut visited,
                    &mut culled,
                    &mut culled_but_visited,
                    &mut refined,
                    expand,
                );
            }
            WorkItem::Finalize(decision) => {
                finalize_node(ctx, frame, &decision, state, buffers, &mut kicked);
            }
        }
    }

    if frame.lod_active {
        for &tile in &buffers.fading_in {
            let status = state.get_mut(tile);
            if status.fade_in_ms == 0 {
                status.fade_in_ms = frame.now_ms;
            }
        }
        for &tile in &buffers.fading_out {
            state.get_mut(tile).fade_in_ms = 0;
        }
    }

    // Score all load requests via the pluggable scorer.
    ctx.scorer.score(&mut buffers.load);

    SelectionOutput {
        selected: std::mem::take(&mut buffers.selected),
        selected_sse: std::mem::take(&mut buffers.selected_sse),
        selection_depths: std::mem::take(&mut buffers.selected_depths),
        selection_final_resolutions: std::mem::take(&mut buffers.selected_final_resolutions),
        has_mixed_selection: false,
        load: std::mem::take(&mut buffers.load),
        fading_in: std::mem::take(&mut buffers.fading_in),
        fading_out: std::mem::take(&mut buffers.fading_out),
        nodes_visited: visited,
        nodes_culled: culled,
        nodes_culled_but_visited: culled_but_visited,
        nodes_kicked: kicked,
        nodes_refined: refined,
        max_depth,
    }
}

pub(crate) fn select_skip_inner(
    ctx: &SelectionContext<'_>,
    frame: &FrameLocals,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    expand: &mut dyn FnMut(TileId) -> ExpandResult,
    fog_densities: &[f64],
    dynamic_sse_densities: &[f64],
) -> SelectionOutput {
    let root = ctx.store.root();

    // Early-out: root is not visible or already meets SSE.
    {
        let root_bounds = ctx.store.bounds(root);
        if !ctx.views.iter().any(|v| is_visible(root_bounds, v)) {
            return SelectionOutput::default();
        }
        let root_sse = ctx
            .views
            .iter()
            .map(|v| compute_sse(ctx.store.geometric_error(root), v, root_bounds))
            .fold(0.0_f64, f64::max);
        if root_sse <= ctx.maximum_screen_space_error {
            return SelectionOutput::default();
        }
    }

    let base_screen_space_error = if ctx.options.lod.immediately_load_desired_lod {
        f64::MAX
    } else {
        ctx.options
            .lod
            .base_lod_metric_threshold
            .max(ctx.maximum_screen_space_error)
    };

    let mut visited = 0usize;
    let mut culled = 0usize;
    let mut refined = 0usize;
    let mut max_depth = 0u32;

    // Phase 1.
    execute_traversal_skip(
        ctx,
        frame,
        state,
        buffers,
        expand,
        fog_densities,
        dynamic_sse_densities,
        base_screen_space_error,
        &mut visited,
        &mut culled,
        &mut refined,
        &mut max_depth,
    );

    // Phase 2.
    let has_mixed_content = traverse_and_select_skip(ctx, state, buffers);

    // Score all load requests via the pluggable scorer.
    ctx.scorer.score(&mut buffers.load);

    SelectionOutput {
        selected: std::mem::take(&mut buffers.selected),
        selected_sse: std::mem::take(&mut buffers.selected_sse),
        selection_depths: std::mem::take(&mut buffers.selected_depths),
        selection_final_resolutions: std::mem::take(&mut buffers.selected_final_resolutions),
        has_mixed_selection: has_mixed_content,
        load: std::mem::take(&mut buffers.load),
        fading_in: std::mem::take(&mut buffers.fading_in),
        fading_out: std::mem::take(&mut buffers.fading_out),
        nodes_visited: visited,
        nodes_culled: culled,
        nodes_culled_but_visited: 0,
        nodes_kicked: 0,
        nodes_refined: refined,
        max_depth,
    }
}

/// Update `ancestor_with_content` / `ancestor_with_content_available` for
/// `tile` based on its parent's state.
/// Mirrors CesiumJS `updateTileAncestorContentLinks`.
fn update_ancestor_content_links(tile: TileId, store: &TileStore, state: &mut TileStates) {
    let Some(parent) = store.parent(tile) else {
        let s = state.get_mut(tile);
        s.ancestor_with_content = None;
        s.ancestor_with_content_available = None;
        return;
    };

    let parent_has_content = !store.content_keys(parent).is_empty()
        || state.get(parent).load_state == TileLoadState::Loading;
    let parent_awc = state.get(parent).ancestor_with_content;
    let parent_awca = state.get(parent).ancestor_with_content_available;
    let parent_load = state.get(parent).load_state;

    let s = state.get_mut(tile);
    s.ancestor_with_content = if parent_has_content {
        Some(parent)
    } else {
        parent_awc
    };
    s.ancestor_with_content_available =
        if matches!(parent_load, TileLoadState::Ready | TileLoadState::Expiring) {
            Some(parent)
        } else {
            parent_awca
        };
}

/// Is tile in the base traversal zone (SSE > base threshold)?
/// Mirrors CesiumJS `inBaseTraversal`.
fn in_base_traversal(
    tile: TileId,
    state: &TileStates,
    store: &TileStore,
    base_sse: f64,
    immediately_load_desired: bool,
) -> bool {
    if immediately_load_desired {
        return false;
    }
    if state.get(tile).ancestor_with_content.is_none() {
        return true; // near root - always include
    }
    let sse = state.get(tile).traversal_sse;
    if sse == 0.0 {
        // Leaf: use parent's SSE.
        return store
            .parent(tile)
            .map(|p| state.get(p).traversal_sse > base_sse)
            .unwrap_or(true);
    }
    sse > base_sse
}

/// Has this tile passed the skip threshold and must be loaded?
/// Mirrors CesiumJS `reachedSkippingThreshold`.
fn reached_skipping_threshold(
    tile: TileId,
    state: &TileStates,
    ctx: &SelectionContext<'_>,
) -> bool {
    let lod = &ctx.options.lod;
    if lod.immediately_load_desired_lod {
        return false;
    }
    let ancestor = match state.get(tile).ancestor_with_content {
        Some(a) => a,
        None => return false,
    };
    let tile_sse = state.get(tile).traversal_sse;
    let ancestor_sse = state.get(ancestor).traversal_sse;
    let tile_depth = state.get(tile).traversal_depth;
    let ancestor_depth = state.get(ancestor).traversal_depth;

    tile_sse < ancestor_sse / lod.skip_lod_metric_factor
        && tile_depth > ancestor_depth + lod.skip_levels
}

/// DFS that marks tiles for selection and queues load requests.
/// Mirrors `Cesium3DTilesetSkipTraversal.executeTraversal`.
#[allow(clippy::too_many_arguments)]
fn execute_traversal_skip(
    ctx: &SelectionContext<'_>,
    frame: &FrameLocals,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    expand: &mut dyn FnMut(TileId) -> ExpandResult,
    fog_densities: &[f64],
    dynamic_sse_densities: &[f64],
    base_sse: f64,
    visited: &mut usize,
    culled: &mut usize,
    refined: &mut usize,
    max_depth: &mut u32,
) {
    buffers.skip.traversal_stack.push(ctx.store.root());

    while let Some(tile) = buffers.skip.traversal_stack.pop() {
        *visited += 1;

        // Check content expiry.
        check_and_mark_expiry(tile, ctx.store, state, frame.now_secs, ctx.eviction_policy);

        let load_state = state.get(tile).load_state;
        if load_state == TileLoadState::Failed {
            continue;
        }

        // Update ancestor content links (top-down; parents processed first).
        update_ancestor_content_links(tile, ctx.store, state);

        let snap = build_snapshot(tile, ctx.store, state);

        let distances: SmallVec<[f64; 4]> = ctx
            .views
            .iter()
            .map(|v| snap.bounds.distance_to(v.position))
            .collect();

        // Frustum visibility.
        if !ctx.views.iter().any(|v| is_visible(snap.bounds, v)) {
            *culled += 1;
            continue;
        }

        // Fog culling.
        if !fog_densities.is_empty() && !frame.lod_active {
            let all_fogged = fog_densities
                .iter()
                .zip(distances.iter())
                .all(|(&d, &dist)| {
                    if d <= 0.0 {
                        return false;
                    }
                    1.0 - (-(d * d * dist * dist)).exp()
                        > ctx.options.culling.fog_opacity_cull_threshold
                });
            if all_fogged {
                *culled += 1;
                continue;
            }
        }

        // SSE.
        let sse = compute_max_sse_with_dynamic(&snap, ctx, &distances, dynamic_sse_densities);

        // Store for ancestor lookup by descendants (DFS ensures parent first).
        {
            let parent_depth = ctx
                .store
                .parent(tile)
                .map(|p| state.get(p).traversal_depth)
                .unwrap_or(0);
            let s = state.get_mut(tile);
            s.traversal_sse = sse;
            s.traversal_depth = if ctx.store.parent(tile).is_some() {
                parent_depth + 1
            } else {
                0
            };
        }

        // GAP-9: track max depth visited in skip traversal.
        let tile_depth = state.get(tile).traversal_depth;
        if tile_depth > *max_depth {
            *max_depth = tile_depth;
        }

        let can_traverse_flag = skip_can_traverse(tile, ctx, state, sse);

        // Expand latent children.
        if ctx.store.might_have_latent_children(tile) && ctx.store.children(tile).is_empty() {
            if !snap.has_content
                || matches!(load_state, TileLoadState::Ready | TileLoadState::Expiring)
            {
                let _ = expand(tile);
            }
        }

        let parent_refines = ctx
            .store
            .parent(tile)
            .map(|p| buffers.skip.refines.contains(p))
            .unwrap_or(true);

        // Occlusion gate: check before pushing children so the traversal stack
        // is never populated for occluded tiles.
        let skip_for_occlusion = can_traverse_flag
            && ctx.options.culling.delay_refinement_for_occlusion
            && ctx.occlusion_proxy.as_ref().map_or(false, |proxy| {
                proxy.tile_occlusion(tile.0.get() as u64) == TileOcclusionState::Occluded
            });

        let tile_refines = if can_traverse_flag && !skip_for_occlusion {
            let any_visible = skip_update_and_push_children(tile, ctx, state, buffers, &distances);
            any_visible && parent_refines
        } else {
            false
        };

        if tile_refines {
            *refined += 1;
            buffers.skip.refines.insert(tile);
        }

        let stopped_refining = !tile_refines && parent_refines;
        let depth = state.get(tile).traversal_depth;

        if !snap.has_content {
            if stopped_refining {
                skip_select_desired_tile(tile, ctx, state, buffers, &distances, sse, depth);
            }
            skip_load_tile(tile, ctx, state, buffers, &distances, sse, depth);
        } else if snap.refinement == RefinementMode::Add {
            skip_select_desired_tile(tile, ctx, state, buffers, &distances, sse, depth);
            skip_load_tile(tile, ctx, state, buffers, &distances, sse, depth);
        } else {
            if in_base_traversal(
                tile,
                state,
                ctx.store,
                base_sse,
                ctx.options.lod.immediately_load_desired_lod,
            ) {
                skip_load_tile(tile, ctx, state, buffers, &distances, sse, depth);
                if stopped_refining {
                    skip_select_desired_tile(tile, ctx, state, buffers, &distances, sse, depth);
                }
            } else if stopped_refining {
                skip_select_desired_tile(tile, ctx, state, buffers, &distances, sse, depth);
                skip_load_tile(tile, ctx, state, buffers, &distances, sse, depth);
            } else if reached_skipping_threshold(tile, state, ctx) {
                skip_load_tile(tile, ctx, state, buffers, &distances, sse, depth);
            }
        }
    }
}

/// Can this tile be traversed into its children?
/// Mirrors CesiumJS `canTraverse`.
fn skip_can_traverse(
    tile: TileId,
    ctx: &SelectionContext<'_>,
    state: &TileStates,
    sse: f64,
) -> bool {
    if ctx.store.children(tile).is_empty() {
        return false;
    }
    if ctx
        .store
        .flags(tile)
        .contains(TileFlags::UNCONDITIONALLY_REFINED)
    {
        return state.get(tile).load_state != TileLoadState::Expiring;
    }
    sse > ctx.maximum_screen_space_error
}

/// Sort children farthest-first, push visible ones onto the traversal
/// stack.  Load invisible siblings when `load_siblings_on_skip` is set.
/// Returns true if any child is visible.
fn skip_update_and_push_children(
    tile: TileId,
    ctx: &SelectionContext<'_>,
    state: &TileStates,
    buffers: &mut TraversalBuffers,
    _parent_distances: &[f64],
) -> bool {
    let children: Vec<TileId> = ctx.store.children(tile).to_vec();
    if children.is_empty() {
        return false;
    }

    // GAP-5: Compute the priority holder for this sibling group (the child with
    // the lowest foveated factor) and store it keyed by the parent tile.
    // Mirrors CesiumJS `tile._priorityHolder` propagation in skip traversal.
    if let Some(ph) = compute_priority_holder(&children, ctx) {
        buffers.skip.priority_holders.insert(tile, ph);
    }

    // Sort farthest-first so LIFO stack visits closest child first.
    let mut sorted: Vec<(TileId, f64)> = children
        .iter()
        .map(|&c| {
            let dist = ctx
                .views
                .iter()
                .map(|v| ctx.store.bounds(c).distance_to(v.position))
                .fold(f64::MAX, f64::min);
            (c, dist)
        })
        .collect();
    sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut any_visible = false;
    for (child, _dist) in &sorted {
        let child = *child;
        let child_bounds = ctx.store.bounds(child);
        let visible = ctx.views.iter().any(|v| is_visible(child_bounds, v));
        if visible {
            buffers.skip.traversal_stack.push(child);
            any_visible = true;
        } else if ctx.options.streaming.load_siblings_on_skip {
            let child_distances: SmallVec<[f64; 4]> = ctx
                .views
                .iter()
                .map(|v| child_bounds.distance_to(v.position))
                .collect();
            let child_sse = if child_distances.len() == ctx.views.len() {
                ctx.views
                    .iter()
                    .zip(child_distances.iter())
                    .map(|(v, &d)| {
                        compute_sse_at_distance(
                            ctx.store.geometric_error(child),
                            v,
                            d,
                            ctx.options.lod.minimum_camera_distance,
                        )
                    })
                    .fold(0.0_f64, f64::max)
            } else {
                0.0
            };
            skip_issue_load(
                child,
                state.get(child).load_state,
                ctx,
                buffers,
                &child_distances,
                child_sse,
                0,
                true,
            );
        }
    }
    any_visible
}

/// Queue a load for this tile if eligible.
fn skip_load_tile(
    tile: TileId,
    ctx: &SelectionContext<'_>,
    state: &TileStates,
    buffers: &mut TraversalBuffers,
    distances: &[f64],
    sse: f64,
    depth: u32,
) {
    let load_state = state.get(tile).load_state;
    if matches!(load_state, TileLoadState::Ready | TileLoadState::Failed) {
        return;
    }
    skip_issue_load(tile, load_state, ctx, buffers, distances, sse, depth, false);
}

fn skip_issue_load(
    tile: TileId,
    load_state: TileLoadState,
    ctx: &SelectionContext<'_>,
    buffers: &mut TraversalBuffers,
    distances: &[f64],
    sse: f64,
    depth: u32,
    is_sibling_preload: bool,
) {
    if matches!(
        load_state,
        TileLoadState::Ready | TileLoadState::Failed | TileLoadState::Loading
    ) {
        return;
    }
    if buffers.contains_load(tile) {
        return;
    }

    // Use the priority holder's distance and foveated factor if available.
    // The priority holder is the sibling with the lowest foveated factor; all
    // siblings share its metrics so their load priority is differentiated only
    // by depth/SSE, not by individual angle and distance.
    // Mirrors CesiumJS `tile._priorityHolder` in skip traversal.
    let (distance, fov_factor) = if let Some(parent) = ctx.store.parent(tile) {
        if let Some(ph) = buffers.skip.priority_holders.get(&parent) {
            (ph.distance, ph.foveated_factor)
        } else {
            let d = distances.iter().cloned().fold(f64::MAX, f64::min);
            let ff = foveated_factor_from_bounds_and_views(ctx.store.bounds(tile), ctx.views);
            (d, ff)
        }
    } else {
        let d = distances.iter().cloned().fold(f64::MAX, f64::min);
        let ff = foveated_factor_from_bounds_and_views(ctx.store.bounds(tile), ctx.views);
        (d, ff)
    };
    let reverse_sse = if sse > 0.0 { 1.0 / sse } else { 0.0 };

    let group = if is_sibling_preload {
        crate::frame_decision::PriorityGroup::PRELOAD
    } else if ctx.options.lod.enable_foveated_rendering
        && is_foveated_deferred_for_bounds(ctx.store.bounds(tile), ctx)
    {
        crate::frame_decision::PriorityGroup::DEFERRED
    } else {
        crate::frame_decision::PriorityGroup::NORMAL
    };

    let score = compute_load_score(ctx.views, ctx.store.bounds(tile));
    buffers.push_load(LoadRequest::new(
        tile,
        LoadPriority { group, score },
        distance,
        fov_factor,
        depth,
        reverse_sse,
        bounding_sphere_radius(ctx.store.bounds(tile)),
    ));
}

/// Mark a tile or its best available ancestor/descendant as `should_select`.
/// Mirrors CesiumJS `selectDesiredTile`.
fn skip_select_desired_tile(
    tile: TileId,
    ctx: &SelectionContext<'_>,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    _distances: &[f64],
    _sse: f64,
    _depth: u32,
) {
    let content_available = matches!(
        state.get(tile).load_state,
        TileLoadState::Ready | TileLoadState::Expiring
    ) && !ctx.store.content_keys(tile).is_empty();

    if content_available {
        buffers.skip.should_select.insert(tile);
    } else if let Some(ancestor) = state.get(tile).ancestor_with_content_available {
        buffers.skip.should_select.insert(ancestor);
    } else {
        skip_select_descendants(tile, ctx, state, buffers);
    }
}

/// Walk up to 2 levels below `root` to find and mark available content.
/// Mirrors CesiumJS `selectDescendants`.
fn skip_select_descendants(
    root: TileId,
    ctx: &SelectionContext<'_>,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
) {
    let skip_lod_fallback_depth = ctx.options.lod.skip_lod_fallback_depth;
    let root_depth = state.get(root).traversal_depth;

    buffers.skip.descendant_stack.push(root);
    while let Some(tile) = buffers.skip.descendant_stack.pop() {
        let node_depth = state.get(tile).traversal_depth;
        for &child in ctx.store.children(tile) {
            if !ctx
                .views
                .iter()
                .any(|v| is_visible(ctx.store.bounds(child), v))
            {
                continue;
            }
            if matches!(
                state.get(child).load_state,
                TileLoadState::Ready | TileLoadState::Expiring
            ) && !ctx.store.content_keys(child).is_empty()
            {
                buffers.skip.should_select.insert(child);
            } else if node_depth - root_depth < skip_lod_fallback_depth {
                buffers.skip.descendant_stack.push(child);
            }
        }
    }
}

/// Preorder traversal that assigns selection_depth (stencil) and
/// emits the selection list.
/// Mirrors CesiumJS `traverseAndSelect`. Returns `has_mixed_content`.
fn traverse_and_select_skip(
    ctx: &SelectionContext<'_>,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
) -> bool {
    let root = ctx.store.root();
    buffers.skip.selection_stack.push(root);

    let mut last_ancestor: Option<TileId> = None;
    let mut has_mixed_content = false;

    loop {
        // Drain ancestor stack entries whose stack position has been reached.
        loop {
            let Some(&(waiting, stack_len)) = buffers.skip.ancestor_stack.last() else {
                break;
            };
            if stack_len != buffers.skip.selection_stack.len() {
                break;
            }
            buffers.skip.ancestor_stack.pop();
            let is_last = last_ancestor == Some(waiting);
            // `is_last = true`  -> this tile is the deepest pushed ancestor = no selected
            // descendants were pushed after it = it IS the final resolution tile.
            // `is_last = false` -> it is a background ancestor with selected descendants
            //                   below it = `final_resolution = false`.
            // Mirrors CesiumJS: `_finalResolution = !(tile !== lastAncestor) = is_last`.
            let depth = buffers.skip.ancestor_stack.len() as u32;
            emit_skip_selection(waiting, depth, is_last, ctx, state, buffers);
            if depth > 0 {
                has_mixed_content = true;
            }
        }

        let Some(tile) = buffers.skip.selection_stack.pop() else {
            if buffers.skip.ancestor_stack.is_empty() {
                break;
            }
            continue;
        };

        let can_traverse = skip_can_traverse(tile, ctx, state, state.get(tile).traversal_sse);

        if buffers.skip.should_select.contains(tile) {
            if ctx.store.refinement(tile) == RefinementMode::Add {
                // ADD tiles are always final resolution: they render alongside
                // their children but are not background ancestors.
                // Mirrors CesiumJS: ADD tiles keep `_finalResolution = true` (default).
                let depth = buffers.skip.ancestor_stack.len() as u32;
                emit_skip_selection(tile, depth, true, ctx, state, buffers);
            } else {
                // REPLACE: defer emission until all descendants are processed.
                let depth = buffers.skip.ancestor_stack.len() as u32;
                if depth > 0 {
                    has_mixed_content = true;
                }
                last_ancestor = Some(tile);
                buffers
                    .skip
                    .ancestor_stack
                    .push((tile, buffers.skip.selection_stack.len()));
            }
        }

        if can_traverse {
            for &child in ctx.store.children(tile) {
                if ctx
                    .views
                    .iter()
                    .any(|v| is_visible(ctx.store.bounds(child), v))
                {
                    buffers.skip.selection_stack.push(child);
                }
            }
        }
    }

    // Drain remaining ancestor stack (should be empty in well-formed tilesets).
    while let Some((waiting, _)) = buffers.skip.ancestor_stack.pop() {
        let depth = buffers.skip.ancestor_stack.len() as u32;
        emit_skip_selection(waiting, depth, true, ctx, state, buffers);
    }

    has_mixed_content
}

fn emit_skip_selection(
    tile: TileId,
    selection_depth: u32,
    final_resolution: bool,
    ctx: &SelectionContext<'_>,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
) {
    if !matches!(
        state.get(tile).load_state,
        TileLoadState::Ready | TileLoadState::Expiring
    ) {
        return;
    }
    // Content bounds check mirrors CesiumJS `selectTile` -> `contentVisibility()`.
    if content_visible(tile, ctx) && buffers.selected_set.insert(tile) {
        let sse = state.get(tile).traversal_sse as f32;
        buffers.selected.push(tile);
        buffers.selected_sse.push(sse);
        buffers.selected_depths.push(selection_depth);
        buffers.selected_final_resolutions.push(final_resolution);
        state.get_mut(tile).last_touched_secs = ctx.frame_time_ms / 1000;
    }
}

fn evaluate_culling(
    snap: &NodeSnapshot<'_>,
    ctx: &SelectionContext<'_>,
    lod_active: bool,
    parent_plane_masks: &[u32],
    fog_densities: &[f64],
    distances: &[f64],
) -> CullDecision {
    // Excluders.
    if !ctx.excluders.is_empty()
        && ctx
            .excluders
            .iter()
            .any(|ex| ex.should_exclude(snap.tile, ctx.store))
    {
        return CullDecision::Skip {
            preload_sibling: false,
        };
    }

    // Viewer-request-volume.
    if let Some(vrv) = ctx.store.viewer_request_volume(snap.tile) {
        if !ctx.views.iter().any(|v| views_inside_volume(v, vrv)) {
            return CullDecision::Skip {
                preload_sibling: false,
            };
        }
    }

    let under_camera =
        ctx.options.culling.render_nodes_under_camera && distances.iter().any(|&d| d == 0.0);

    // Frustum culling with plane masks (sets masks to 0 -> instant pass).
    let all_masks_zero = parent_plane_masks.iter().all(|&m| m == 0);

    let (frustum_culled, child_plane_masks) = if !ctx.options.culling.enable_frustum_culling
        || lod_active
        || matches!(snap.bounds, SpatialBounds::Empty)
        || under_camera
        || all_masks_zero
    {
        (false, vec![0u32; ctx.views.len()])
    } else {
        // C3: cullWithChildrenBounds - for Replace tiles test children bounds.
        let use_children = snap.refinement == RefinementMode::Replace
            && !snap.children.is_empty()
            && snap.children.iter().all(|&c| {
                !ctx.store
                    .flags(c)
                    .contains(TileFlags::UNCONDITIONALLY_REFINED)
            });

        if use_children {
            // Cull only if NO child is visible in ANY view.
            let any_child_visible = ctx.views.iter().enumerate().any(|(vi, _v)| {
                let pm = parent_plane_masks.get(vi).copied().unwrap_or(u32::MAX);
                let cv = &ctx.culling_volumes[vi];
                snap.children.iter().any(|&c| {
                    cv.visibility_bounds_masked(ctx.store.bounds(c), pm).0
                        != zukei::CullingResult::Outside
                })
            });
            if any_child_visible {
                (false, parent_plane_masks.to_vec())
            } else {
                (true, vec![0u32; ctx.views.len()])
            }
        } else {
            test_frustum_masked(
                snap.bounds,
                ctx.views,
                ctx.culling_volumes,
                parent_plane_masks,
            )
        }
    };

    if frustum_culled {
        if ctx.store.parent(snap.tile).is_none() {
            return CullDecision::Pass {
                frustum_culled: true,
                child_plane_masks,
            };
        }
        if snap.unconditionally_refined
            && ctx.options.loading.prevent_holes
            && snap.refinement == RefinementMode::Replace
        {
            return CullDecision::ForceVisit;
        }
        let preload_sibling = ctx.options.loading.preload_siblings
            && snap.has_content
            && !matches!(
                snap.load_state,
                TileLoadState::Ready | TileLoadState::Failed | TileLoadState::Expiring
            );
        return CullDecision::Skip { preload_sibling };
    }

    // Clipping planes.
    for plane in &ctx.options.culling.clipping_planes {
        if bounds_entirely_clipped(snap.bounds, plane) {
            return CullDecision::Skip {
                preload_sibling: false,
            };
        }
    }

    // Fog culling.
    if !fog_densities.is_empty() && !lod_active {
        let all_fogged = fog_densities
            .iter()
            .zip(distances.iter())
            .all(|(&density, &dist)| {
                if density <= 0.0 {
                    return false;
                }
                1.0 - (-density * density * dist * dist).exp()
                    > ctx.options.culling.fog_opacity_cull_threshold
            });
        if all_fogged {
            return CullDecision::Skip {
                preload_sibling: false,
            };
        }
    }

    CullDecision::Pass {
        frustum_culled: false,
        child_plane_masks,
    }
}

fn evaluate_sse(
    snap: &NodeSnapshot<'_>,
    ctx: &SelectionContext<'_>,
    frustum_culled: bool,
    ancestor_meets_sse: bool,
    distances: &[f64],
    dynamic_sse_densities: &[f64],
) -> SseDecision {
    let max_sse = compute_max_sse_with_dynamic(snap, ctx, distances, dynamic_sse_densities);

    let meets_sse = if frustum_culled {
        !ctx.options.culling.enforce_culled_screen_space_error
            || max_sse <= ctx.options.culling.culled_screen_space_error
    } else {
        max_sse <= ctx.maximum_screen_space_error
    };

    let refine_for_sse = !meets_sse && !ancestor_meets_sse;
    let is_ready = matches!(
        snap.load_state,
        TileLoadState::Ready | TileLoadState::Expiring
    );

    let mut refines = snap.children_len > 0 && (snap.unconditionally_refined || refine_for_sse);
    let mut child_ancestor_meets_sse = ancestor_meets_sse;
    let mut must_continue_refining = false;

    if !refines && snap.children_len > 0 {
        let must_continue =
            snap.last_result.original() == TileRefinementResult::Refined && !is_ready;
        if must_continue {
            refines = true;
            must_continue_refining = true;
            child_ancestor_meets_sse = true;
        }
    }

    let self_selected = is_ready && (!refines || snap.refinement == RefinementMode::Add);

    SseDecision {
        max_sse,
        refines,
        child_ancestor_meets_sse,
        self_selected,
        must_continue_refining,
    }
}

/// Returns true if this tile (child of an ADD parent) already meets SSE
/// using its parent's geometric error - skip visiting it.
/// Mirrors CesiumJS `meetsScreenSpaceErrorEarly`.
fn meets_screen_space_error_early(
    tile: TileId,
    parent_geometric_error: Option<f64>,
    ctx: &SelectionContext<'_>,
    distances: &[f64],
) -> bool {
    let parent_ge = match parent_geometric_error {
        Some(ge) => ge,
        None => return false,
    };
    let parent = match ctx.store.parent(tile) {
        Some(p) => p,
        None => return false,
    };
    if ctx.store.refinement(parent) != RefinementMode::Add {
        return false;
    }
    if ctx
        .store
        .flags(parent)
        .contains(TileFlags::UNCONDITIONALLY_REFINED)
    {
        return false;
    }

    let max_sse = if distances.len() == ctx.views.len() {
        ctx.views
            .iter()
            .zip(distances.iter())
            .map(|(v, &d)| {
                compute_sse_at_distance(parent_ge, v, d, ctx.options.lod.minimum_camera_distance)
            })
            .fold(0.0_f64, f64::max)
    } else {
        ctx.views
            .iter()
            .map(|v| compute_sse(parent_ge, v, ctx.store.bounds(tile)))
            .fold(0.0_f64, f64::max)
    };

    max_sse <= ctx.maximum_screen_space_error
}

fn evaluate_load(
    snap: &NodeSnapshot<'_>,
    sse: &SseDecision,
    ancestor_meets_sse: bool,
    depth: u32,
    ctx: &SelectionContext<'_>,
    distances: &[f64],
    priority_holder: Option<PriorityHolderData>,
) -> LoadDecision {
    let can_load = snap.has_content
        && !ancestor_meets_sse
        && matches!(
            snap.load_state,
            TileLoadState::Unloaded
                | TileLoadState::RetryScheduled
                | TileLoadState::Evicted
                | TileLoadState::Expiring
        );

    if !can_load {
        return LoadDecision {
            urgent: None,
            normal: None,
        };
    }

    // Use priority holder's data if provided.
    let (distance, fov_factor) = if let Some(ph) = priority_holder {
        (ph.distance, ph.foveated_factor)
    } else {
        let dist = distances.iter().cloned().fold(f64::MAX, f64::min);
        let ff = foveated_factor(snap, ctx.views);
        (dist, ff)
    };

    let reverse_sse = if sse.max_sse > 0.0 {
        1.0 / sse.max_sse
    } else {
        0.0
    };
    let score = compute_load_score(ctx.views, snap.bounds);

    if sse.must_continue_refining {
        LoadDecision {
            urgent: Some(LoadRequest::new(
                snap.tile,
                LoadPriority::urgent(score),
                distance,
                fov_factor,
                depth,
                reverse_sse,
                bounding_sphere_radius(snap.bounds),
            )),
            normal: None,
        }
    } else {
        let is_foveated_deferred =
            ctx.options.lod.enable_foveated_rendering && is_foveated_deferred(snap, ctx);
        // Progressive resolution: tiles that are only needed at full quality
        // (their reduced-height SSE already meets the threshold) are deprioritised
        // so the tileset settles at coarser detail first.
        // Mirrors CesiumJS `_priorityProgressiveResolution` in `updatePriority()`.
        let is_progressive_resolution_deferred = ctx.options.lod.enable_progressive_resolution
            && sse.max_sse * ctx.options.lod.progressive_resolution_height_fraction
                <= ctx.maximum_screen_space_error;
        let priority = if is_foveated_deferred || is_progressive_resolution_deferred {
            LoadPriority::deferred(score)
        } else {
            LoadPriority::normal(score)
        };
        LoadDecision {
            urgent: None,
            normal: Some(LoadRequest::new(
                snap.tile,
                priority,
                distance,
                fov_factor,
                depth,
                reverse_sse,
                bounding_sphere_radius(snap.bounds),
            )),
        }
    }
}

/// Find the sibling with the minimum foveated factor (= closest to look dir).
/// All siblings share its distance and foveated factor for load scoring.
fn compute_priority_holder(
    children: &[TileId],
    ctx: &SelectionContext<'_>,
) -> Option<PriorityHolderData> {
    if children.is_empty() {
        return None;
    }
    let mut best_ff = f64::MAX;
    let mut best_dist = f64::MAX;
    for &child in children {
        let bounds = ctx.store.bounds(child);
        let ff = foveated_factor_from_bounds_and_views(bounds, ctx.views);
        if ff < best_ff {
            best_ff = ff;
            best_dist = ctx
                .views
                .iter()
                .map(|v| bounds.distance_to(v.position))
                .fold(f64::MAX, f64::min);
        }
    }
    if best_ff == f64::MAX {
        None
    } else {
        Some(PriorityHolderData {
            distance: best_dist,
            foveated_factor: best_ff,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn visit_node(
    ctx: &SelectionContext<'_>,
    frame: &FrameLocals,
    tile: TileId,
    ancestor_selected: bool,
    ancestor_meets_sse: bool,
    depth: u32,
    parent_plane_masks: &[u32],
    priority_holder: Option<PriorityHolderData>,
    parent_geometric_error: Option<f64>,
    fog_densities: &[f64],
    dynamic_sse_densities: &[f64],
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    visited: &mut usize,
    culled: &mut usize,
    culled_but_visited: &mut usize,
    refined: &mut usize,
    expand: &mut dyn FnMut(TileId) -> ExpandResult,
) {
    *visited += 1;

    // Check content expiry before reading load_state.
    check_and_mark_expiry(tile, ctx.store, state, frame.now_secs, ctx.eviction_policy);

    let load_state = state.get(tile).load_state;

    if load_state == TileLoadState::Failed {
        buffers.detail_stack.push(TraversalDetails::empty());
        return;
    }
    if load_state == TileLoadState::RetryScheduled
        && frame.frame_index < state.get(tile).next_retry_frame
    {
        buffers.detail_stack.push(TraversalDetails::empty());
        return;
    }

    let snap = build_snapshot(tile, ctx.store, state);

    let distances_per_view: SmallVec<[f64; 4]> = ctx
        .views
        .iter()
        .map(|v| snap.bounds.distance_to(v.position))
        .collect();

    // meetsScreenSpaceErrorEarly for children of ADD parents.
    if meets_screen_space_error_early(tile, parent_geometric_error, ctx, &distances_per_view) {
        *culled += 1;
        buffers.detail_stack.push(TraversalDetails::empty());
        return;
    }

    let (frustum_culled, child_plane_masks) = match evaluate_culling(
        &snap,
        ctx,
        frame.lod_active,
        parent_plane_masks,
        fog_densities,
        &distances_per_view,
    ) {
        CullDecision::Skip { preload_sibling } => {
            *culled += 1;
            state.get_mut(tile).last_result = TileRefinementResult::Culled;
            if ctx.options.loading.prevent_holes
                && snap.refinement == RefinementMode::Replace
                && !snap.unconditionally_refined
            {
                if snap.has_content && !load_state_is_terminal(snap.load_state) {
                    let score = compute_load_score(ctx.views, snap.bounds);
                    let dist = distances_per_view.iter().cloned().fold(f64::MAX, f64::min);
                    let ff = foveated_factor(&snap, ctx.views);
                    buffers.push_load(LoadRequest::new(
                        tile,
                        LoadPriority::normal(score),
                        dist,
                        ff,
                        depth,
                        0.0,
                        bounding_sphere_radius(snap.bounds),
                    ));
                }
                let is_ready = matches!(
                    snap.load_state,
                    TileLoadState::Ready | TileLoadState::Expiring
                );
                let was_selected =
                    matches!(snap.last_result.original(), TileRefinementResult::Selected);
                let details = if is_ready {
                    TraversalDetails::leaf_ready(was_selected)
                } else {
                    TraversalDetails::leaf_not_ready()
                };
                buffers.detail_stack.push(details);
            } else {
                if preload_sibling {
                    let score = compute_load_score(ctx.views, snap.bounds);
                    let dist = distances_per_view.iter().cloned().fold(f64::MAX, f64::min);
                    let ff = foveated_factor(&snap, ctx.views);
                    buffers.push_load(LoadRequest::new(
                        tile,
                        LoadPriority::preload_with_score(score),
                        dist,
                        ff,
                        depth,
                        0.0,
                        bounding_sphere_radius(snap.bounds),
                    ));
                }
                buffers.detail_stack.push(TraversalDetails::empty());
            }
            return;
        }
        CullDecision::ForceVisit => {
            *culled += 1;
            *culled_but_visited += 1;
            if ctx.options.loading.preload_siblings
                && snap.has_content
                && !matches!(
                    snap.load_state,
                    TileLoadState::Ready | TileLoadState::Failed | TileLoadState::Expiring
                )
            {
                let score = compute_load_score(ctx.views, snap.bounds);
                let dist = distances_per_view.iter().cloned().fold(f64::MAX, f64::min);
                let ff = foveated_factor(&snap, ctx.views);
                buffers.push_load(LoadRequest::new(
                    tile,
                    LoadPriority::preload_with_score(score),
                    dist,
                    ff,
                    depth,
                    0.0,
                    bounding_sphere_radius(snap.bounds),
                ));
            }
            (true, vec![0u32; ctx.views.len()])
        }
        CullDecision::Pass {
            frustum_culled,
            child_plane_masks,
        } => (frustum_culled, child_plane_masks),
    };

    // Expand latent children.
    if ctx.store.might_have_latent_children(tile) && ctx.store.children(tile).is_empty() {
        if !snap.has_content
            || matches!(
                snap.load_state,
                TileLoadState::Ready | TileLoadState::Expiring
            )
        {
            let _ = expand(tile);
        }
    }

    let sse = evaluate_sse(
        &snap,
        ctx,
        frustum_culled,
        ancestor_meets_sse,
        &distances_per_view,
        dynamic_sse_densities,
    );
    let loads = evaluate_load(
        &snap,
        &sse,
        ancestor_meets_sse,
        depth,
        ctx,
        &distances_per_view,
        priority_holder,
    );

    let queued_for_load = if let Some(req) = loads.urgent {
        buffers.push_load(req);
        true
    } else {
        false
    };

    // Mirrors CesiumJS `selectTile` -> `contentVisibility()` check: only add
    // to the selection list when the content bounding volume (which may be tighter
    // than the tile bounds) is actually visible.  Traversal decisions (refine,
    // load) are unaffected.
    let content_bounds_visible = !sse.self_selected || content_visible(tile, ctx);

    let selected_start = buffers.selected.len();
    let load_start = buffers.load.len();
    let detail_start = buffers.detail_stack.len();

    buffers.work_stack.push(WorkItem::Finalize(VisitDecision {
        tile,
        selected_start,
        load_start,
        detail_start,
        self_selected: sse.self_selected,
        self_ready: matches!(
            snap.load_state,
            TileLoadState::Ready | TileLoadState::Expiring
        ),
        has_content: snap.has_content,
        unconditionally_refined: snap.unconditionally_refined,
        ancestor_meets_sse,
        load_state: snap.load_state,
        refinement: snap.refinement,
        refines: sse.refines,
        queued_for_load,
        content_bounds_visible,
    }));

    if sse.self_selected && content_bounds_visible {
        buffers.selected.push(tile);
        buffers.selected_set.insert(tile);
        buffers.selected_sse.push(sse.max_sse as f32);
        buffers.selected_depths.push(0);
        buffers.selected_final_resolutions.push(true);
    }

    if let Some(req) = loads.normal {
        buffers.push_load(req);
    }

    if is_zoom_out_transition(&snap, &sse, frame.lod_active) {
        add_rendered_descendants_to_fading_out(ctx.store, state, tile, buffers);
    }

    if sse.refines {
        // Occlusion gate: skip pushing children if the renderer reports this
        // tile as occluded, deferring refinement for another frame.
        let skip_for_occlusion = ctx.options.culling.delay_refinement_for_occlusion
            && ctx.occlusion_proxy.as_ref().map_or(false, |proxy| {
                proxy.tile_occlusion(tile.0.get() as u64) == TileOcclusionState::Occluded
            });

        if !skip_for_occlusion {
            *refined += 1;

            let children: &[TileId] = snap.children;

            // When CHILDREN_WITHIN_PARENT and not frustum_culled, pass mask=0
            // -> children skip frustum test entirely.
            //
            // `child_masks` is shared by every child of this tile, so wrap it
            // in `Arc<[u32]>` and hand out cheap `Arc::clone` references instead
            // of deep-copying the Vec once per child.
            let child_masks: Arc<[u32]> = if !frustum_culled && snap.children_within_parent {
                Arc::from(vec![0u32; ctx.views.len()])
            } else {
                Arc::from(child_plane_masks)
            };

            // Compute priority holder for this group of children.
            let ph = compute_priority_holder(children, ctx);

            // Sort farthest-first so stack LIFO visits closest child first.
            // Reuse the per-frame scratch buffer to avoid a Vec allocation per
            // refining tile.
            let mut sorted = std::mem::take(&mut buffers.child_sort_buf);
            sorted.clear();
            sorted.reserve(children.len());
            for &c in children {
                let dist = ctx
                    .views
                    .iter()
                    .map(|v| ctx.store.bounds(c).distance_to(v.position))
                    .fold(f64::MAX, f64::min);
                sorted.push((c, dist));
            }
            sorted.sort_unstable_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            for &(child, _) in &sorted {
                buffers.work_stack.push(WorkItem::Visit {
                    tile: child,
                    ancestor_selected: ancestor_selected || sse.self_selected,
                    ancestor_meets_sse: sse.child_ancestor_meets_sse,
                    depth: depth + 1,
                    parent_plane_masks: Arc::clone(&child_masks),
                    priority_holder: ph,
                    parent_geometric_error: Some(snap.geometric_error),
                });
            }

            buffers.child_sort_buf = sorted;
        } // end if !skip_for_occlusion
    }
}

fn add_rendered_descendants_to_fading_out(
    store: &TileStore,
    state: &TileStates,
    root: TileId,
    buffers: &mut TraversalBuffers,
) {
    let mut stack: Vec<TileId> = store.children(root).to_vec();
    while let Some(tile) = stack.pop() {
        let last = state.get(tile).last_result;
        let was_selected = last == TileRefinementResult::Selected
            || (last == TileRefinementResult::Refined
                && store.refinement(tile) == RefinementMode::Add);
        if was_selected {
            if buffers.fading_out_set.insert(tile) {
                buffers.fading_out.push(tile);
            }
        }
        stack.extend_from_slice(store.children(tile));
    }
}

fn combine_child_details(
    decision: &VisitDecision,
    state: &TileStates,
    buffers: &mut TraversalBuffers,
) -> TraversalDetails {
    let child_details: Vec<TraversalDetails> = buffers
        .detail_stack
        .drain(decision.detail_start..)
        .collect();

    let mut combined = if child_details.is_empty() {
        if decision.self_ready {
            let last = state.get(decision.tile).last_result;
            let selected_last_frame = matches!(
                last,
                // Only count states where the tile (or its children) were
                // *actually present* in the selection list - not kicked variants.
                // Kicked = was briefly added then removed, so the parent
                // must remain the fallback until all siblings are renderable.
                TileRefinementResult::Selected | TileRefinementResult::Refined
            );
            TraversalDetails::leaf_ready(selected_last_frame)
        } else {
            TraversalDetails::leaf_not_ready()
        }
    } else {
        let mut d = TraversalDetails::empty();
        for c in child_details {
            d.combine(c);
        }
        d
    };

    if decision.self_selected {
        combined.all_ready = true;
        combined.any_selected_last_frame |= matches!(
            state.get(decision.tile).last_result,
            TileRefinementResult::Selected | TileRefinementResult::SelectedAndKicked
        );
    }
    combined
}

fn should_kick(
    ctx: &SelectionContext<'_>,
    decision: &VisitDecision,
    combined: TraversalDetails,
    state: &TileStates,
) -> bool {
    let kick_while_fading = ctx.options.streaming.enable_lod_transition
        && ctx.options.streaming.kick_descendants_while_fading_in
        && state.get(decision.tile).last_result == TileRefinementResult::Selected
        && state.get(decision.tile).lod_fade_pct < 1.0
        && decision.has_content;

    decision.refinement == RefinementMode::Replace
        && !decision.unconditionally_refined
        && decision.has_content
        && decision.self_ready
        && !decision.self_selected
        && (!combined.any_selected_last_frame || kick_while_fading)
        && !combined.all_ready
}

fn apply_kick(
    ctx: &SelectionContext<'_>,
    decision: &VisitDecision,
    combined: &mut TraversalDetails,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    kicked: &mut usize,
) {
    if ctx.options.streaming.enable_lod_transition {
        let kicked_end = buffers.selected.len();
        let selected_start = decision.selected_start;
        for i in selected_start..kicked_end {
            let kicked_node = buffers.selected[i];
            let last = state.get(kicked_node).last_result;
            let was_selected = last == TileRefinementResult::Selected
                || (last == TileRefinementResult::Refined
                    && ctx.store.refinement(kicked_node) == RefinementMode::Add);
            if was_selected {
                if buffers.fading_out_set.insert(kicked_node) {
                    buffers.fading_out.push(kicked_node);
                }
            }
        }
        let selected_start = decision.selected_start;
        buffers
            .fading_in
            .retain(|&n| !buffers.selected[selected_start..kicked_end].contains(&n));
    }

    for i in decision.selected_start..buffers.selected.len() {
        let r = state.get(buffers.selected[i]).last_result;
        state.get_mut(buffers.selected[i]).last_result = r.kick();
    }
    buffers.selected.truncate(decision.selected_start);
    buffers.selected_sse.truncate(decision.selected_start);
    buffers.selected_depths.truncate(decision.selected_start);
    buffers
        .selected_final_resolutions
        .truncate(decision.selected_start);

    let was_selected_last_frame =
        state.get(decision.tile).last_result == TileRefinementResult::Selected;
    // GAP-6: Drop the extra `!was_selected_last_frame` guard that kiban had but
    // cesium-native/CesiumJS do not.  The load-queue checkpoint should be
    // abandoned whenever the in-flight descendant count exceeds the limit,
    // regardless of whether the parent was selected in the previous frame.
    if combined.not_yet_ready_count > ctx.options.loading.loading_descendant_limit {
        buffers.load.truncate(decision.load_start);
    }

    // Respect content bounds visibility even in the kick path: if the
    // fallback parent's content is also outside all frustums, there is
    // nothing to show and we let the "not all renderable" signal propagate
    // up the tree.  Mirrors CesiumJS: kicks call `selectTile()` which
    // calls `contentVisibility()`.
    if decision.content_bounds_visible {
        buffers.selected.push(decision.tile);
        buffers.selected_sse.push(0.0);
        buffers.selected_depths.push(0);
        buffers.selected_final_resolutions.push(true);
        buffers.selected_set.insert(decision.tile);
        *kicked += 1;
        combined.all_ready = true;
    }
    combined.any_selected_last_frame = was_selected_last_frame;
}

fn update_fading(
    ctx: &SelectionContext<'_>,
    decision: &VisitDecision,
    state: &TileStates,
    buffers: &mut TraversalBuffers,
) {
    if !ctx.options.streaming.enable_lod_transition {
        return;
    }
    let is_selected = buffers.selected_set.contains(decision.tile);
    let last = state.get(decision.tile).last_result;
    // Use .original() so RenderedAndKicked / RefinedAndKicked are treated the
    // same as Rendered / Refined.  Without this, any tile that was briefly
    // kicked and then re-enters the render set would start a fresh 0->1 fade
    // every frame it oscillates - the source of per-frame alpha flicker.
    let was_selected = last.original() == TileRefinementResult::Selected
        || (last.original() == TileRefinementResult::Refined
            && decision.refinement == RefinementMode::Add);

    if is_selected && !was_selected {
        buffers.fading_in.push(decision.tile);
    } else if !is_selected && was_selected {
        if buffers.fading_out_set.insert(decision.tile) {
            buffers.fading_out.push(decision.tile);
        }
    }
}

fn commit_node_result(
    ctx: &SelectionContext<'_>,
    decision: &VisitDecision,
    now_secs: u64,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
) {
    let is_selected = buffers.selected_set.contains(decision.tile);
    let status = state.get_mut(decision.tile);
    status.last_result = if is_selected {
        TileRefinementResult::Selected
    } else if decision.refines {
        TileRefinementResult::Refined
    } else {
        TileRefinementResult::None
    };
    if is_selected {
        status.importance = 1.0;
    }
    if status.last_result != TileRefinementResult::None {
        status.last_touched_secs = now_secs;
    }

    // Prevent-holes.
    if ctx.options.loading.prevent_holes
        && decision.has_content
        && !load_state_is_terminal(decision.load_state)
        && !is_selected
        && !buffers.contains_load(decision.tile)
    {
        buffers.push_load(LoadRequest::new(
            decision.tile,
            LoadPriority::urgent(0.0),
            0.0,
            0.0,
            0,
            0.0,
            bounding_sphere_radius(ctx.store.bounds(decision.tile)),
        ));
    }

    // preloadAncestors.
    if ctx.options.loading.preload_ancestors
        && decision.refines
        && decision.has_content
        && !decision.queued_for_load
        && !load_state_is_terminal(decision.load_state)
        && !decision.ancestor_meets_sse
        && !buffers.contains_load(decision.tile)
    {
        buffers.push_load(LoadRequest::new(
            decision.tile,
            LoadPriority::preload(),
            0.0,
            0.0,
            0,
            0.0,
            bounding_sphere_radius(ctx.store.bounds(decision.tile)),
        ));
    }
}

fn finalize_node(
    ctx: &SelectionContext<'_>,
    frame: &FrameLocals,
    decision: &VisitDecision,
    state: &mut TileStates,
    buffers: &mut TraversalBuffers,
    kicked: &mut usize,
) {
    let mut combined = combine_child_details(decision, state, buffers);
    if should_kick(ctx, decision, combined, state) {
        apply_kick(ctx, decision, &mut combined, state, buffers, kicked);
    }
    update_fading(ctx, decision, state, buffers);
    commit_node_result(ctx, decision, frame.now_secs, state, buffers);
    buffers.detail_stack.push(combined);
}

/// Normalize load priority scores using the 4-component CesiumJS formula.
///
/// If renderable content has exceeded its max age, transition to Expiring.
#[inline]
fn check_and_mark_expiry(
    tile: TileId,
    store: &TileStore,
    state: &mut TileStates,
    now_secs: u64,
    policy: &dyn EvictionPolicy,
) {
    if policy.should_evict(tile, state.get(tile), store, now_secs) {
        state.mark_expiring(tile);
    }
}

#[inline]
fn build_snapshot<'a>(tile: TileId, store: &'a TileStore, state: &TileStates) -> NodeSnapshot<'a> {
    NodeSnapshot {
        tile,
        bounds: store.bounds(tile),
        geometric_error: store.geometric_error(tile),
        has_content: !store.content_keys(tile).is_empty(),
        refinement: store.refinement(tile),
        unconditionally_refined: store
            .flags(tile)
            .contains(TileFlags::UNCONDITIONALLY_REFINED),
        load_state: state.get(tile).load_state,
        last_result: state.get(tile).last_result,
        children_len: store.children(tile).len(),
        children: store.children(tile),
        children_within_parent: store
            .flags(tile)
            .contains(TileFlags::CHILDREN_WITHIN_PARENT),
    }
}

fn compute_max_sse_with_dynamic(
    snap: &NodeSnapshot<'_>,
    ctx: &SelectionContext<'_>,
    distances: &[f64],
    dynamic_sse_densities: &[f64],
) -> f64 {
    if distances.len() == ctx.views.len() {
        ctx.views
            .iter()
            .enumerate()
            .zip(distances.iter())
            .map(|((vi, v), &d)| {
                let mut sse = compute_sse_at_distance(
                    snap.geometric_error,
                    v,
                    d,
                    ctx.options.lod.minimum_camera_distance,
                );
                if let Some(&density) = dynamic_sse_densities.get(vi) {
                    if density > 0.0 {
                        let scalar = d * density;
                        let dynamic_error = (1.0 - (-(scalar * scalar)).exp())
                            * ctx.options.lod.dynamic_detail_reduction_factor;
                        sse = (sse - dynamic_error).max(0.0);
                    }
                }
                sse
            })
            .fold(0.0_f64, f64::max)
    } else {
        ctx.views
            .iter()
            .map(|v| compute_sse(snap.geometric_error, v, snap.bounds))
            .fold(0.0_f64, f64::max)
    }
}

fn foveated_factor(snap: &NodeSnapshot<'_>, views: &[ViewState]) -> f64 {
    foveated_factor_from_bounds_and_views(snap.bounds, views)
}

fn foveated_factor_from_bounds_and_views(bounds: &SpatialBounds, views: &[ViewState]) -> f64 {
    let center = match bounds {
        SpatialBounds::Obb(o) => o.center,
        SpatialBounds::Sphere(s) => s.center,
        SpatialBounds::Aabb(a) => a.center(),
        _ => return 0.0,
    };
    let mut best = 1.0_f64;
    for view in views {
        let to_center = center - view.position;
        let dist = to_center.length();
        if dist >= 1e-5 {
            let cos_angle = (to_center / dist).dot(view.direction);
            let ff = 1.0 - cos_angle.abs();
            if ff < best {
                best = ff;
            }
        }
    }
    best
}

fn is_foveated_deferred(snap: &NodeSnapshot<'_>, ctx: &SelectionContext<'_>) -> bool {
    is_foveated_deferred_for_bounds(snap.bounds, ctx)
}

fn is_foveated_deferred_for_bounds(bounds: &SpatialBounds, ctx: &SelectionContext<'_>) -> bool {
    let lod = &ctx.options.lod;
    let max_foveated = ctx
        .views
        .iter()
        .map(|v| match &v.projection {
            Projection::Perspective { fov_y, .. } => 1.0 - (*fov_y * 0.5).cos(),
            Projection::Orthographic { .. } => 0.0,
        })
        .fold(0.0_f64, f64::max);

    if max_foveated <= 0.0 {
        return false;
    }
    let cone_factor = lod.foveated_cone_size * max_foveated;
    foveated_factor_from_bounds_and_views(bounds, ctx.views) > cone_factor
}

fn is_zoom_out_transition(snap: &NodeSnapshot<'_>, sse: &SseDecision, lod_active: bool) -> bool {
    lod_active
        && !sse.refines
        && matches!(
            snap.load_state,
            TileLoadState::Ready | TileLoadState::Expiring
        )
        && snap.last_result.original() == TileRefinementResult::Refined
}

#[inline]
fn compute_load_score(views: &[ViewState], bounds: &SpatialBounds) -> f32 {
    let center = match bounds {
        SpatialBounds::Obb(o) => o.center,
        SpatialBounds::Sphere(s) => s.center,
        SpatialBounds::Aabb(a) => a.center(),
        _ => return 0.0,
    };
    let mut best = f64::MAX;
    for view in views {
        let to_center = center - view.position;
        let dist = to_center.length();
        if dist >= 1e-5 {
            let cos_angle = (to_center / dist).dot(view.direction);
            let priority = (1.0 - cos_angle) * dist;
            if priority < best {
                best = priority;
            }
        }
    }
    (-(if best == f64::MAX { 0.0 } else { best })) as f32
}

#[inline]
fn views_inside_volume(view: &ViewState, volume: &SpatialBounds) -> bool {
    let p = view.position;
    match volume {
        SpatialBounds::Obb(o) => {
            let to_p = p - o.center;
            let x = to_p.dot(o.half_axes.x_axis);
            let y = to_p.dot(o.half_axes.y_axis);
            let z = to_p.dot(o.half_axes.z_axis);
            let ex = o.half_axes.x_axis.length();
            let ey = o.half_axes.y_axis.length();
            let ez = o.half_axes.z_axis.length();
            x.abs() <= ex && y.abs() <= ey && z.abs() <= ez
        }
        SpatialBounds::Sphere(s) => (p - s.center).length() <= s.radius,
        _ => true,
    }
}

#[inline]
fn bounds_entirely_clipped(bounds: &SpatialBounds, plane: &Plane) -> bool {
    let normal = plane.normal;
    let support = match bounds {
        SpatialBounds::Obb(o) => {
            let c_proj = o.center.dot(normal);
            let r = normal.dot(o.half_axes.x_axis).abs()
                + normal.dot(o.half_axes.y_axis).abs()
                + normal.dot(o.half_axes.z_axis).abs();
            c_proj + r
        }
        SpatialBounds::Sphere(s) => s.center.dot(normal) + s.radius,
        SpatialBounds::Aabb(a) => {
            let px = if normal.x >= 0.0 { a.max.x } else { a.min.x };
            let py = if normal.y >= 0.0 { a.max.y } else { a.min.y };
            let pz = if normal.z >= 0.0 { a.max.z } else { a.min.z };
            DVec3::new(px, py, pz).dot(normal)
        }
        _ => return false,
    };
    support + plane.distance < 0.0
}

#[inline]
fn load_state_is_terminal(state: TileLoadState) -> bool {
    matches!(
        state,
        TileLoadState::Ready | TileLoadState::Failed | TileLoadState::Expiring
    )
}

fn interpolate_fog_density(table: &[(f64, f64)], height: f64) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    if height <= table[0].0 {
        return table[0].1;
    }
    let last = table.last().unwrap();
    if height >= last.0 {
        return 0.0;
    }
    let idx = table.partition_point(|&(h, _)| h <= height);
    let (h0, d0) = table[idx - 1];
    let (h1, d1) = table[idx];
    let t = (height - h0) / (h1 - h0);
    d0 + t * (d1 - d0)
}

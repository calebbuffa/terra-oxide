//! Output types for the selection algorithm.
//!
//! [`ViewUpdateResult`] is the public result of one `update_view_group()` call,
//! mirroring `Cesium3DTilesSelection::ViewUpdateResult`.
//!
//! [`SelectionOutput`] is the internal full output of `select()`, which also
//! carries the load queue - split out before becoming a `ViewUpdateResult`.
//!
//! [`LoadEvent`] is the normalised output of a completed tile load, produced
//! by `LoadScheduler::drain()` and consumed by `NodeStates::apply()`.

use crate::tile_store::{TileDescriptor, TileId};
use std::collections::HashMap;

/// Load priority tier - an open newtype so callers can define custom tiers
/// without modifying kiban.
///
/// Ordered from lowest to highest urgency by the inner `u8` so that `Ord`
/// comparisons and `max`-based selection naturally prefer the most urgent tier.
/// The four built-in tiers use widely-spaced values so user-defined constants
/// can be slotted between them:
///
/// | Constant              | Value | Use                                                 |
/// |-----------------------|-------|-----------------------------------------------------|
/// | `PriorityGroup::PRELOAD`  |   0 | Off-screen siblings loaded speculatively.         |
/// | `PriorityGroup::DEFERRED` |  64 | Outside the foveated cone; loads after `NORMAL`.  |
/// | `PriorityGroup::NORMAL`   | 128 | In-frustum tiles that need content for SSE.       |
/// | `PriorityGroup::URGENT`   | 255 | `mustContinueRefining` - avoids a visible hole.   |
///
/// # Extension
///
/// ```rust
/// use kiban::PriorityGroup;
/// // A tier between NORMAL and URGENT:
/// pub const HIGH_PRIORITY: PriorityGroup = PriorityGroup(192);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PriorityGroup(pub u8);

impl PriorityGroup {
    /// Off-screen siblings / ancestors loaded speculatively.
    pub const PRELOAD: Self = Self(0);
    /// Foveated-deferred: outside the center-of-view cone.
    /// Loads after all `NORMAL`-priority (in-cone) tiles have been issued.
    pub const DEFERRED: Self = Self(64);
    /// In-frustum tiles that need content to meet the SSE threshold.
    pub const NORMAL: Self = Self(128);
    /// `mustContinueRefiningToDeeperTiles` - avoids a visible hole.
    pub const URGENT: Self = Self(255);
}

#[derive(Clone, Copy, Debug)]
pub struct LoadPriority {
    pub group: PriorityGroup,
    /// Higher score = load sooner within the same group.
    pub score: f32,
}

impl LoadPriority {
    pub fn normal(score: f32) -> Self {
        Self {
            group: PriorityGroup::NORMAL,
            score,
        }
    }
    pub fn urgent(score: f32) -> Self {
        Self {
            group: PriorityGroup::URGENT,
            score,
        }
    }
    pub fn preload() -> Self {
        Self {
            group: PriorityGroup::PRELOAD,
            score: 0.0,
        }
    }

    /// Preload with a priority score, so off-screen siblings are loaded in
    /// look-direction order rather than all at equal priority.
    pub fn preload_with_score(score: f32) -> Self {
        Self {
            group: PriorityGroup::PRELOAD,
            score,
        }
    }

    /// Foveated-deferred: tile is outside the center-of-view cone.
    /// Loaded after all `Normal`-priority (in-cone) tiles have been issued.
    /// Matches CesiumJS `priorityDeferred = true` path in `updatePriority()`.
    pub fn deferred(score: f32) -> Self {
        Self {
            group: PriorityGroup::DEFERRED,
            score,
        }
    }
}

/// A request to load content for a tile emitted by the selection algorithm.
#[derive(Clone, Debug)]
pub struct LoadRequest {
    pub tile: TileId,
    pub priority: LoadPriority,
    /// Raw distance to camera (from priority holder).  Used by the
    /// multi-component normalisation step at the end of `select()`.
    pub(crate) raw_distance: f64,
    /// Raw foveated factor in [0..1] (from priority holder).  0 = looking
    /// directly at the tile.
    pub(crate) raw_foveated_factor: f64,
    /// Depth of the tile in the tree.  Used for `prefer_leaves` scoring.
    pub(crate) raw_depth: u32,
    /// Reverse screen-space error = 1/SSE.  Higher SSE -> lower rSSE ->
    /// should load sooner.  0 when SSE is 0.
    pub(crate) raw_reverse_sse: f64,
    /// Approximate bounding-sphere radius of this tile (metres).
    /// Captured at request-creation time; used by [`DispatchGate`]s so they
    /// do not need access to the [`TileStore`].
    pub(crate) bounding_radius: f64,
}

impl LoadRequest {
    /// Construct a `LoadRequest` from its individual components.
    ///
    /// Centralises all field names so call sites only need to supply values.
    /// If new raw metrics are added later, only this constructor changes.
    #[inline]
    pub(crate) fn new(
        tile: TileId,
        priority: LoadPriority,
        raw_distance: f64,
        raw_foveated_factor: f64,
        raw_depth: u32,
        raw_reverse_sse: f64,
        bounding_radius: f64,
    ) -> Self {
        Self {
            tile,
            priority,
            raw_distance,
            raw_foveated_factor,
            raw_depth,
            raw_reverse_sse,
            bounding_radius,
        }
    }
}

/// Result of expanding latent children for a tile.
pub enum ExpandResult {
    None,
    RetryLater,
    Children(Vec<TileDescriptor>),
}

/// `tiles_fading_out` is **persistent**: tiles are inserted when they leave
/// the render set and are only removed once their fade percentage reaches 1.0
/// (fully transparent).
#[derive(Default)]
pub struct ViewUpdateResult {
    /// Nodes whose content should be rendered this frame.
    pub selected_tiles: Vec<TileId>,
    /// Screen-space error for each tile in `tiles_to_render_this_frame` (parallel vec).
    ///
    /// Index `i` corresponds to `tiles_to_render_this_frame[i]`.
    /// Mirrors `ViewUpdateResult::tileScreenSpaceErrorThisFrame`.
    pub tile_screen_space_errors: Vec<f32>,
    /// Skip-LOD stencil selection depth for each tile in `tiles_to_render_this_frame` (parallel vec).
    ///
    /// Index `i` corresponds to `tiles_to_render_this_frame[i]`.
    /// A depth of 0 means normal rendering; depth > 0 means this tile is being rendered
    /// simultaneously with a lower-detail ancestor (mixed content).  Use as a stencil
    /// value: render tiles with larger depths before tiles with smaller depths so
    /// children are drawn on top of ancestors regardless of z-depth.
    ///
    /// Always all-zeros when `skip_level_of_detail` is false.
    /// Mirrors CesiumJS `tile._selectionDepth`.
    pub tile_selection_depths: Vec<u32>,
    /// Per-tile final-resolution flag, parallel to `selected_tiles`.
    /// `true` = leaf/foreground tile; `false` = background ancestor tile rendered
    /// simultaneously with higher-detail descendants (skip-LOD mixed content).
    /// Always all-`true` when `skip_level_of_detail` is `false`.
    /// Mirrors CesiumJS `tile._finalResolution`.
    pub tile_final_resolutions: Vec<bool>,
    /// True when any tile has `selection_depth > 0` this frame.
    /// Callers should clear the stencil buffer before rendering when this is true.
    /// Mirrors CesiumJS `Cesium3DTileset.hasMixedContent`.
    pub has_mixed_selection: bool,
    /// Nodes fading into the render set.
    pub tiles_fading_in: Vec<TileId>,
    /// Nodes fading out of the render set (still rendered while fading).
    ///
    /// **Persistent across frames.** Rebuilt each frame by merging traversal
    /// output; advanced toward 1.0 via delta_time; entries removed only when
    /// their percentage reaches 1.0.  Callers should render these tiles at
    /// `1.0 - fade_percentage` alpha (they are becoming invisible).
    pub tiles_fading_out: Vec<TileId>,
    /// Per-tile fade percentage in `[0.0, 1.0]`.
    ///
    /// **Both** fading-in (`tiles_fading_in`) and fading-out (`tiles_fading_out`)
    /// nodes use this map.  For fading-in tiles render alpha =
    /// `fade_percentage`; for fading-out tiles render alpha =
    /// `1.0 - fade_percentage`.
    pub tile_fade_percentages: HashMap<TileId, f32>,
    /// Number of tiles visited during traversal.
    pub tiles_visited: u32,
    /// Number of tiles culled during traversal.
    pub tiles_culled: u32,
    /// Number of tiles that were frustum-culled but still visited (force-visit path).
    pub tiles_culled_but_visited: u32,
    /// Number of tiles kicked from the render list.
    pub tiles_kicked: u32,
    /// Maximum depth reached in the tile tree during this traversal.
    pub max_depth_visited: u32,
    /// Frame counter - incremented each `update_view_group` call.
    pub frame_number: u64,
    /// Number of tiles currently fetching content over the network.
    pub tiles_loading: u32,
    /// Number of tiles whose content is fully resident and renderable.
    pub tiles_ready: u32,
    /// Number of tiles that have permanently failed to load.
    pub tiles_failed: u32,
    /// Total resident content size in bytes across all `Renderable` and
    /// `Expiring` tiles.
    pub resident_bytes: usize,
    /// Effective SSE threshold used for traversal this frame, which may
    /// differ from the configured `maximum_screen_space_error` when the
    /// engine is under memory pressure.
    ///
    /// When `resident_bytes > max_cached_bytes + maximum_cache_overflow_bytes`
    /// this value is raised by 2% per frame (coarser rendering = less memory).
    /// When `resident_bytes < max_cached_bytes` it ratchets back down toward
    /// the configured nominal threshold.
    pub memory_adjusted_screen_space_error: f64,
}

/// Internal complete output of one `select_tiles()` pass.
///
/// Carries the load queue in addition to the render list; the load queue is
/// consumed by [`ContentManager::load_tiles`] and never exposed publicly.
#[derive(Default)]
pub struct SelectionOutput {
    pub selected: Vec<TileId>,
    /// SSE for each entry in `render` (parallel vec).
    pub selected_sse: Vec<f32>,
    pub load: Vec<LoadRequest>,
    pub fading_in: Vec<TileId>,
    pub fading_out: Vec<TileId>,
    pub nodes_visited: usize,
    pub nodes_culled: usize,
    /// Nodes that were frustum-culled but still visited (force-visit path).
    pub nodes_culled_but_visited: usize,
    pub nodes_kicked: usize,
    pub max_depth: u32,
    /// Skip-LOD stencil selection depths, parallel to `selected`.
    /// Entry `i` is the selection depth for `selected[i]`.
    /// Non-zero only when `skip_level_of_detail` is active and the tile is
    /// being rendered simultaneously with a lower-detail ancestor (mixed content).
    pub selection_depths: Vec<u32>,
    /// Per-tile final-resolution flags, parallel to `selected`.
    /// `true` = leaf/foreground tile; `false` = background ancestor rendered
    /// simultaneously with higher-detail descendants (skip-LOD mixed content).
    /// Always all-`true` when `skip_level_of_detail` is `false`.
    pub selection_final_resolutions: Vec<bool>,
    /// True when any tile in `render` has `selection_depth > 0` - i.e., a
    /// child tile and an ancestor are both selected this frame.
    pub has_mixed_selection: bool,
    /// Number of tiles that were refined (children pushed) during traversal.
    pub nodes_refined: usize,
}

/// Normalised output of a completed tile load.
///
/// Produced by `LoadScheduler::drain()` and consumed by
/// `NodeStates::apply(&event)` - drives state-machine transitions.
///
/// `ExternalTileset` and `StoreInit` are handled entirely inside
/// `LoadScheduler::drain()` before the events reach `NodeStates`.
pub enum LoadEvent {
    /// Tile geometry loaded successfully; model ready for caller to prepare.
    Loaded {
        tile: TileId,
        model: moderu::GltfModel,
    },
    /// Tile is an empty tile (no geometry); mark directly ready.
    Empty { tile: TileId },
    /// Tile load produced custom (non-glTF) content; passed through to the
    /// caller via [`TileEvent::CustomTileLoaded`].
    ///
    /// [`TileEvent::CustomTileLoaded`]: crate::TileEvent::CustomTileLoaded
    Custom {
        tile: TileId,
        content: std::sync::Arc<dyn std::any::Any + Send + Sync>,
    },
    /// Tile load failed permanently.
    Failed {
        tile: TileId,
        /// Source URL of the failed tile, if known.
        url: Option<String>,
        /// Human-readable failure reason.
        message: String,
    },
    /// Loader is not ready; schedule a retry with exponential backoff.
    RetryLater { tile: TileId },
}

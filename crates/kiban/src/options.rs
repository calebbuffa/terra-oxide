/// Options governing content load scheduling, retries, and memory budget.
///
/// Accessed via `engine.options().loading`.
#[derive(Clone, Debug)]
pub struct LoadingOptions {
    /// Maximum number of children simultaneously in the `Loading` state before the
    /// traversal stops descending further.
    pub loading_descendant_limit: usize,

    /// If `true`, the selection must not produce holes - a parent tile is always
    /// selected as a fallback if replacement children are not yet `Renderable`.
    pub prevent_holes: bool,

    /// If `true`, ancestors of rendered nodes are pre-loaded at `Preload` priority.
    pub preload_ancestors: bool,

    /// If `true`, siblings of rendered nodes are pre-loaded at `Preload` priority.
    pub preload_siblings: bool,

    /// Maximum new load requests to dispatch per load pass.
    pub max_simultaneous_loads: usize,

    /// Maximum load retry attempts before a tile transitions to `Failed`.
    pub retry_limit: u8,

    /// Frames to wait before re-queuing a `RetryScheduled` tile.
    pub retry_backoff_frames: u32,

    /// Memory ceiling in bytes for resident content. Eviction is triggered when exceeded.
    /// Mirrors CesiumJS `cacheBytes` (default 512 MiB).
    pub max_cached_bytes: usize,

    /// Additional headroom above `max_cached_bytes` tolerated before the engine
    /// begins raising `memory_adjusted_screen_space_error`.
    ///
    /// When total resident content exceeds `max_cached_bytes + maximum_cache_overflow_bytes`,
    /// `effective_screen_space_error` is multiplied by 1.02 each frame until
    /// memory pressure eases.  Mirrors CesiumJS `maximumCacheOverflowBytes` (default 512 MiB).
    pub maximum_cache_overflow_bytes: usize,

    /// Capacity of the bounded channel used to deliver completed load results
    /// from background threads to the main thread (default `256`).
    /// Increase if your tileset can have many concurrent loads completing in a
    /// single frame; decrease to bound memory if your thread pool is large.
    pub result_channel_capacity: usize,

    /// Load tiles even when the tileset is not visible.
    ///
    /// When `true`, the tileset will continue loading tiles in the background
    /// even if the layer is hidden. This allows instant display when shown.
    /// Default: `false`.
    pub preload_when_hidden: bool,

    /// Preload tiles at camera flight destination during a flight.
    ///
    /// When `true` and the camera is on a flight path (large movement),
    /// attempt to also load tiles at the flight destination.
    /// Requires integrating camera destination hints from the application via
    /// [`Layer::set_preload_hint`].
    /// Default: `false`.
    ///
    /// [`Layer::set_preload_hint`]: crate::Layer::set_preload_hint
    pub preload_flight_destinations: bool,
}

impl Default for LoadingOptions {
    fn default() -> Self {
        Self {
            loading_descendant_limit: 20,
            prevent_holes: true,
            preload_ancestors: true,
            preload_siblings: true,
            max_simultaneous_loads: 20,
            retry_limit: 3,
            retry_backoff_frames: 8,
            max_cached_bytes: 512 * 1024 * 1024,
            maximum_cache_overflow_bytes: 512 * 1024 * 1024,
            result_channel_capacity: 256,
            preload_when_hidden: false,
            preload_flight_destinations: false,
        }
    }
}

/// Options governing frustum, occlusion, fog, and clipping-plane culling.
///
/// Accessed via `engine.options().culling`.
#[derive(Clone, Debug)]
pub struct CullingOptions {
    /// Whether frustum culling is enabled.
    pub enable_frustum_culling: bool,

    /// Whether occlusion culling is enabled - occluded nodes are removed from the
    /// render set. Requires an occlusion tester.
    pub enable_occlusion_culling: bool,

    /// Whether to delay refinement (descent into children) for occluded nodes.
    pub delay_refinement_for_occlusion: bool,

    /// If `true`, apply fog-density-based culling.
    pub enable_fog_culling: bool,

    /// Fog density lookup table: `(height_above_ellipsoid, fog_density)` pairs,
    /// sorted by ascending height. Only used when `enable_fog_culling` is `true`.
    pub fog_density_table: Vec<(f64, f64)>,

    /// Secondary screen-space error applied to culled nodes.
    pub culled_screen_space_error: f64,

    /// If `true`, apply [`culled_screen_space_error`](Self::culled_screen_space_error) to culled nodes.
    pub enforce_culled_screen_space_error: bool,

    /// Fog opacity threshold above which a tile is considered fully fogged and
    /// culled from the render set.  Must be in `(0, 1)` - default `0.9999`.
    pub fog_opacity_cull_threshold: f64,

    /// If `true`, nodes directly below the camera are always included even if outside the frustum.
    pub render_nodes_under_camera: bool,

    /// Clipping planes applied to the entire spatial hierarchy.
    /// Each plane's visible side is `normal x p + distance >= 0`.
    pub clipping_planes: Vec<zukei::Plane>,
}

/// Default fog density table from cesium-native `TilesetOptions.h`.
/// Maps camera height above ellipsoid (metres) to fog density.
/// Sorted ascending by height - used by `interpolate_fog_density`.
pub const DEFAULT_FOG_DENSITY_TABLE: &[(f64, f64)] = &[
    (359.393, 2.0e-5),
    (800.749, 2.0e-4),
    (1_275.650_1, 1.0e-4),
    (2_151.119_2, 7.0e-5),
    (3_141.776_3, 5.0e-5),
    (4_777.519_8, 4.0e-5),
    (6_281.249_3, 3.0e-5),
    (12_364.307, 1.9e-5),
    (15_900.765, 1.0e-5),
    (49_889.054_9, 8.5e-6),
    (78_026.825_9, 6.2e-6),
    (99_260.734_4, 5.8e-6),
    (120_036.387_3, 5.3e-6),
    (151_011.015_8, 5.2e-6),
    (156_091.195_3, 5.1e-6),
    (203_849.311_2, 4.2e-6),
    (274_866.980_3, 4.0e-6),
    (319_916.314_9, 3.4e-6),
    (493_552.052_8, 2.6e-6),
    (628_733.587_4, 2.2e-6),
    (1_000_000.0, 0.0),
];

impl Default for CullingOptions {
    fn default() -> Self {
        Self {
            enable_frustum_culling: true,
            enable_occlusion_culling: false,
            delay_refinement_for_occlusion: true,
            enable_fog_culling: true,
            fog_density_table: DEFAULT_FOG_DENSITY_TABLE.to_vec(),
            culled_screen_space_error: 64.0,
            enforce_culled_screen_space_error: true,
            fog_opacity_cull_threshold: 0.9999,
            render_nodes_under_camera: true,
            clipping_planes: Vec::new(),
        }
    }
}

/// Options governing LOD refinement heuristics: skip-LOD, dynamic reduction,
/// foveation, and progressive resolution.
///
/// Accessed via `engine.options().lod`.
#[derive(Clone, Debug)]
pub struct LodRefinementOptions {
    /// If `true`, enables skip-LOD: the engine may skip intermediate LOD levels.
    pub skip_level_of_detail: bool,

    /// Factor by which parent LOD metric must exceed threshold to trigger a skip.
    pub skip_lod_metric_factor: f64,

    /// Minimum LOD metric value to be a skip candidate.
    pub base_lod_metric_threshold: f64,

    /// Minimum levels to skip between consecutively rendered nodes when skip-LOD is on.
    pub skip_levels: u32,

    /// When `true` and skip-LOD enabled, only the desired tile is downloaded - no placeholder.
    pub immediately_load_desired_lod: bool,

    /// If `true`, reduces effective SSE for distant tiles when the camera is near
    /// the horizon, matching CesiumJS `dynamicScreenSpaceError` (default `true`).
    ///
    /// For each tile the engine subtracts
    /// `fog(distance, computed_density) * factor` from its raw SSE before the
    /// refinement threshold test, where `fog(d, ρ) = 1 - exp(-(dxρ)^2)` and
    /// `computed_density = density * (1 - |dot(camDir, up)|) * (1 - heightT)`.
    /// Effect is strongest at street/horizon views and vanishes when viewing
    /// from altitude, dramatically reducing tile counts in dense city scenes.
    pub enable_dynamic_detail_reduction: bool,

    /// Base fog density used in the dynamic SSE computation.
    /// Matches CesiumJS `dynamicScreenSpaceErrorDensity` (default `2.0e-4`).
    pub dynamic_detail_reduction_density: f64,

    /// Multiplier applied to the fog term before subtracting from SSE.
    /// Matches CesiumJS `dynamicScreenSpaceErrorFactor` (default `24.0`).
    pub dynamic_detail_reduction_factor: f64,

    /// Fraction of the tileset height range above which the dynamic SSE effect
    /// starts to fade.  Matches CesiumJS `dynamicScreenSpaceErrorHeightFalloff`
    /// (default `0.25`).
    ///
    /// When the camera height `h` is in `[minH + fraction*(maxH-minH), maxH]`
    /// the horizon factor is linearly interpolated to zero so the reduction has
    /// no effect at altitude.  kiban approximates this without tileset bounds
    /// by using only the horizon component.
    pub dynamic_detail_reduction_height_falloff: f64,

    /// If `true`, load requests for tiles outside the foveated center cone are
    /// assigned `Deferred` priority so center tiles load first.
    /// Matches CesiumJS `foveatedScreenSpaceError` (default `true`).
    pub enable_foveated_rendering: bool,

    /// Fraction (0..1) of the half-FOV cone treated as the foveated center.
    /// Tiles with an angular offset < `cone_size * half_fov` are not deferred.
    /// Matches CesiumJS `foveatedConeSize` (default `0.1`).
    pub foveated_cone_size: f64,

    /// Minimum LOD metric multiplier (0..1) applied to peripheral nodes.
    pub foveated_min_lod_metric_relaxation: f64,

    /// Seconds after the camera stops before peripheral nodes ramp back to full detail.
    pub foveated_time_delay: f32,

    /// If `true`, temporarily lowers the effective LOD threshold for progressive streaming.
    pub enable_progressive_resolution: bool,

    /// Fraction of viewport height used for the progressive-resolution pass.
    pub progressive_resolution_height_fraction: f64,

    /// Minimum camera-to-tile distance (metres) used as the denominator floor
    /// in the SSE formula.  Prevents divide-by-zero when the camera is inside
    /// a tile's bounding volume (default `1.0`).
    pub minimum_camera_distance: f64,

    /// Depth limit for descendant selection during skip-LOD traversal.
    /// When a skip candidate has not yet loaded, the engine selects up to this
    /// many levels of its descendants as placeholders (default `2`).
    pub skip_lod_fallback_depth: u32,
}

impl Default for LodRefinementOptions {
    fn default() -> Self {
        Self {
            skip_level_of_detail: false,
            skip_lod_metric_factor: 16.0,
            base_lod_metric_threshold: 1024.0,
            skip_levels: 1,
            immediately_load_desired_lod: false,
            enable_dynamic_detail_reduction: true,
            dynamic_detail_reduction_density: 2.0e-4,
            dynamic_detail_reduction_factor: 24.0,
            dynamic_detail_reduction_height_falloff: 0.25,
            enable_foveated_rendering: true,
            foveated_cone_size: 0.1,
            foveated_min_lod_metric_relaxation: 0.0,
            foveated_time_delay: 0.2,
            enable_progressive_resolution: false,
            progressive_resolution_height_fraction: 0.3,
            minimum_camera_distance: 1.0,
            skip_lod_fallback_depth: 2,
        }
    }
}

/// Options governing load prioritisation, flight preloading, and LOD transitions.
///
/// Accessed via `engine.options().streaming`.
#[derive(Clone, Debug)]
pub struct StreamingOptions {
    /// If `true`, when skip-LOD skips a level, siblings at that level are preloaded.
    pub load_siblings_on_skip: bool,

    /// Cancel `Normal`-priority requests while the camera is moving fast.
    pub cull_requests_while_moving: bool,

    /// Speed multiplier threshold for `cull_requests_while_moving`.
    pub cull_requests_while_moving_multiplier: f64,

    /// If `true`, leaf tiles are assigned higher load priority than interior tiles.
    ///
    /// When `false` (default), tiles closer to the camera root load first (breadth-first
    /// favouring).  When `true`, the depth component of the composite priority score is
    /// inverted so the most-detailed tiles in view always load before their ancestors.
    ///
    /// Mirrors CesiumJS `Cesium3DTileset.preferLeaves` (default `false`).
    pub prefer_leaves: bool,

    /// If `true`, newly-visible nodes fade in over `lod_transition_length` seconds
    /// instead of popping in. Matches cesium-native `enableLodTransitionPeriod`.
    pub enable_lod_transition: bool,

    /// Duration in seconds of the fade-in/fade-out LOD transition.
    /// Matches cesium-native `lodTransitionLength` (default 1.0 s).
    pub lod_transition_length: f32,

    /// If `true` and `enable_lod_transition` is on, descendants are kicked while
    /// their parent is still fading in, preventing pop-in of children over a
    /// partially-transparent parent. Matches cesium-native `kickDescendantsWhileFadingIn`.
    pub kick_descendants_while_fading_in: bool,

    /// Minimum camera displacement (world-units) per frame that counts as
    /// "moving" for the purposes of cull-requests-while-moving gating
    /// (default `1e-3`).
    pub camera_movement_threshold: f64,
}

impl Default for StreamingOptions {
    fn default() -> Self {
        Self {
            load_siblings_on_skip: false,
            cull_requests_while_moving: true,
            cull_requests_while_moving_multiplier: 60.0,
            prefer_leaves: false,
            enable_lod_transition: false,
            lod_transition_length: 1.0,
            kick_descendants_while_fading_in: true,
            camera_movement_threshold: 1e-3,
        }
    }
}

/// Debug-only engine options.
///
/// Accessed via `engine.options().debug`.
#[derive(Clone, Debug, Default)]
pub struct DebugOptions {
    /// When `true`, the engine skips traversal and returns the previous frame's result.
    pub enable_freeze_frame: bool,
}

/// Core engine options, grouped into five nested structs.
///
/// # Example
/// ```rust,ignore
/// let mut opts = engine.options().clone();
/// opts.loading.max_simultaneous_loads = 8;
/// opts.culling.enable_frustum_culling = false;
/// engine.set_options(opts);
/// ```
#[derive(Clone, Debug, Default)]
pub struct SelectionOptions {
    pub loading: LoadingOptions,
    pub culling: CullingOptions,
    pub lod: LodRefinementOptions,
    pub streaming: StreamingOptions,
    pub debug: DebugOptions,
}

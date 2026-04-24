use glam::DVec3 as Vec3;

use std::collections::HashSet;
use std::sync::Arc;

use crate::fade::{FadeStrategy, LinearFadeStrategy};
use crate::frame_decision::{LoadRequest, SelectionOutput, ViewUpdateResult};
use crate::options::StreamingOptions;
use crate::selection_state::TileStates;
use crate::tile_store::TileId;
use crate::traversal::TraversalBuffers;

/// Projection model for a view.
#[derive(Clone, Debug)]
pub enum Projection {
    /// Symmetric perspective projection.
    Perspective {
        /// Horizontal field-of-view angle in radians.
        fov_x: f64,
        /// Vertical field-of-view angle in radians.
        fov_y: f64,
    },
    /// Orthographic projection.
    Orthographic {
        /// Half-width of the view volume in world units.
        half_width: f64,
        /// Half-height of the view volume in world units.
        half_height: f64,
    },
}

/// Per-view camera state passed into each `update_view_group` call.
/// All positions and directions are in the engine's working coordinate system.
#[derive(Clone, Debug)]
pub struct ViewState {
    /// Viewport dimensions in physical pixels, `[width, height]`.
    pub viewport_px: [u32; 2],
    /// Camera world-space position.
    pub position: Vec3,
    /// Camera view direction (unit-length, world-space).
    pub direction: Vec3,
    /// Camera up vector (unit-length, world-space).
    pub up: Vec3,
    /// Projection model (perspective or orthographic).
    pub projection: Projection,
    /// Multiplier applied to the raw LOD metric before passing to `LodEvaluator`.
    /// Use values > 1.0 to over-load (sharper detail); < 1.0 to under-load.
    pub lod_metric_multiplier: f32,
    /// Reference ellipsoid for geodetic computations (ECEF -> cartographic).
    ///
    /// When set, enables `render_nodes_under_camera` to convert the ECEF
    /// position to cartographic and test geographic containment against tile
    /// bounding regions.
    pub ellipsoid: Option<terra::Ellipsoid>,
}

impl ViewState {
    /// Create a perspective view state.
    ///
    /// If `ellipsoid` is provided, `position_cartographic` is computed from
    /// the ECEF `position`. Pass `Some(&Ellipsoid::wgs84())` for geospatial data.
    pub fn perspective(
        position: Vec3,
        direction: Vec3,
        up: Vec3,
        viewport_px: [u32; 2],
        fov_x: f64,
        fov_y: f64,
    ) -> Self {
        Self {
            viewport_px,
            position,
            direction,
            up,
            projection: Projection::Perspective { fov_x, fov_y },
            lod_metric_multiplier: 1.0,
            ellipsoid: None,
        }
    }

    /// Set the reference ellipsoid for geodetic computations.
    pub fn with_ellipsoid(mut self, ellipsoid: terra::Ellipsoid) -> Self {
        self.ellipsoid = Some(ellipsoid);
        self
    }

    /// Convert the ECEF position to cartographic using the stored ellipsoid.
    pub fn position_cartographic(&self) -> Option<terra::Cartographic> {
        self.ellipsoid.as_ref()?.ecef_to_cartographic(self.position)
    }

    /// Create an orthographic view state.
    pub fn orthographic(
        position: Vec3,
        direction: Vec3,
        up: Vec3,
        viewport_px: [u32; 2],
        half_width: f64,
        half_height: f64,
    ) -> Self {
        Self {
            viewport_px,
            position,
            direction,
            up,
            projection: Projection::Orthographic {
                half_width,
                half_height,
            },
            lod_metric_multiplier: 1.0,
            ellipsoid: None,
        }
    }

    /// Horizontal field-of-view in radians, or `None` for orthographic views.
    pub fn fov_x(&self) -> Option<f64> {
        match &self.projection {
            Projection::Perspective { fov_x, .. } => Some(*fov_x),
            Projection::Orthographic { .. } => None,
        }
    }

    /// Vertical field-of-view in radians, or `None` for orthographic views.
    pub fn fov_y(&self) -> Option<f64> {
        match &self.projection {
            Projection::Perspective { fov_y, .. } => Some(*fov_y),
            Projection::Orthographic { .. } => None,
        }
    }

    /// Construct a `ViewState` from a column-major view matrix and a projection matrix.
    ///
    /// Extracts camera world position, direction, and up vector from the inverse of
    /// `view_matrix`, and derives the projection from the projection matrix coefficients.
    ///
    /// Assumes a standard OpenGL/Vulkan column-major convention:
    /// - `view_matrix` transforms world -> camera space
    /// - `proj_matrix` is a perspective or orthographic projection matrix
    ///
    /// `viewport_px` is `[width, height]` in physical pixels.
    pub fn from_matrices(
        view_matrix: glam::DMat4,
        proj_matrix: glam::DMat4,
        viewport_px: [u32; 2],
    ) -> Self {
        // Extract camera-to-world transformation (inverse of view matrix).
        debug_assert!(
            view_matrix.determinant().abs() > 1e-10,
            "view_matrix is singular or near-singular; inverse is undefined"
        );
        let cam_to_world = view_matrix.inverse();
        let position = cam_to_world.col(3).truncate();
        // Camera looks down −Z in camera space; transform to world space.
        let direction = -(cam_to_world.col(2).truncate()).normalize();
        let up = cam_to_world.col(1).truncate().normalize();

        // Detect perspective vs orthographic from the [3][3] element.
        // Perspective: proj[3][3] == 0; Orthographic: proj[3][3] == 1.
        let projection = if proj_matrix.col(3).w.abs() < 0.5 {
            // Perspective: fov_y from proj[1][1] = 1/tan(fov_y/2)
            let fov_y = 2.0 * (1.0 / proj_matrix.col(1).y).atan();
            let aspect = proj_matrix.col(1).y / proj_matrix.col(0).x;
            let fov_x = 2.0 * (aspect / proj_matrix.col(1).y).atan();
            Projection::Perspective { fov_x, fov_y }
        } else {
            // Orthographic: half extents from proj[0][0] and proj[1][1].
            // proj[0][0] = 2 / (right - left) ~ 2 / (2 * half_width)
            let half_width = 1.0 / proj_matrix.col(0).x;
            let half_height = 1.0 / proj_matrix.col(1).y;
            Projection::Orthographic {
                half_width,
                half_height,
            }
        };

        Self {
            viewport_px,
            position,
            direction,
            up,
            projection,
            lod_metric_multiplier: 1.0,
            ellipsoid: None,
        }
    }
}

/// Holds the per-view-group traversal state for one tileset.
///
/// Mirrors `Cesium3DTilesSelection::TilesetViewGroup`.
pub struct ViewGroup {
    /// Reusable working buffers for the selection traversal.
    pub(crate) buffers: TraversalBuffers,
    /// Load requests produced by the last traversal; consumed by `load_tiles`.
    pub(crate) load_queue: Vec<LoadRequest>,
    /// Public result of the last traversal.
    last_result: ViewUpdateResult,
    /// Controls the alpha curves for fade-in / fade-out LOD transitions.
    fade_strategy: Arc<dyn FadeStrategy>,
}

impl ViewGroup {
    /// Creates a new, empty `ViewGroup` with the default [`LinearFadeStrategy`].
    pub fn new() -> Self {
        Self {
            buffers: TraversalBuffers::new(),
            load_queue: Vec::new(),
            last_result: ViewUpdateResult::default(),
            fade_strategy: Arc::new(LinearFadeStrategy),
        }
    }

    /// Creates a `ViewGroup` with a custom [`FadeStrategy`].
    ///
    /// Use this to substitute ease-in-out curves, gamma-corrected blending,
    /// or any other per-frame alpha mapping for LOD transitions.
    pub fn with_fade_strategy(fade_strategy: Arc<dyn FadeStrategy>) -> Self {
        Self {
            buffers: TraversalBuffers::new(),
            load_queue: Vec::new(),
            last_result: ViewUpdateResult::default(),
            fade_strategy,
        }
    }

    /// Returns the result of the most recent [`Tileset::update_view_group`]
    /// call for this view group.
    ///
    /// Mirrors `TilesetViewGroup::getViewUpdateResult()`.
    pub fn view_update_result(&self) -> &ViewUpdateResult {
        &self.last_result
    }

    /// Returns the active [`FadeStrategy`] for alpha-curve computation.
    ///
    /// [`Layer`] reads this in step 10 to convert raw fade-progress values
    /// into per-tile alphas.  Custom renderers can also call this to apply the
    /// same curve when building their own render commands.
    ///
    /// [`Layer`]: crate::Layer
    pub fn fade_strategy(&self) -> &dyn FadeStrategy {
        self.fade_strategy.as_ref()
    }

    /// Merge a traversal [`SelectionOutput`] into the persistent
    /// [`ViewUpdateResult`], advancing LOD fade percentages.
    ///
    /// This is the previously-inline fading merge loop in
    /// `ContentManager::update_view_group` (steps 5 + fade bookkeeping),
    /// extracted so the Tileset orchestrator can call it explicitly.
    ///
    /// After this call `view_update_result()` reflects the new frame.
    pub(crate) fn commit_result(
        &mut self,
        output: SelectionOutput,
        frame_number: u64,
        delta_time: f32,
        streaming: &StreamingOptions,
        states: &mut TileStates,
        memory_adjusted_sse: f64,
    ) {
        let transition_length = streaming.lod_transition_length.max(f32::EPSILON);
        let delta_pct = if streaming.enable_lod_transition {
            delta_time / transition_length
        } else {
            f32::MAX
        };

        let prev = &mut self.last_result;

        let render_set: HashSet<TileId> = output.selected.iter().copied().collect();

        // Remove tiles that re-entered the render set from fading_out.
        // Instead of resetting the fade-in pct to 0 (which causes a hard jump
        // to alpha=0), start the fade-in from the complement of the current
        // fade-out pct so the visual alpha is continuous.
        prev.tiles_fading_out.retain(|n| {
            if render_set.contains(n) {
                let fade_out_pct = prev.tile_fade_percentages.get(n).copied().unwrap_or(0.0);
                prev.tile_fade_percentages.insert(*n, 1.0 - fade_out_pct);
                false
            } else {
                true
            }
        });

        // Insert newly-fading-out tiles. Instead of always starting at 0
        // (which causes a hard jump to alpha=1 for tiles that were still
        // fading in), use the complement of the current fade-in pct so the
        // alpha stays continuous.
        for &tile in &output.fading_out {
            if !prev.tiles_fading_out.contains(&tile) {
                prev.tiles_fading_out.push(tile);
                // Remove the fade-in entry (if any) and derive fade-out start.
                let fade_in_pct = prev.tile_fade_percentages.remove(&tile).unwrap_or(1.0);
                prev.tile_fade_percentages.insert(tile, 1.0 - fade_in_pct);
            }
        }

        // Advance fade-out percentages; remove completed.
        prev.tiles_fading_out.retain(|n| {
            let pct = prev.tile_fade_percentages.entry(*n).or_insert(0.0);
            *pct = (*pct + delta_pct).min(1.0);
            if *pct >= 1.0 {
                prev.tile_fade_percentages.remove(n);
                false
            } else {
                true
            }
        });

        // Advance fade-in percentages for render tiles.
        let fading_in_set: HashSet<TileId> = output.fading_in.iter().copied().collect();
        for &tile in &output.selected {
            let already_in_map = prev.tile_fade_percentages.contains_key(&tile);
            if already_in_map || fading_in_set.contains(&tile) {
                let pct = prev.tile_fade_percentages.entry(tile).or_insert(0.0);
                *pct = (*pct + delta_pct).min(1.0);
                states.get_mut(tile).lod_fade_pct = *pct;
            }
        }

        // Remove entries that completed fade-in.
        prev.tile_fade_percentages.retain(|n, pct| {
            (render_set.contains(n) && *pct < 1.0) || prev.tiles_fading_out.contains(n)
        });

        // Write non-persistent fields.
        prev.selected_tiles = output.selected;
        prev.tile_screen_space_errors = output.selected_sse;
        prev.tile_selection_depths = output.selection_depths;
        prev.tile_final_resolutions = output.selection_final_resolutions;
        prev.has_mixed_selection = output.has_mixed_selection;
        prev.tiles_fading_in = output.fading_in;
        prev.tiles_visited = output.nodes_visited as u32;
        prev.tiles_culled = output.nodes_culled as u32;
        prev.tiles_culled_but_visited = output.nodes_culled_but_visited as u32;
        prev.tiles_kicked = output.nodes_kicked as u32;
        prev.max_depth_visited = output.max_depth;
        prev.frame_number = frame_number;

        // Content statistics derived from tile state machine.
        // Mirrors CesiumJS `Cesium3DTilesetStatistics`.
        let (tiles_loading, tiles_renderable, tiles_failed, resident_bytes) = states.load_stats();
        prev.tiles_loading = tiles_loading;
        prev.tiles_ready = tiles_renderable;
        prev.tiles_failed = tiles_failed;
        prev.resident_bytes = resident_bytes;
        prev.memory_adjusted_screen_space_error = memory_adjusted_sse;

        // Also stash the load queue.
        self.load_queue = output.load;
    }
}

impl Default for ViewGroup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_decision::SelectionOutput;
    use crate::options::StreamingOptions;
    use crate::selection_state::TileStates;
    use crate::tile_store::TileId;

    fn tile(slot: u32) -> TileId {
        TileId::from_slot(slot)
    }

    fn streaming_no_fade() -> StreamingOptions {
        StreamingOptions {
            enable_lod_transition: false,
            ..StreamingOptions::default()
        }
    }

    fn streaming_with_fade(length: f32) -> StreamingOptions {
        StreamingOptions {
            enable_lod_transition: true,
            lod_transition_length: length,
            ..StreamingOptions::default()
        }
    }

    #[test]
    fn new_view_group_has_empty_result() {
        let vg = ViewGroup::new();
        let r = vg.view_update_result();
        assert!(r.selected_tiles.is_empty());
        assert!(r.tiles_fading_out.is_empty());
        assert!(r.tile_fade_percentages.is_empty());
    }

    #[test]
    fn commit_sets_render_list() {
        let mut vg = ViewGroup::new();
        let mut states = TileStates::new();
        let output = SelectionOutput {
            selected: vec![tile(0), tile(1)],
            selected_sse: vec![1.0, 2.0],
            ..SelectionOutput::default()
        };
        vg.commit_result(output, 1, 0.016, &streaming_no_fade(), &mut states, 16.0);
        let r = vg.view_update_result();
        assert_eq!(r.selected_tiles, vec![tile(0), tile(1)]);
        assert_eq!(r.tile_screen_space_errors, vec![1.0, 2.0]);
    }

    #[test]
    fn fading_out_advances_and_removes_at_complete() {
        let mut vg = ViewGroup::new();
        let mut states = TileStates::new();
        let streaming = streaming_with_fade(1.0); // 1 second transition

        // Frame 1: tile 0 leaves the render set -> enters fading_out.
        let out1 = SelectionOutput {
            selected: vec![],
            fading_out: vec![tile(0)],
            ..SelectionOutput::default()
        };
        vg.commit_result(out1, 1, 0.5, &streaming, &mut states, 16.0);
        let r = vg.view_update_result();
        assert!(r.tiles_fading_out.contains(&tile(0)));
        // After 0.5 s at 1.0 s transition: pct = 0.5
        let pct = r.tile_fade_percentages[&tile(0)];
        assert!((pct - 0.5).abs() < 1e-4);

        // Frame 2: another 0.6 s -> total 1.1 s -> pct clamped to 1.0 and tile removed.
        let out2 = SelectionOutput::default();
        vg.commit_result(out2, 2, 0.6, &streaming, &mut states, 16.0);
        let r2 = vg.view_update_result();
        assert!(!r2.tiles_fading_out.contains(&tile(0)));
        assert!(!r2.tile_fade_percentages.contains_key(&tile(0)));
    }

    #[test]
    fn lod_transition_disabled_completes_instantly() {
        let mut vg = ViewGroup::new();
        let mut states = TileStates::new();

        let out = SelectionOutput {
            selected: vec![],
            fading_out: vec![tile(0)],
            ..SelectionOutput::default()
        };
        // With transitions disabled delta_pct = MAX so the tile should disappear in one frame.
        vg.commit_result(out, 1, 0.016, &streaming_no_fade(), &mut states, 16.0);
        let r = vg.view_update_result();
        assert!(!r.tiles_fading_out.contains(&tile(0)));
    }

    #[test]
    fn reenter_render_gives_continuous_alpha() {
        let mut vg = ViewGroup::new();
        let mut states = TileStates::new();
        let streaming = streaming_with_fade(1.0);

        // Frame 1: tile leaves render -> starts fading out.
        let out1 = SelectionOutput {
            selected: vec![],
            fading_out: vec![tile(0)],
            ..SelectionOutput::default()
        };
        vg.commit_result(out1, 1, 0.4, &streaming, &mut states, 16.0);
        // fade_out pct = 0.4

        // Frame 2: tile re-enters render before fade completes.
        let out2 = SelectionOutput {
            selected: vec![tile(0)],
            selected_sse: vec![1.0],
            ..SelectionOutput::default()
        };
        vg.commit_result(out2, 2, 0.0, &streaming, &mut states, 16.0);
        let r = vg.view_update_result();

        // Tile must be back in the render set and no longer fading_out.
        assert!(r.selected_tiles.contains(&tile(0)));
        assert!(!r.tiles_fading_out.contains(&tile(0)));
        // Fade-in starts at 1.0 - 0.4 = 0.6 (complement of fade-out pct).
        let pct = r
            .tile_fade_percentages
            .get(&tile(0))
            .copied()
            .unwrap_or(1.0);
        assert!((pct - 0.6).abs() < 1e-4);
    }
}

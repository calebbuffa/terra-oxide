//! [`RenderNode`] - per-tile render data returned by
//! [`ContentManager::render_nodes`](crate::ContentManager::render_nodes).
//!
//! Content lookup is deliberately absent - query your own store with `tile`
//! as the key. kiban owns selection state; you own renderer resources.

use glam::DMat4;

use crate::tile_store::TileId;

/// One tile in the render set for the current frame.
///
/// Content lookup is left to the caller - query your own content store
/// with [`tile`](RenderNode::tile) as the key.
pub struct SelectedTile {
    pub tile: TileId,
    /// World-space transform (column-major).
    pub transform: DMat4,
    /// Opacity for LOD transitions: `1.0` = fully visible.
    pub alpha: f32,
    /// Skip-LOD stencil depth (`0` = normal rendering).
    pub selection_depth: u32,
    /// `true` for leaf/foreground tiles; `false` for background ancestor tiles
    /// rendered simultaneously with higher-detail descendants (skip-LOD only).
    ///
    /// Renderers should use this to implement the bivariate stencil pass that
    /// prevents z-fighting when `has_mixed_selection` is true:
    /// tiles with `final_resolution = false` are background ancestors whose
    /// frontfaces should be masked by children already drawn at that pixel.
    ///
    /// Mirrors CesiumJS `tile._finalResolution`.
    pub final_resolution: bool,
}

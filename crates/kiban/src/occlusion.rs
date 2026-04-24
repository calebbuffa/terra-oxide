//! Renderer-driven per-tile occlusion interface.
//!
//! Implement [`TileOcclusionProxy`] and set it on [`LayerOptions`] to enable
//! renderer-driven occlusion culling of tile refinement.

/// The occlusion state of a tile as determined by the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TileOcclusionState {
    /// The renderer has not yet determined occlusion for this tile.
    NotReady,
    /// The tile is definitely occluded (behind other geometry).
    Occluded,
    /// The tile is definitely not occluded.
    NotOccluded,
}

/// Interface for the renderer to report per-tile occlusion each frame.
///
/// Implement this trait and pass it to [`LayerOptions`] to enable
/// renderer-driven occlusion culling.
///
/// The traversal queries `tile_occlusion()` for any tile it is considering
/// refining when `SelectionOptions::culling::delay_refinement_for_occlusion`
/// is set to `true`.
///
/// [`LayerOptions`]: crate::LayerOptions
pub trait TileOcclusionProxy: Send + Sync {
    /// Query occlusion state for a tile, identified by its numeric id.
    ///
    /// The `tile_id` value is `TileId(slot + 1)` cast to `u64`.  It is stable
    /// within a session but may change if the tile store is rebuilt.
    ///
    /// Called once per tile per frame. The implementation may return
    /// [`TileOcclusionState::NotReady`] if the occlusion query result
    /// is not yet available (e.g., a GPU query has not yet returned).
    fn tile_occlusion(&self, tile_id: u64) -> TileOcclusionState;
}

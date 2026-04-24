//! Lifecycle events emitted by the overlay engine.

use crate::overlay::{OverlayId, RasterOverlayTile};

/// A lifecycle event emitted by [`OverlayEngine`](crate::OverlayEngine)
/// during a frame update.
///
/// Consumers drain events after each [`update`](crate::OverlayEngine::update)
/// call to react to overlay state changes (e.g. uploading a texture to the GPU
/// when an overlay tile arrives, or releasing it when detached).
///
/// Marked `#[non_exhaustive]` - new variants may be added without breaking.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum OverlayEvent<T> {
    /// An overlay tile is now ready and has been attached to a geometry tile.
    Attached {
        tile_id: T,
        overlay_id: OverlayId,
        uv_index: u32,
        tile: RasterOverlayTile,
    },
    /// An overlay was detached from a tile (tile disappeared or overlay removed).
    Detached { tile_id: T, overlay_id: OverlayId },
}

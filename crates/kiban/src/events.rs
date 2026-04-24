//! Typed argument structs for kiban's public events.
//!
//! Each event on [`Layer`](crate::Layer) carries a dedicated arg struct rather
//! than a variant of a shared enum. Structs are [`Clone`] so they fan-out to
//! multiple listeners without extra allocation (pixel data is `Arc`-backed).

use std::sync::Arc;

use glam::DMat4;

use sovra::{OverlayId, RasterOverlayTile};

use crate::tile_store::TileId;

/// Arguments for [`Layer::tile_loaded`](crate::Layer::tile_loaded).
///
/// A tile's geometry has finished loading and is ready to be prepared.
/// Pass `model` to a background thread for encoding (GLB
/// serialisation, collision mesh, etc.), then call
/// [`LayerHandle::mark_tile_ready`](crate::LayerHandle::mark_tile_ready) once
/// your GPU resources are ready.
#[derive(Clone)]
pub struct TileLoadedArgs {
    pub tile: TileId,
    /// Loaded glTF model. `Arc`-wrapped so multiple listeners can share it
    /// without cloning the underlying geometry buffers.
    pub model: Arc<moderu::GltfModel>,
    /// World-space transform at load time.
    ///
    /// Use `layer.selected_tiles()` for the authoritative per-frame transform;
    /// this value is correct at load time but may drift if the tileset's root
    /// transform changes.
    pub transform: DMat4,
}

/// Arguments for [`Layer::custom_tile_loaded`](crate::Layer::custom_tile_loaded).
///
/// A [`ContentLoader`](crate::ContentLoader) returned
/// [`TileContentKind::Custom`](crate::TileContentKind::Custom). Downcast
/// `content` to your concrete type, prepare it, then call
/// [`LayerHandle::mark_tile_ready`](crate::LayerHandle::mark_tile_ready).
#[derive(Clone)]
pub struct CustomTileLoadedArgs {
    pub tile: TileId,
    /// Opaque content payload from the loader.
    pub content: Arc<dyn std::any::Any + Send + Sync>,
    /// World-space transform at load time.
    pub transform: DMat4,
}

/// Arguments for [`Layer::overlay_ready`](crate::Layer::overlay_ready).
///
/// A raster overlay tile is ready to be uploaded and attached to the geometry
/// tile. Only fired after kiban has received the readiness ack for `tile`
/// (via [`LayerHandle::mark_tile_ready`](crate::LayerHandle::mark_tile_ready)),
/// so the tile's GPU geometry is guaranteed to exist.
///
/// Compute overlay UVs via `sovra::compute_overlay_uvs_from_positions` using
/// the ECEF positions extracted from the model during the
/// [`tile_loaded`](crate::Layer::tile_loaded) handler.
#[derive(Clone)]
pub struct OverlayReadyArgs {
    pub tile: TileId,
    pub overlay_id: OverlayId,
    /// UV channel assigned by the overlay engine.
    pub uv_index: u32,
    /// Overlay pixel data and its geographic extent.
    ///
    /// `RasterOverlayTile` holds pixel data behind an `Arc<[u8]>`, so cloning
    /// is cheap.
    pub overlay: RasterOverlayTile,
}

/// Arguments for [`Layer::tile_failed`](crate::Layer::tile_failed).
#[derive(Clone)]
pub struct TileFailedArgs {
    pub tile: TileId,
    /// Source URL of the failed tile, if known.
    /// Wrapped in Arc to avoid cloning the string for each event listener.
    pub url: Option<Arc<str>>,
    /// Human-readable failure reason.
    /// Wrapped in Arc to avoid cloning the string for each event listener.
    pub message: Arc<str>,
}

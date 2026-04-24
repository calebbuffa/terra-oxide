//! Minimal spatial hierarchy trait for overlay resolution.
//!
//! [`OverlayHierarchy`] describes just enough structure for the overlay engine
//! to walk parent chains and compute geographic extents.  It is intentionally
//! decoupled from any tile selection engine - the composition layer (e.g.
//! `kiban`) bridges a full `SceneGraph` into this trait.

/// Read-only spatial hierarchy used by the overlay engine.
///
/// Generic over the caller's opaque tile identifier `T` so consumers don't
/// need to smuggle their IDs through `u64`. `T` is typically an enum, a
/// newtype, or a `NonZeroU32`-backed handle from the tile selection engine.
///
/// Only two capabilities are required:
/// - Walk up the tree via [`parent`](Self::parent).
/// - Obtain the geographic extent of a tile via [`globe_rectangle`](Self::globe_rectangle).
pub trait OverlayHierarchy<T>: Send + Sync {
    /// Returns the parent of `tile`, or `None` if it is a root.
    fn parent(&self, tile: T) -> Option<T>;

    /// Geographic extent of this tile in geodetic longitude/latitude (radians).
    fn globe_rectangle(&self, tile: T) -> Option<terra::GlobeRectangle>;
}

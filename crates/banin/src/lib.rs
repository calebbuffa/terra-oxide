//! `banin` — Cesium terrain tile builder.
//!
//! Converts GeoTIFF Digital Elevation Model (DEM) rasters to
//! [`heightmap-1.0`](https://github.com/CesiumGS/quantized-mesh) and
//! `quantized-mesh-1.0` tilesets for use with `CesiumTerrainProvider`.
//!
//! Pure Rust — no GDAL or other C dependencies required.
//!
//! # Example
//!
//! ```
//! use banin::grid::GlobalGeodetic;
//!
//! let grid = GlobalGeodetic::default();
//! let bounds = grid.tile_bounds(0, 0, 0);
//! assert!((bounds.west.to_degrees() - (-180.0)).abs() < 1e-9);
//! ```

pub mod grid;
pub mod tiler;

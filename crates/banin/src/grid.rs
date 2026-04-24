//! TMS Global Geodetic grid (EPSG:4326).
//!
//! A direct port of CTB's `GlobalGeodetic` / `Grid` classes, implemented as
//! pure Rust without GDAL.
//!
//! ## Coordinate conventions
//!
//! * `x` increases west -> east.
//! * `y` increases south -> north (TMS convention, same as CTB / Cesium).
//! * Zoom 0 has **two** root tiles: `(0,0)` covers the western hemisphere and
//!   `(1,0)` covers the eastern hemisphere.
//! * Bounds are returned in **radians** as [`terra::GlobeRectangle`].
//!
//! ## Example
//!
//! ```
//! use banin::grid::GlobalGeodetic;
//! use std::f64::consts::{PI, FRAC_PI_2};
//!
//! let grid = GlobalGeodetic::default();
//!
//! // At zoom 0 the western root tile spans [−\pi, −\pi/2] -> [0, \pi/2] (radians).
//! let w = grid.tile_bounds(0, 0, 0);
//! assert!((w.west  - -PI).abs()       < 1e-9);
//! assert!((w.south - -FRAC_PI_2).abs() < 1e-9);
//! assert!((w.east  -  0.0).abs()      < 1e-9);
//! assert!((w.north -  FRAC_PI_2).abs() < 1e-9);
//!
//! // Resolution at zoom 0 with tile_size=65 (degrees/pixel).
//! let res = grid.resolution(0);
//! assert!((res - 180.0 / 65.0).abs() < 1e-9);
//! ```

use terra::GlobeRectangle;
use zukei::{QuadtreeTileID, QuadtreeTilingScheme};

/// TMS Global Geodetic profile (two root tiles at zoom 0).
///
/// This matches the CTB / Cesium terrain tiling grid exactly.
#[derive(Debug, Clone, Copy)]
pub struct GlobalGeodetic {
    /// Tile width/height in pixels (default: 65, including the 1-pixel border).
    pub tile_size: u32,
}

impl Default for GlobalGeodetic {
    fn default() -> Self {
        Self { tile_size: 65 }
    }
}

impl GlobalGeodetic {
    pub fn new(tile_size: u32) -> Self {
        Self { tile_size }
    }

    /// Degrees per pixel at `zoom` level.
    ///
    /// At zoom 0 each root tile spans 180° in longitude; the resolution is
    /// therefore `180 / tile_size`.
    #[inline]
    pub fn resolution(&self, zoom: u32) -> f64 {
        180.0 / (self.tile_size as f64 * (1u64 << zoom) as f64)
    }

    /// Geographic bounds of tile `(x, y)` at `zoom`, returned in **radians**.
    ///
    /// Delegates to [`zukei::QuadtreeTilingScheme::geographic`] for the coordinate
    /// maths, then wraps the result as a [`terra::GlobeRectangle`].
    pub fn tile_bounds(&self, zoom: u32, x: u32, y: u32) -> GlobeRectangle {
        let r = QuadtreeTilingScheme::geographic()
            .tile_to_rectangle(QuadtreeTileID::new(zoom, x, y))
            .expect("tile coordinates out of range");
        GlobeRectangle {
            west: r.minimum_x,
            south: r.minimum_y,
            east: r.maximum_x,
            north: r.maximum_y,
        }
    }

    /// The zoom level whose resolution most closely matches `res` (degrees/px).
    ///
    /// Rounds up (smaller zoom = lower resolution, so we choose the finer one).
    pub fn zoom_for_resolution(&self, res: f64) -> u32 {
        let z0_res = 180.0 / self.tile_size as f64;
        (z0_res / res).log2().ceil().max(0.0) as u32
    }

    /// Number of tiles in the x direction at `zoom`.
    #[inline]
    pub fn tiles_x(&self, zoom: u32) -> u32 {
        QuadtreeTilingScheme::geographic().tiles_x_at_level(zoom)
    }

    /// Number of tiles in the y direction at `zoom`.
    #[inline]
    pub fn tiles_y(&self, zoom: u32) -> u32 {
        QuadtreeTilingScheme::geographic().tiles_y_at_level(zoom)
    }

    /// Tile coordinate containing `(lon_deg, lat_deg)` at `zoom`.
    ///
    /// Returns a [`QuadtreeTileID`] clamped to the valid tile range.
    pub fn tile_for_point(&self, zoom: u32, lon_deg: f64, lat_deg: f64) -> QuadtreeTileID {
        let nx = self.tiles_x(zoom);
        let ny = self.tiles_y(zoom);
        let x = ((lon_deg + 180.0) / 360.0 * nx as f64)
            .max(0.0)
            .min((nx - 1) as f64) as u32;
        let y = ((lat_deg + 90.0) / 180.0 * ny as f64)
            .max(0.0)
            .min((ny - 1) as f64) as u32;
        QuadtreeTileID::new(zoom, x, y)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use super::*;

    fn grid() -> GlobalGeodetic {
        GlobalGeodetic::default() // tile_size = 65
    }

    #[test]
    fn resolution_zoom0() {
        let g = grid();
        assert!((g.resolution(0) - 180.0 / 65.0).abs() < 1e-10);
    }

    #[test]
    fn resolution_doubles_each_zoom() {
        let g = grid();
        let r0 = g.resolution(0);
        let r1 = g.resolution(1);
        assert!((r0 / r1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn tile_bounds_zoom0_west() {
        let g = grid();
        let b = g.tile_bounds(0, 0, 0);
        assert!((b.west - -PI).abs() < 1e-9);
        assert!((b.south - -FRAC_PI_2).abs() < 1e-9);
        assert!((b.east - 0.0).abs() < 1e-9);
        assert!((b.north - FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn tile_bounds_zoom0_east() {
        let g = grid();
        let b = g.tile_bounds(0, 1, 0);
        assert!((b.west - 0.0).abs() < 1e-9);
        assert!((b.east - PI).abs() < 1e-9);
    }

    #[test]
    fn tile_bounds_adjacent_no_gap() {
        let g = grid();
        let left = g.tile_bounds(1, 1, 0);
        let right = g.tile_bounds(1, 2, 0);
        assert!((left.east - right.west).abs() < 1e-10);
    }

    #[test]
    fn zoom_for_resolution_roundtrip() {
        let g = grid();
        for zoom in 0u32..=8 {
            let res = g.resolution(zoom);
            assert_eq!(g.zoom_for_resolution(res), zoom);
        }
    }

    #[test]
    fn tile_for_point_origin() {
        let g = grid();
        // (0°, 0°) is on the boundary between the two root tiles
        let id = g.tile_for_point(0, 0.0, 0.0);
        // 0° longitude -> exactly at tile boundary; both 0 and 1 are valid
        assert!(id.x <= 1);
    }

    #[test]
    fn tile_counts() {
        let g = grid();
        assert_eq!(g.tiles_x(0), 2);
        assert_eq!(g.tiles_y(0), 1);
        assert_eq!(g.tiles_x(1), 4);
        assert_eq!(g.tiles_y(1), 2);
    }
}

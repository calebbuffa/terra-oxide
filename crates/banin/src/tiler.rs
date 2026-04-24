//! Pure-Rust terrain tiler backed by oxigdal (no C / GDAL dependency).
//!
//! [`TerrainTiler`] opens a GeoTIFF with [`oxigdal_geotiff::CogReader`], reads
//! the full raster into memory once, then iterates over `(zoom, x, y)` tile
//! coordinates.  For each tile it bilinearly resamples the source data into a
//! `grid_size × grid_size` elevation grid, warping from the source CRS to
//! EPSG:4326 using [`oxigdal_proj::Transformer`].

use std::ops::RangeInclusive;
use std::path::Path;

use oxigdal_core::io::FileDataSource;
use oxigdal_core::types::RasterDataType;
use oxigdal_geotiff::CogReader;
use oxigdal_geotiff::tiff::ByteOrderType;
use oxigdal_proj::{Coordinate, Crs, Transformer};
use terra::GlobeRectangle;
use thiserror::Error;
use zukei::QuadtreeTileID;

use crate::grid::GlobalGeodetic;

/// Default tile grid size — 65×65 matches CTB and Cesium defaults.
pub const DEFAULT_GRID_SIZE: usize = 65;

#[derive(Debug, Error)]
pub enum TilerError {
    #[error("cannot open raster: {0}")]
    Open(#[from] oxigdal_core::error::OxiGdalError),

    #[error("raster has no geo-transform; cannot determine spatial extent")]
    NoGeoTransform,

    #[error("raster has no CRS (EPSG code); cannot reproject to WGS84")]
    NoCrs,

    #[error("unsupported raster data type: {0:?}")]
    UnsupportedDataType(RasterDataType),

    #[error("CRS error: {0}")]
    Crs(String),
}

/// Elevation data for one tile.
///
/// Pass `heights` and bounds to [`arazi::encode_quantized_mesh`], or convert
/// heights with [`elevation_to_u16`] and wrap in [`arazi::HeightmapTile`] for
/// [`arazi::encode_heightmap`].
pub struct TileData {
    /// Tile coordinate in the TMS quadtree.
    pub id: QuadtreeTileID,
    /// Elevation samples in **metres**, length = `grid_size²`,
    /// row-major **south-to-north**, west-to-east.
    pub heights: Vec<f64>,
    pub grid_size: usize,
    /// Tile bounds in **radians** (EPSG:4326).
    pub bounds: GlobeRectangle,
}

/// Converts a GeoTIFF DEM into terrain tiles at multiple zoom levels.
pub struct TerrainTiler {
    /// Full raster data, row-major top-to-bottom (north first) in source pixel space.
    data: Vec<f64>,
    width: usize,
    height: usize,
    nodata: Option<f64>,
    /// Affine geotransform [origin_x, px_w, row_rot, origin_y, col_rot, px_h].
    geotransform: [f64; 6],
    /// Inverse transformer: WGS84 (EPSG:4326) → source CRS.
    inv_transformer: Option<Transformer>,
    grid: GlobalGeodetic,
    max_zoom: u32,
    grid_size: usize,
    /// Dataset extent in WGS84 degrees `(west, south, east, north)`.
    bounds_deg: (f64, f64, f64, f64),
}

impl TerrainTiler {
    /// Open a GeoTIFF at `path` using [`DEFAULT_GRID_SIZE`].
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, TilerError> {
        Self::open_with_grid_size(path, DEFAULT_GRID_SIZE)
    }

    /// Open a GeoTIFF at `path` with a custom grid size.
    pub fn open_with_grid_size<P: AsRef<Path>>(
        path: P,
        grid_size: usize,
    ) -> Result<Self, TilerError> {
        let source = FileDataSource::open(path)?;
        let reader = CogReader::open(source)?;

        let gt_struct = reader.geo_transform()?.ok_or(TilerError::NoGeoTransform)?;
        let geotransform = [
            gt_struct.origin_x,
            gt_struct.pixel_width,
            gt_struct.row_rotation,
            gt_struct.origin_y,
            gt_struct.col_rotation,
            gt_struct.pixel_height,
        ];

        let epsg = reader.epsg_code().ok_or(TilerError::NoCrs)?;
        let nodata = reader.nodata()?.as_f64();

        let width = reader.width() as usize;
        let height = reader.height() as usize;

        let data_type = reader
            .primary_info()
            .data_type()
            .unwrap_or(RasterDataType::Float32);
        let byte_order = reader.tiff().byte_order();

        let data = read_full_raster(&reader, width, height, data_type, byte_order, nodata)?;

        // Build WGS84 extent by transforming the four corners.
        let (bounds_deg, inv_transformer) = if epsg == 4326 {
            let west = geotransform[0];
            let north = geotransform[3];
            let east = west + geotransform[1] * width as f64;
            let south = north + geotransform[5] * height as f64;
            (
                (
                    west.min(east),
                    south.min(north),
                    west.max(east),
                    south.max(north),
                ),
                None,
            )
        } else {
            let src_crs = Crs::from_epsg(epsg).map_err(|e| TilerError::Crs(e.to_string()))?;
            let wgs84 = Crs::wgs84();
            let fwd = Transformer::new(src_crs.clone(), wgs84.clone())
                .map_err(|e| TilerError::Crs(e.to_string()))?;
            let inv =
                Transformer::new(wgs84, src_crs).map_err(|e| TilerError::Crs(e.to_string()))?;

            let corners_src = [
                geo_from_pixel(0.0, 0.0, &geotransform),
                geo_from_pixel(width as f64, 0.0, &geotransform),
                geo_from_pixel(0.0, height as f64, &geotransform),
                geo_from_pixel(width as f64, height as f64, &geotransform),
            ];
            let mut lons = [0.0f64; 4];
            let mut lats = [0.0f64; 4];
            for (i, (x, y)) in corners_src.iter().enumerate() {
                let c = fwd
                    .transform(&Coordinate::new(*x, *y))
                    .map_err(|e| TilerError::Crs(e.to_string()))?;
                lons[i] = c.x;
                lats[i] = c.y;
            }
            let west = lons.iter().cloned().fold(f64::INFINITY, f64::min);
            let east = lons.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let south = lats.iter().cloned().fold(f64::INFINITY, f64::min);
            let north = lats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            ((west, south, east, north), Some(inv))
        };

        let native_res = geotransform[1].abs();
        let grid = GlobalGeodetic::new(grid_size as u32);
        let max_zoom = grid.zoom_for_resolution(native_res);

        Ok(Self {
            data,
            width,
            height,
            nodata,
            geotransform,
            inv_transformer,
            grid,
            max_zoom,
            grid_size,
            bounds_deg,
        })
    }

    /// Maximum zoom level inferred from the native raster resolution.
    pub fn max_zoom(&self) -> u32 {
        self.max_zoom
    }

    /// Dataset extent in WGS84 degrees as `(west, south, east, north)`.
    pub fn bounds_deg(&self) -> (f64, f64, f64, f64) {
        self.bounds_deg
    }

    /// Iterate all tiles across `zooms` that overlap the source raster.
    ///
    /// Tiles are yielded in order: zoom → y → x.
    pub fn tiles(
        &self,
        zooms: RangeInclusive<u32>,
    ) -> impl Iterator<Item = Result<TileData, TilerError>> + '_ {
        zooms.flat_map(move |zoom| {
            let (west, south, east, north) = self.bounds_deg;
            let min = self.grid.tile_for_point(zoom, west, south);
            let max = self.grid.tile_for_point(zoom, east - 1e-10, north - 1e-10);
            (min.y..=max.y)
                .flat_map(move |y| (min.x..=max.x).map(move |x| self.create_tile(zoom, x, y)))
        })
    }

    fn create_tile(&self, zoom: u32, x: u32, y: u32) -> Result<TileData, TilerError> {
        let bounds = self.grid.tile_bounds(zoom, x, y);
        // Sample directly at the tile boundary coordinates (no pixel-center offset,
        // no expansion): the quantized-mesh spec requires u=0/v=0 to be exactly at
        // the west/south edge and u=32767/v=32767 at the east/north edge.  Adjacent
        // tiles therefore sample the same geographic positions at shared edges.
        let heights = self.sample_window(
            bounds.west.to_degrees(),
            bounds.south.to_degrees(),
            bounds.east.to_degrees(),
            bounds.north.to_degrees(),
        )?;
        Ok(TileData {
            id: QuadtreeTileID::new(zoom, x, y),
            heights,
            grid_size: self.grid_size,
            bounds,
        })
    }

    /// Sample a WGS84 window `[west, south, east, north]` (degrees) into an
    /// `n × n` elevation grid (south-to-north, west-to-east).
    ///
    /// Vertices are placed at exact grid positions (not pixel centers):
    /// col=0 → west, col=n-1 → east, row=0 → south, row=n-1 → north.
    /// This ensures adjacent tiles sample the same geographic positions at
    /// shared edges, and that edge vertices sit exactly on tile boundaries.
    fn sample_window(
        &self,
        west: f64,
        south: f64,
        east: f64,
        north: f64,
    ) -> Result<Vec<f64>, TilerError> {
        let n = self.grid_size;
        let inv_n1 = 1.0 / (n - 1) as f64;

        let mut heights = vec![0.0f64; n * n];
        for row in 0..n {
            for col in 0..n {
                // Exact vertex positions — col=0 = west edge, col=n-1 = east edge.
                let lon = west + col as f64 * inv_n1 * (east - west);
                let lat = south + row as f64 * inv_n1 * (north - south);

                // Transform WGS84 → source CRS when needed.
                let (src_x, src_y) = if let Some(inv) = &self.inv_transformer {
                    let c = inv
                        .transform(&Coordinate::new(lon, lat))
                        .map_err(|e| TilerError::Crs(e.to_string()))?;
                    (c.x, c.y)
                } else {
                    (lon, lat)
                };

                let (src_col, src_row) = pixel_from_geo(src_x, src_y, &self.geotransform);
                heights[row * n + col] = self.bilinear(src_col, src_row).unwrap_or(0.0);
            }
        }
        Ok(heights)
    }

    /// Bilinear interpolation from the in-memory raster.
    ///
    /// Returns `None` for points outside the DEM extent (the caller uses
    /// `unwrap_or(0.0)` which maps them to sea level / ellipsoid height).
    /// Within the DEM, bilinear neighbours that fall just outside the raster
    /// edge are clamped so that samples right at the boundary don't drop to
    /// sea level due to a missing one-pixel neighbour.
    fn bilinear(&self, src_col: f64, src_row: f64) -> Option<f64> {
        let max_col = (self.width - 1) as f64;
        let max_row = (self.height - 1) as f64;

        // Return None for points clearly outside the DEM extent.
        // Allow a half-pixel margin so that the bilinear footprint of a sample
        // right at the raster edge is still handled correctly.
        if src_col < -0.5 || src_col > max_col + 0.5 || src_row < -0.5 || src_row > max_row + 0.5 {
            return None;
        }

        // Clamp only for the bilinear neighbour access at the DEM edge.
        let src_col = src_col.clamp(0.0, max_col);
        let src_row = src_row.clamp(0.0, max_row);

        let col0 = src_col.floor() as isize;
        let row0 = src_row.floor() as isize;
        let dx = src_col - col0 as f64;
        let dy = src_row - row0 as f64;

        let v00 = self.pixel(col0, row0)?;
        let v10 = self.pixel(col0 + 1, row0).unwrap_or(v00);
        let v01 = self.pixel(col0, row0 + 1).unwrap_or(v00);
        let v11 = self.pixel(col0 + 1, row0 + 1).unwrap_or(v00);

        Some(
            v00 * (1.0 - dx) * (1.0 - dy)
                + v10 * dx * (1.0 - dy)
                + v01 * (1.0 - dx) * dy
                + v11 * dx * dy,
        )
    }

    fn pixel(&self, col: isize, row: isize) -> Option<f64> {
        if col < 0 || row < 0 {
            return None;
        }
        let (col, row) = (col as usize, row as usize);
        if col >= self.width || row >= self.height {
            return None;
        }
        let v = self.data[row * self.width + col];
        if let Some(nd) = self.nodata {
            if (v - nd).abs() < 1e-10 || v.is_nan() {
                return None;
            }
        }
        Some(v)
    }
}

/// Forward affine: pixel (col, row) → geographic (x, y).
fn geo_from_pixel(col: f64, row: f64, gt: &[f64; 6]) -> (f64, f64) {
    let x = gt[0] + col * gt[1] + row * gt[2];
    let y = gt[3] + col * gt[4] + row * gt[5];
    (x, y)
}

/// Inverse affine: geographic (x, y) → pixel (col, row).
fn pixel_from_geo(x: f64, y: f64, gt: &[f64; 6]) -> (f64, f64) {
    let det = gt[1] * gt[5] - gt[2] * gt[4];
    if det.abs() < 1e-12 {
        return (f64::NAN, f64::NAN);
    }
    let dx = x - gt[0];
    let dy = y - gt[3];
    let col = (gt[5] * dx - gt[2] * dy) / det;
    let row = (-gt[4] * dx + gt[1] * dy) / det;
    (col, row)
}

/// Reads every strip/tile of the primary image into a flat `Vec<f64>`,
/// row-major top-to-bottom (north-first), converting to f64 based on data type.
fn read_full_raster<S: oxigdal_core::io::DataSource>(
    reader: &CogReader<S>,
    width: usize,
    height: usize,
    data_type: RasterDataType,
    byte_order: ByteOrderType,
    nodata: Option<f64>,
) -> Result<Vec<f64>, TilerError> {
    let info = reader.primary_info();
    let (tiles_x, tiles_y) = (info.tiles_across() as usize, info.tiles_down() as usize);
    let is_tiled = info.tile_width.is_some() && info.tile_height.is_some();

    let tile_w = if is_tiled {
        info.tile_width.unwrap_or(width as u32) as usize
    } else {
        width
    };
    let tile_h = if is_tiled {
        info.tile_height.unwrap_or(height as u32) as usize
    } else {
        info.rows_per_strip.unwrap_or(height as u32) as usize
    };

    let mut out = vec![nodata.unwrap_or(0.0); width * height];

    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let raw = reader.read_tile(0, tx as u32, ty as u32)?;
            let px_row_start = ty * tile_h;
            let px_col_start = tx * tile_w;

            // How many actual rows/cols this tile covers (edge tiles may be smaller).
            let actual_h = (height - px_row_start).min(tile_h);
            let actual_w = (width - px_col_start).min(tile_w);

            for tile_row in 0..actual_h {
                for tile_col in 0..actual_w {
                    let tile_idx = tile_row * tile_w + tile_col;
                    let out_idx = (px_row_start + tile_row) * width + (px_col_start + tile_col);
                    out[out_idx] = bytes_to_f64(&raw, tile_idx, data_type, byte_order)?;
                }
            }
        }
    }

    Ok(out)
}

/// Reads one sample from a raw byte slice at `index`, interpreting it as `data_type`.
fn bytes_to_f64(
    bytes: &[u8],
    index: usize,
    data_type: RasterDataType,
    byte_order: ByteOrderType,
) -> Result<f64, TilerError> {
    let bps = data_type.size_bytes();
    let offset = index * bps;
    let s = bytes
        .get(offset..offset + bps)
        .ok_or(TilerError::UnsupportedDataType(data_type))?;

    let to_u16 = |b: &[u8]| -> u16 {
        match byte_order {
            ByteOrderType::LittleEndian => u16::from_le_bytes([b[0], b[1]]),
            ByteOrderType::BigEndian => u16::from_be_bytes([b[0], b[1]]),
        }
    };
    let to_i16 = |b: &[u8]| -> i16 {
        match byte_order {
            ByteOrderType::LittleEndian => i16::from_le_bytes([b[0], b[1]]),
            ByteOrderType::BigEndian => i16::from_be_bytes([b[0], b[1]]),
        }
    };
    let to_u32 = |b: &[u8]| -> u32 {
        match byte_order {
            ByteOrderType::LittleEndian => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            ByteOrderType::BigEndian => u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        }
    };
    let to_i32 = |b: &[u8]| -> i32 {
        match byte_order {
            ByteOrderType::LittleEndian => i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            ByteOrderType::BigEndian => i32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        }
    };
    let to_f32 = |b: &[u8]| -> f32 {
        match byte_order {
            ByteOrderType::LittleEndian => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            ByteOrderType::BigEndian => f32::from_be_bytes([b[0], b[1], b[2], b[3]]),
        }
    };
    let to_f64 = |b: &[u8]| -> f64 {
        match byte_order {
            ByteOrderType::LittleEndian => {
                f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            }
            ByteOrderType::BigEndian => {
                f64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
            }
        }
    };

    let v = match data_type {
        RasterDataType::UInt8 => s[0] as f64,
        RasterDataType::Int8 => s[0] as i8 as f64,
        RasterDataType::UInt16 => to_u16(s) as f64,
        RasterDataType::Int16 => to_i16(s) as f64,
        RasterDataType::UInt32 => to_u32(s) as f64,
        RasterDataType::Int32 => to_i32(s) as f64,
        RasterDataType::Float32 => to_f32(s) as f64,
        RasterDataType::Float64 => to_f64(s),
        other => return Err(TilerError::UnsupportedDataType(other)),
    };
    Ok(v)
}

/// Convert elevation in metres to the Cesium `u16` quantised range `[0, 32767]`.
///
/// Maps `[-1000 m, 9000 m]` linearly. Values outside that range are clamped.
#[inline]
pub fn elevation_to_u16(h: f64) -> u16 {
    let normalised = (h + 1000.0) / 10000.0;
    (normalised.clamp(0.0, 1.0) * 32767.0).round() as u16
}

/// Convert a Cesium `u16` quantised height back to metres.
#[inline]
pub fn u16_to_elevation(v: u16) -> f64 {
    (v as f64 / 32767.0) * 10000.0 - 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elevation_round_trip() {
        for h in [-1000.0f64, 0.0, 500.0, 3000.0, 8848.0, 9000.0] {
            let q = elevation_to_u16(h);
            let r = u16_to_elevation(q);
            // max quantisation error ~= 10000/32767 ~= 0.305 m
            assert!((h.clamp(-1000.0, 9000.0) - r).abs() < 0.31, "h={h}");
        }
    }

    #[test]
    fn clamp_below_range() {
        assert_eq!(elevation_to_u16(-2000.0), 0);
    }

    #[test]
    fn clamp_above_range() {
        assert_eq!(elevation_to_u16(10000.0), 32767);
    }
}

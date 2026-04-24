//! Raster overlay provider backed by a [`vekuta::GeoJsonDocument`].
//!
//! Rasterizes GeoJSON vector data into RGBA tiles on demand using
//! [`vekuta::VectorRasterizer`].

use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::{Context, Task};
use terra::GlobeRectangle;
use vekuta::{GeoJsonDocument, VectorStyle};

use super::url_template::compute_tile_rectangle;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// A raster overlay that rasterizes a [`GeoJsonDocument`] into image tiles.
///
/// The document is shared via [`Arc`] across all tile requests.
///
/// # Example
/// ```ignore
/// let doc = Arc::new(GeoJsonDocument::from_bytes(geojson_bytes)?);
/// let overlay = VekutaRasterOverlay::new(doc);
/// stratum.add_overlay(overlay);
/// ```
pub struct VekutaRasterOverlay {
    pub document: Arc<GeoJsonDocument>,
    pub style: VectorStyle,
    pub tile_width: u32,
    pub tile_height: u32,
    pub bounds: GlobeRectangle,
    pub minimum_level: u32,
    pub maximum_level: u32,
}

impl VekutaRasterOverlay {
    /// Create an overlay with sensible defaults.
    ///
    /// Bounds are derived from the document's own `bbox` if present, otherwise
    /// the full globe is used.
    pub fn new(document: Arc<GeoJsonDocument>) -> Self {
        let bounds = document.bounds().unwrap_or(GlobeRectangle::MAX);
        Self {
            document,
            style: VectorStyle::default(),
            tile_width: 256,
            tile_height: 256,
            bounds,
            minimum_level: 0,
            maximum_level: 18,
        }
    }

    pub fn with_style(mut self, style: VectorStyle) -> Self {
        self.style = style;
        self
    }

    pub fn with_bounds(mut self, bounds: GlobeRectangle) -> Self {
        self.bounds = bounds;
        self
    }

    pub fn with_max_level(mut self, level: u32) -> Self {
        self.maximum_level = level;
        self
    }

    pub fn with_tile_size(mut self, width: u32, height: u32) -> Self {
        self.tile_width = width;
        self.tile_height = height;
        self
    }
}

impl RasterOverlay for VekutaRasterOverlay {
    fn create_tile_provider(
        &self,
        _context: &Context,
        _accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let provider = VekutaTileProvider {
            document: Arc::clone(&self.document),
            style: self.style.clone(),
            bounds: self.bounds,
            tile_width: self.tile_width,
            tile_height: self.tile_height,
            minimum_level: self.minimum_level,
            maximum_level: self.maximum_level,
        };
        orkester::resolved(Arc::new(provider) as Arc<dyn RasterOverlayTileProvider>)
    }
}

struct VekutaTileProvider {
    document: Arc<GeoJsonDocument>,
    style: VectorStyle,
    bounds: GlobeRectangle,
    tile_width: u32,
    tile_height: u32,
    minimum_level: u32,
    maximum_level: u32,
}

impl RasterOverlayTileProvider for VekutaTileProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        let tile_rect =
            compute_tile_rectangle(x, y, level, &self.bounds, OverlayProjection::Geographic);

        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut rasterizer =
                vekuta::VectorRasterizer::new(tile_rect, self.tile_width, self.tile_height);
            rasterizer.draw_object(&self.document.root, &self.style);
            let pixels = rasterizer.finish();
            orkester::resolved(Ok(RasterOverlayTile {
                pixels: Arc::from(pixels.as_slice()),
                width: self.tile_width,
                height: self.tile_height,
                rectangle: tile_rect,
                projection: OverlayProjection::Geographic,
            }))
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Rasterization is not available on WASM (no tiny-skia).
            // Return a transparent tile.
            let n = (self.tile_width * self.tile_height * 4) as usize;
            orkester::resolved(Ok(RasterOverlayTile {
                pixels: Arc::from(vec![0u8; n].as_slice()),
                width: self.tile_width,
                height: self.tile_height,
                rectangle: tile_rect,
                projection: OverlayProjection::Geographic,
            }))
        }
    }

    fn bounds(&self) -> GlobeRectangle {
        self.bounds
    }

    fn minimum_level(&self) -> u32 {
        self.minimum_level
    }

    fn maximum_level(&self) -> u32 {
        self.maximum_level
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }

    fn projection(&self) -> OverlayProjection {
        OverlayProjection::Geographic
    }
}

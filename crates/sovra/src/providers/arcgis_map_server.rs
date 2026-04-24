//! ArcGIS MapServer raster overlay.
//!
//! Fetches `?f=json` metadata from an ArcGIS MapServer REST endpoint to
//! discover tile dimensions, LODs, and projection, then serves tiles via
//! the `tile/{level}/{row}/{col}` URL scheme.
//!
//! Row/col notation: ArcGIS counts rows from the **top** (north), which
//! matches our `reverseY` convention.

use std::sync::Arc;

use courtier::AssetAccessor;
use courtier::esri::MapServerClient;
use courtier::esri::types::{MapServerMetadata, TileInfo};
use orkester::{Context, Task};

use super::url_template::compute_tile_rectangle;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// A raster overlay backed by an ArcGIS MapServer cached tile service.
///
/// Fetches service metadata on `create_tile_provider` to discover available
/// LODs, tile dimensions, and projection. Then serves tiles using the
/// standard ArcGIS `tile/{level}/{row}/{col}` URL scheme.
pub struct ArcGisMapServerRasterOverlay {
    base_url: String,
    /// Optional HTTP headers (e.g. `Authorization: Bearer …` for token-auth services).
    headers: Vec<(String, String)>,
    /// Override for the minimum tile level (default: 0).
    minimum_level: u32,
    /// Override for the maximum tile level. `None` = use the highest LOD reported
    /// by the service metadata.
    maximum_level: Option<u32>,
    /// Override for the geographic bounds. `None` = derive from `fullExtent`
    /// in the service metadata.
    bounds: Option<terra::GlobeRectangle>,
}

impl ArcGisMapServerRasterOverlay {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            headers: Vec::new(),
            minimum_level: 0,
            maximum_level: None,
            bounds: None,
        }
    }

    pub fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }

    pub fn with_maximum_level(mut self, level: u32) -> Self {
        self.maximum_level = Some(level);
        self
    }

    pub fn with_bounds(mut self, bounds: terra::GlobeRectangle) -> Self {
        self.bounds = Some(bounds);
        self
    }
}

impl RasterOverlay for ArcGisMapServerRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let client = MapServerClient::new(self.base_url.clone(), Arc::clone(accessor));
        let headers = self.headers.clone();
        let minimum_level = self.minimum_level;
        let maximum_level_override = self.maximum_level;
        let bounds_override = self.bounds;
        let ctx = context.clone();
        let accessor = Arc::clone(accessor);

        client.metadata().then(&ctx.clone(), move |result| {
            let provider: Arc<dyn RasterOverlayTileProvider> = match result {
                Err(e) => {
                    log::warn!("ArcGIS MapServer metadata fetch failed: {e}");
                    Arc::new(super::bing_maps::NoopTileProvider)
                }
                Ok(meta) => {
                    match build_provider(
                        &meta,
                        client,
                        headers,
                        minimum_level,
                        maximum_level_override,
                        bounds_override,
                        accessor,
                        ctx,
                    ) {
                        Ok(p) => Arc::new(p),
                        Err(e) => {
                            log::warn!("ArcGIS MapServer provider build failed: {e}");
                            Arc::new(super::bing_maps::NoopTileProvider)
                        }
                    }
                }
            };
            orkester::resolved(provider)
        })
    }
}

struct ArcGisTileProvider {
    client: MapServerClient,
    headers: Vec<(String, String)>,
    bounds: terra::GlobeRectangle,
    minimum_level: u32,
    maximum_level: u32,
    projection: OverlayProjection,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
}

impl ArcGisTileProvider {
    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        // ArcGIS row = y from top. Our (x, y, level) uses y=0 at south (TMS).
        // Convert: row = tiles_at_level - 1 - y.
        let tiles_at_level = 1u64 << level;
        let row = (tiles_at_level.saturating_sub(1).saturating_sub(y as u64)) as u32;
        self.client.tile_url(level, row, x)
    }
}

impl RasterOverlayTileProvider for ArcGisTileProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        let url = self.build_url(x, y, level);
        let rect = compute_tile_rectangle(x, y, level, &self.bounds, self.projection);
        super::fetch_and_decode_tile(
            &self.accessor,
            self.ctx.clone(),
            &url,
            &self.headers,
            rect,
            self.projection,
        )
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        self.bounds
    }

    fn maximum_level(&self) -> u32 {
        self.maximum_level
    }

    fn minimum_level(&self) -> u32 {
        self.minimum_level
    }

    fn projection(&self) -> OverlayProjection {
        self.projection
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }
}

fn build_provider(
    meta: &MapServerMetadata,
    client: MapServerClient,
    headers: Vec<(String, String)>,
    minimum_level: u32,
    maximum_level_override: Option<u32>,
    bounds_override: Option<terra::GlobeRectangle>,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
) -> Result<ArcGisTileProvider, String> {
    let tile_info = meta
        .tile_info
        .as_ref()
        .ok_or("ArcGIS MapServer has no tileInfo — service may not be a cached tile layer")?;

    let maximum_level =
        maximum_level_override.unwrap_or_else(|| max_level_from_tile_info(tile_info));
    let projection = projection_from_tile_info(tile_info);
    let bounds = bounds_override
        .or_else(|| bounds_from_metadata(meta, projection))
        .unwrap_or_else(|| default_bounds_for(projection));

    Ok(ArcGisTileProvider {
        client,
        headers,
        bounds,
        minimum_level,
        maximum_level,
        projection,
        accessor,
        ctx,
    })
}

fn max_level_from_tile_info(tile_info: &TileInfo) -> u32 {
    tile_info.lods.iter().map(|l| l.level).max().unwrap_or(18)
}

fn projection_from_tile_info(tile_info: &TileInfo) -> OverlayProjection {
    if tile_info.spatial_reference.is_web_mercator() {
        OverlayProjection::WebMercator
    } else {
        OverlayProjection::Geographic
    }
}

fn bounds_from_metadata(
    meta: &MapServerMetadata,
    projection: OverlayProjection,
) -> Option<terra::GlobeRectangle> {
    let extent = meta.full_extent.as_ref()?;
    Some(extent_to_globe_rectangle(extent, projection))
}

fn extent_to_globe_rectangle(
    extent: &courtier::esri::types::Extent,
    projection: OverlayProjection,
) -> terra::GlobeRectangle {
    match projection {
        OverlayProjection::WebMercator => {
            // For web mercator extents, xmin/xmax are in metres (-20037508 to 20037508).
            // Clamp to valid web mercator lat range.
            terra::GlobeRectangle::from_degrees(
                extent.xmin.max(-180.0).min(180.0),
                extent.ymin.max(-85.051_128_78).min(85.051_128_78),
                extent.xmax.max(-180.0).min(180.0),
                extent.ymax.max(-85.051_128_78).min(85.051_128_78),
            )
        }
        OverlayProjection::Geographic => terra::GlobeRectangle::from_degrees(
            extent.xmin.max(-180.0),
            extent.ymin.max(-90.0),
            extent.xmax.min(180.0),
            extent.ymax.min(90.0),
        ),
    }
}

fn default_bounds_for(projection: OverlayProjection) -> terra::GlobeRectangle {
    match projection {
        OverlayProjection::WebMercator => {
            terra::GlobeRectangle::from_degrees(-180.0, -85.051_128_78, 180.0, 85.051_128_78)
        }
        OverlayProjection::Geographic => terra::GlobeRectangle::MAX,
    }
}

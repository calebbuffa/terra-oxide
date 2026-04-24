//! Cesium Ion raster overlay.
//!
//! Resolves the asset endpoint via `courtier::ion::Connection`, then fetches
//! tiles from the returned TMS URL with the Bearer access token.

use std::sync::Arc;

use courtier::ion::Connection;
use courtier::{AssetAccessor, FetchError};
use orkester::{Context, Task};

use super::url_template::compute_tile_rectangle;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// A raster overlay that streams imagery from a Cesium Ion asset.
///
/// The Ion asset endpoint is resolved once during `create_tile_provider`.
/// Tiles are served via a TMS-compatible URL built from the endpoint response.
///
/// ```no_run
/// use std::sync::Arc;
/// use courtier::ion::Connection;
/// use sovra::providers::IonRasterOverlay;
///
/// // `connection` must be backed by an accessor with a BearerTokenAuth
/// // carrying the Ion access token.
/// let connection: Arc<Connection> = todo!("build from HttpAccessor + BearerTokenAuth");
/// let overlay = IonRasterOverlay::new(connection, 3954);
/// ```
pub struct IonRasterOverlay {
    connection: Arc<Connection>,
    asset_id: u64,
    /// Tile image file extension (default `"png"`).
    file_extension: String,
    minimum_level: u32,
    /// Maximum tile level. `None` means use whatever the endpoint metadata says
    /// (falling back to 18).
    maximum_level: Option<u32>,
    tile_width: u32,
    tile_height: u32,
    projection: OverlayProjection,
    bounds: Option<terra::GlobeRectangle>,
}

impl IonRasterOverlay {
    pub fn new(connection: Arc<Connection>, asset_id: u64) -> Self {
        Self {
            connection,
            asset_id,
            file_extension: "png".to_owned(),
            minimum_level: 0,
            maximum_level: None,
            tile_width: 256,
            tile_height: 256,
            projection: OverlayProjection::WebMercator,
            bounds: None,
        }
    }

    pub fn with_file_extension(mut self, ext: impl Into<String>) -> Self {
        self.file_extension = ext.into();
        self
    }

    pub fn with_maximum_level(mut self, level: u32) -> Self {
        self.maximum_level = Some(level);
        self
    }

    pub fn with_projection(mut self, projection: OverlayProjection) -> Self {
        self.projection = projection;
        self
    }

    pub fn with_bounds(mut self, bounds: terra::GlobeRectangle) -> Self {
        self.bounds = Some(bounds);
        self
    }
}

impl RasterOverlay for IonRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        _accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let connection = Arc::clone(&self.connection);
        let asset_id = self.asset_id;
        let file_extension = self.file_extension.clone();
        let minimum_level = self.minimum_level;
        let maximum_level = self.maximum_level;
        let tile_width = self.tile_width;
        let tile_height = self.tile_height;
        let projection = self.projection;
        let bounds = self.bounds;
        let ctx = context.clone();
        let accessor = Arc::clone(connection.accessor());

        connection
            .asset_endpoint(asset_id)
            .then(&ctx.clone(), move |endpoint_result| {
                let provider: Arc<dyn RasterOverlayTileProvider> = match endpoint_result {
                    Err(e) => {
                        log::warn!("Ion asset endpoint fetch failed for asset {asset_id}: {e}");
                        Arc::new(super::bing_maps::NoopTileProvider)
                    }
                    Ok(endpoint) => {
                        let headers = vec![(
                            "Authorization".to_owned(),
                            format!("Bearer {}", endpoint.access_token),
                        )];
                        let base_url = endpoint.url.trim_end_matches('/').to_owned();
                        let max_level = maximum_level.unwrap_or(18);
                        let provider_bounds =
                            bounds.unwrap_or_else(|| default_bounds_for(projection));

                        Arc::new(IonTileProvider {
                            base_url,
                            file_extension,
                            headers,
                            bounds: provider_bounds,
                            minimum_level,
                            maximum_level: max_level,
                            tile_width,
                            tile_height,
                            projection,
                            accessor,
                            ctx,
                        })
                    }
                };
                orkester::resolved(provider)
            })
    }
}

struct IonTileProvider {
    /// TMS base URL from the Ion endpoint (trailing slash stripped).
    base_url: String,
    file_extension: String,
    headers: Vec<(String, String)>,
    bounds: terra::GlobeRectangle,
    minimum_level: u32,
    maximum_level: u32,
    tile_width: u32,
    tile_height: u32,
    projection: OverlayProjection,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
}

impl IonTileProvider {
    /// Ion imagery uses TMS convention: `{base}/{z}/{x}/{reverseY}.{ext}`
    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        let tiles_at_level = 1u64 << level;
        let reverse_y = (tiles_at_level.saturating_sub(1).saturating_sub(y as u64)) as u32;
        format!(
            "{}/{level}/{x}/{reverse_y}.{}",
            self.base_url, self.file_extension
        )
    }
}

impl RasterOverlayTileProvider for IonTileProvider {
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

fn default_bounds_for(projection: OverlayProjection) -> terra::GlobeRectangle {
    match projection {
        OverlayProjection::WebMercator => {
            terra::GlobeRectangle::from_degrees(-180.0, -85.051_128_78, 180.0, 85.051_128_78)
        }
        OverlayProjection::Geographic => terra::GlobeRectangle::MAX,
    }
}

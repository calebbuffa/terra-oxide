//! URL template raster overlay - the most common overlay source.
//!
//! Fetches tiles from a server using a URL pattern with `{x}`, `{y}`, `{z}`
//! (and variants) substituted per tile request.

use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::{Context, Task};
use outil::expand_tile_url;

use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider,
    get_tiles_for_extent,
};
use terra::WebMercatorProjection;

/// Options for constructing a [`UrlTemplateRasterOverlay`].
#[derive(Clone, Debug)]
pub struct UrlTemplateOptions {
    /// URL template with substitution tokens. Supported tokens:
    /// - `{x}` - tile column
    /// - `{y}` - tile row
    /// - `{z}` - zoom level
    /// - `{reverseY}` - `(2^z - 1 - y)`, for TMS-style Y-axis
    /// - `{reverseX}` - `(2^z - 1 - x)`
    /// - `{s}` - subdomain selected by round-robin from [`subdomains`](UrlTemplateOptions::subdomains)
    pub url: String,
    /// HTTP headers to send with each tile request.
    pub headers: Vec<(String, String)>,
    /// Geographic bounds of the overlay in radians.
    pub bounds: terra::GlobeRectangle,
    /// Tile width in pixels (default 256).
    pub tile_width: u32,
    /// Tile height in pixels (default 256).
    pub tile_height: u32,
    /// Minimum zoom level (default 0).
    pub minimum_level: u32,
    /// Maximum zoom level (default 24).
    pub maximum_level: u32,
    /// Number of color channels in the decoded tiles (default 4 = RGBA).
    pub channels: u32,
    /// Projection used by this tile source.
    ///
    /// Use [`OverlayProjection::WebMercator`] for OSM-style slippy-map tiles
    /// (OpenStreetMap, Bing Maps, Google Maps, most WMTS EPSG:3857 sources).
    /// Defaults to [`OverlayProjection::Geographic`].
    pub projection: OverlayProjection,
    /// Subdomain list used to expand `{s}` in the URL template.
    ///
    /// The subdomain is selected deterministically from `(x + y + z) %
    /// subdomains.len()` so the same tile always maps to the same server.
    /// Defaults to `["a", "b", "c"]`.
    pub subdomains: Vec<String>,
}

impl Default for UrlTemplateOptions {
    fn default() -> Self {
        Self {
            url: String::new(),
            headers: Vec::new(),
            bounds: terra::GlobeRectangle::MAX,
            tile_width: 256,
            tile_height: 256,
            minimum_level: 0,
            maximum_level: 24,
            channels: 4,
            projection: OverlayProjection::Geographic,
            subdomains: vec!["a".into(), "b".into(), "c".into()],
        }
    }
}

/// A raster overlay that fetches tiles from a URL template.
pub struct UrlTemplateRasterOverlay {
    options: UrlTemplateOptions,
}

impl UrlTemplateRasterOverlay {
    pub fn new(options: UrlTemplateOptions) -> Self {
        Self { options }
    }
}

impl RasterOverlay for UrlTemplateRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let provider = UrlTemplateTileProvider {
            options: self.options.clone(),
            accessor: Arc::clone(accessor),
            ctx: context.clone(),
        };
        orkester::resolved(Arc::new(provider) as Arc<dyn RasterOverlayTileProvider>)
    }
}

pub(crate) struct UrlTemplateTileProvider {
    options: UrlTemplateOptions,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
}

impl UrlTemplateTileProvider {
    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        let reverse_y = (1u64 << level).saturating_sub(1).saturating_sub(y as u64);
        let reverse_x = (1u64 << level).saturating_sub(1).saturating_sub(x as u64);
        let x_s = x.to_string();
        let y_s = y.to_string();
        let level_s = level.to_string();
        let ry_s = reverse_y.to_string();
        let rx_s = reverse_x.to_string();

        // Deterministic subdomain selection: same (x, y, z) always maps to the
        // same server, while different tiles are spread across subdomains.
        let subdomain_idx = if !self.options.subdomains.is_empty() {
            ((x as usize)
                .wrapping_add(y as usize)
                .wrapping_add(level as usize))
                % self.options.subdomains.len()
        } else {
            0
        };
        let subdomain = self
            .options
            .subdomains
            .get(subdomain_idx)
            .map(|s| s.as_str())
            .unwrap_or("");

        expand_tile_url(
            &self.options.url,
            &[
                ("x", &x_s),
                ("y", &y_s),
                ("z", &level_s),
                ("reverseY", &ry_s),
                ("reverseX", &rx_s),
                ("s", subdomain),
            ],
        )
    }
}

impl RasterOverlayTileProvider for UrlTemplateTileProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, crate::overlay::TileFetchError>> {
        let url = self.build_url(x, y, level);
        let rect =
            compute_tile_rectangle(x, y, level, &self.options.bounds, self.options.projection);
        super::fetch_and_decode_tile(
            &self.accessor,
            self.ctx.clone(),
            &url,
            &self.options.headers,
            rect,
            self.options.projection,
        )
    }

    fn projection(&self) -> OverlayProjection {
        self.options.projection
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        self.options.bounds
    }

    fn maximum_level(&self) -> u32 {
        self.options.maximum_level
    }

    fn minimum_level(&self) -> u32 {
        self.options.minimum_level
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }
}

/// Compute the geographic rectangle for a tile at `(x, y, level)` within the
/// provider's bounds.
///
/// For [`OverlayProjection::Geographic`] providers the tile grid is uniform in
/// geographic lat/lon.  For [`OverlayProjection::WebMercator`] the grid is
/// uniform in Mercator Y - matching the OSM / slippy-map tile scheme -  and
/// the geographic lat boundaries are derived from the Mercator Y boundaries.
pub(crate) fn compute_tile_rectangle(
    x: u32,
    y: u32,
    level: u32,
    provider_bounds: &terra::GlobeRectangle,
    projection: OverlayProjection,
) -> terra::GlobeRectangle {
    let tiles = (1u64 << level) as f64;
    let full_lon = provider_bounds.east - provider_bounds.west;
    let tile_lon = full_lon / tiles;
    let lon_west = provider_bounds.west + x as f64 * tile_lon;
    let lon_east = provider_bounds.west + (x + 1) as f64 * tile_lon;

    let (lat_south, lat_north) = match projection {
        OverlayProjection::Geographic => {
            let full_lat = provider_bounds.north - provider_bounds.south;
            let tile_lat = full_lat / tiles;
            (
                provider_bounds.south + y as f64 * tile_lat,
                provider_bounds.south + (y + 1) as f64 * tile_lat,
            )
        }
        OverlayProjection::WebMercator => {
            // The tile grid is uniform in Mercator Y (atanh(sin(lat))).
            // y=0 is the southernmost tile; reverseY is used in URLs to match
            // OSM's north-origin convention.
            let merc_pb_south =
                WebMercatorProjection::geodetic_latitude_to_mercator_angle(provider_bounds.south);
            let merc_pb_north =
                WebMercatorProjection::geodetic_latitude_to_mercator_angle(provider_bounds.north);
            let full_merc = merc_pb_north - merc_pb_south;
            let tile_merc = full_merc / tiles;
            (
                WebMercatorProjection::mercator_angle_to_geodetic_latitude(
                    merc_pb_south + y as f64 * tile_merc,
                ),
                WebMercatorProjection::mercator_angle_to_geodetic_latitude(
                    merc_pb_south + (y + 1) as f64 * tile_merc,
                ),
            )
        }
    };

    terra::GlobeRectangle::new(lon_west, lat_south, lon_east, lat_north)
}

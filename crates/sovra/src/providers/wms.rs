//! Web Map Service (WMS) raster overlay.
//!
//! Fetches map images from an OGC WMS server via GetMap requests.

use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::{Context, Task};

use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider,
    get_tiles_for_extent,
};

/// Options for a WMS overlay.
#[derive(Clone, Debug)]
pub struct WmsOptions {
    /// Base URL of the WMS service (e.g. `https://example.com/wms`).
    pub url: String,
    /// Comma-separated layer names.
    pub layers: String,
    /// WMS version string (default `"1.3.0"`). Controls coordinate axis order.
    pub version: String,
    /// Image format MIME type (default `"image/png"`).
    pub format: String,
    /// Coordinate reference system (default `"EPSG:4326"`).
    pub crs: String,
    /// HTTP headers sent with each request.
    pub headers: Vec<(String, String)>,
    /// Geographic bounds in radians. Defaults to the whole globe.
    pub bounds: Option<terra::GlobeRectangle>,
    /// Tile width in pixels (default 256).
    pub tile_width: u32,
    /// Tile height in pixels (default 256).
    pub tile_height: u32,
    /// Minimum zoom level (default 0).
    pub minimum_level: u32,
    /// Maximum zoom level (default 24).
    pub maximum_level: u32,
    /// Projection used by this WMS source.
    ///
    /// Most WMS services using `EPSG:4326` or `CRS:84` are geographic.
    /// Some services using `EPSG:3857` are WebMercator.
    /// Defaults to [`OverlayProjection::Geographic`].
    pub projection: OverlayProjection,
}

impl Default for WmsOptions {
    fn default() -> Self {
        Self {
            url: String::new(),
            layers: String::new(),
            version: "1.3.0".into(),
            format: "image/png".into(),
            crs: "EPSG:4326".into(),
            headers: Vec::new(),
            bounds: None,
            tile_width: 256,
            tile_height: 256,
            minimum_level: 0,
            maximum_level: 24,
            projection: OverlayProjection::Geographic,
        }
    }
}

/// A raster overlay fetching images from an OGC WMS server.
///
/// Each tile request issues a GetMap call with the tile's geographic extent
/// as the BBOX parameter. WMS 1.3.0+ swaps axis order for geographic CRS
/// (lat,lon instead of lon,lat).
pub struct WebMapServiceRasterOverlay {
    options: WmsOptions,
}

impl WebMapServiceRasterOverlay {
    pub fn new(options: WmsOptions) -> Self {
        Self { options }
    }
}

impl RasterOverlay for WebMapServiceRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let provider = WmsTileProvider {
            options: self.options.clone(),
            accessor: Arc::clone(accessor),
            ctx: context.clone(),
        };
        orkester::resolved(Arc::new(provider) as Arc<dyn RasterOverlayTileProvider>)
    }
}

struct WmsTileProvider {
    options: WmsOptions,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
}

impl WmsTileProvider {
    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        let bounds = self.options.bounds.unwrap_or(terra::GlobeRectangle::MAX);

        // Compute the geographic extent of this tile.
        let x_tiles = 1u64 << level;
        let y_tiles = 1u64 << level;
        let lon_span = bounds.east - bounds.west;
        let lat_span = bounds.north - bounds.south;
        let tile_lon = lon_span / x_tiles as f64;
        let tile_lat = lat_span / y_tiles as f64;

        let west = bounds.west + x as f64 * tile_lon;
        let south = bounds.south + y as f64 * tile_lat;
        let east = west + tile_lon;
        let north = south + tile_lat;

        // Convert to degrees for the BBOX.
        let to_deg = |r: f64| r.to_degrees();

        // WMS 1.3.0+ with geographic CRS uses lat,lon ordering.
        //
        // Strictly per OGC, `CRS:84` is defined as lon,lat order, but
        // cesium-native's `WebMapServiceRasterOverlay` (the reference we
        // mirror) always emits south,west,north,east for *any* geographic
        // CRS in WMS 1.3.0, so we do the same for compatibility.
        let is_130_plus = self.options.version.starts_with("1.3");
        let is_geographic = self.options.crs == "EPSG:4326" || self.options.crs == "CRS:84";

        let bbox = if is_130_plus && is_geographic {
            // WMS 1.3.0 geographic: lat,lon
            format!(
                "{},{},{},{}",
                to_deg(south),
                to_deg(west),
                to_deg(north),
                to_deg(east)
            )
        } else {
            // Pre-1.3.0 or non-geographic: lon,lat (or x,y for projected)
            format!(
                "{},{},{},{}",
                to_deg(west),
                to_deg(south),
                to_deg(east),
                to_deg(north)
            )
        };

        let crs_key = if is_130_plus { "CRS" } else { "SRS" };

        format!(
            "{}?SERVICE=WMS&VERSION={}&REQUEST=GetMap&LAYERS={}&{}={}&BBOX={}&WIDTH={}&HEIGHT={}&FORMAT={}",
            self.options.url,
            self.options.version,
            self.options.layers,
            crs_key,
            self.options.crs,
            bbox,
            self.options.tile_width,
            self.options.tile_height,
            self.options.format,
        )
    }
}

impl RasterOverlayTileProvider for WmsTileProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, crate::overlay::TileFetchError>> {
        let url = self.build_url(x, y, level);
        let provider_bounds = self.options.bounds.unwrap_or(terra::GlobeRectangle::MAX);
        let rect = super::url_template::compute_tile_rectangle(
            x,
            y,
            level,
            &provider_bounds,
            self.options.projection,
        );
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
        self.options.bounds.unwrap_or(terra::GlobeRectangle::MAX)
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

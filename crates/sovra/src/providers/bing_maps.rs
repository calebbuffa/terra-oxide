//! Bing Maps raster overlay.
//!
//! Fetches the Bing Maps Imagery REST Metadata to discover the tile URL
//! template and subdomain list, then serves tiles using quadkey addressing
//! with Web Mercator projection.

use std::sync::Arc;

use courtier::{AssetAccessor, FetchError, RequestPriority};
use orkester::{Context, Task};
use serde::Deserialize;

use super::url_template::compute_tile_rectangle;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// Bing Maps imagery style.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BingMapsStyle {
    #[default]
    Aerial,
    AerialWithLabels,
    Road,
    CanvasDark,
    CanvasGray,
    CanvasLight,
}

impl BingMapsStyle {
    fn as_str(self) -> &'static str {
        match self {
            BingMapsStyle::Aerial => "Aerial",
            BingMapsStyle::AerialWithLabels => "AerialWithLabelsOnDemand",
            BingMapsStyle::Road => "RoadOnDemand",
            BingMapsStyle::CanvasDark => "CanvasDark",
            BingMapsStyle::CanvasGray => "CanvasGray",
            BingMapsStyle::CanvasLight => "CanvasLight",
        }
    }
}

/// A raster overlay backed by Bing Maps imagery.
///
/// On `create_tile_provider`, fetches the Bing Imagery Metadata API to
/// discover the URL template and subdomain list. Tiles are fetched using
/// quadkey addressing under Web Mercator projection.
///
/// ```no_run
/// use sovra::providers::BingMapsRasterOverlay;
/// use sovra::providers::bing_maps::BingMapsStyle;
///
/// let overlay = BingMapsRasterOverlay::new("YOUR_BING_API_KEY", BingMapsStyle::Aerial);
/// ```
pub struct BingMapsRasterOverlay {
    api_key: String,
    style: BingMapsStyle,
    culture: String,
}

impl BingMapsRasterOverlay {
    pub fn new(api_key: impl Into<String>, style: BingMapsStyle) -> Self {
        Self {
            api_key: api_key.into(),
            style,
            culture: "en-US".to_owned(),
        }
    }

    pub fn with_culture(mut self, culture: impl Into<String>) -> Self {
        self.culture = culture.into();
        self
    }
}

impl RasterOverlay for BingMapsRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let url = format!(
            "https://dev.virtualearth.net/REST/v1/Imagery/Metadata/{}?\
             incl=ImageryProviders&key={}&uriScheme=https&include=ImageryProviders",
            self.style.as_str(),
            self.api_key,
        );
        let accessor = Arc::clone(accessor);
        let ctx = context.clone();

        accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .then(&ctx.clone(), move |result| {
                let resp = result.map_err(|e: FetchError| {
                    // On fetch failure fall back to a no-op provider rather than panicking.
                    log::warn!("Bing Maps metadata fetch failed: {e}");
                    Arc::new(NoopTileProvider) as Arc<dyn RasterOverlayTileProvider>
                });
                match resp {
                    Err(provider) => return orkester::resolved(provider),
                    Ok(resp) => {
                        let provider: Arc<dyn RasterOverlayTileProvider> =
                            match build_provider_from_metadata(&resp.data, &accessor, &ctx) {
                                Ok(p) => Arc::new(p),
                                Err(e) => {
                                    log::warn!("Bing Maps metadata parse failed: {e}");
                                    Arc::new(NoopTileProvider)
                                }
                            };
                        orkester::resolved(provider)
                    }
                }
            })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BingMetadataResponse {
    resource_sets: Vec<BingResourceSet>,
}

#[derive(Deserialize)]
struct BingResourceSet {
    resources: Vec<BingResource>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct BingResource {
    image_url: String,
    image_url_subdomains: Vec<String>,
    image_width: u32,
    image_height: u32,
    zoom_min: u32,
    zoom_max: u32,
}

fn build_provider_from_metadata(
    data: &[u8],
    accessor: &Arc<dyn AssetAccessor>,
    ctx: &Context,
) -> Result<BingMapsTileProvider, String> {
    let meta: BingMetadataResponse = serde_json::from_slice(data).map_err(|e| e.to_string())?;
    let resource = meta
        .resource_sets
        .into_iter()
        .next()
        .and_then(|rs| rs.resources.into_iter().next())
        .ok_or_else(|| "Bing metadata: no resources".to_owned())?;

    if resource.image_url_subdomains.is_empty() {
        return Err("Bing metadata: no subdomains".to_owned());
    }

    Ok(BingMapsTileProvider {
        url_template: resource.image_url,
        subdomains: resource.image_url_subdomains,
        tile_width: resource.image_width,
        tile_height: resource.image_height,
        minimum_level: resource.zoom_min,
        maximum_level: resource.zoom_max,
        accessor: Arc::clone(accessor),
        ctx: ctx.clone(),
    })
}

/// Convert slippy-map tile coordinates (x, y, level) to a Bing Maps quadkey.
///
/// In slippy-map convention y=0 is the northernmost tile row.
fn tile_xy_to_quadkey(x: u32, y: u32, level: u32) -> String {
    let mut quadkey = String::with_capacity(level as usize);
    for i in (0..level).rev() {
        let mut digit = 0u8;
        let mask = 1u32 << i;
        if x & mask != 0 {
            digit |= 1;
        }
        if y & mask != 0 {
            digit |= 2;
        }
        quadkey.push(char::from(b'0' + digit));
    }
    quadkey
}

/// The whole-world Web Mercator bounds in radians (approx ±85.05° lat).
fn web_mercator_bounds() -> terra::GlobeRectangle {
    terra::GlobeRectangle::from_degrees(-180.0, -85.051_128_78, 180.0, 85.051_128_78)
}

struct BingMapsTileProvider {
    url_template: String,
    subdomains: Vec<String>,
    tile_width: u32,
    tile_height: u32,
    minimum_level: u32,
    maximum_level: u32,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
}

impl BingMapsTileProvider {
    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        // Bing uses y=0 at north (same convention as our reverseY / slippy-map y).
        let quadkey = tile_xy_to_quadkey(x, y, level);
        // Rotate subdomain by hashing tile coordinates.
        let idx = ((x as usize).wrapping_add(y as usize)) % self.subdomains.len();
        let subdomain = &self.subdomains[idx];
        self.url_template
            .replace("{quadkey}", &quadkey)
            .replace("{subdomain}", subdomain)
    }
}

impl RasterOverlayTileProvider for BingMapsTileProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        let url = self.build_url(x, y, level);
        let bounds = web_mercator_bounds();
        // Bing uses slippy-map convention (y=0 at top), same as our reverseY.
        let rect = compute_tile_rectangle(x, y, level, &bounds, OverlayProjection::WebMercator);
        super::fetch_and_decode_tile(
            &self.accessor,
            self.ctx.clone(),
            &url,
            &[],
            rect,
            OverlayProjection::WebMercator,
        )
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        web_mercator_bounds()
    }

    fn maximum_level(&self) -> u32 {
        self.maximum_level
    }

    fn minimum_level(&self) -> u32 {
        self.minimum_level
    }

    fn projection(&self) -> OverlayProjection {
        OverlayProjection::WebMercator
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }
}

pub(crate) struct NoopTileProvider;

impl RasterOverlayTileProvider for NoopTileProvider {
    fn get_tile(
        &self,
        _x: u32,
        _y: u32,
        _level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        orkester::resolved(Err(TileFetchError::Decode("provider unavailable".into())))
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        terra::GlobeRectangle::MAX
    }

    fn maximum_level(&self) -> u32 {
        0
    }

    fn tiles_for_extent(&self, _: terra::GlobeRectangle, _: glam::DVec2) -> Vec<(u32, u32, u32)> {
        vec![]
    }
}

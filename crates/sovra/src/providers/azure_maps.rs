//! Azure Maps raster tile provider.

use std::sync::Arc;

use courtier::AssetAccessor;
use orkester::{Context, Task};

use super::url_template::compute_tile_rectangle;
use crate::credit::Credit;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// Azure Maps tile set identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AzureMapsStyle {
    MicrosoftImagery,
    MicrosoftBase,
    MicrosoftBaseLabels,
    MicrosoftBaseHybrid,
    MicrosoftTerrain,
    MicrosoftWeatherInfraredMain,
    MicrosoftWeatherRadarMain,
    MicrosoftDark,
}

impl AzureMapsStyle {
    fn tile_set_id(self) -> &'static str {
        match self {
            Self::MicrosoftImagery => "microsoft.imagery",
            Self::MicrosoftBase => "microsoft.base",
            Self::MicrosoftBaseLabels => "microsoft.base.labels",
            Self::MicrosoftBaseHybrid => "microsoft.base.hybrid",
            Self::MicrosoftTerrain => "microsoft.terrain.main",
            Self::MicrosoftWeatherInfraredMain => "microsoft.weather.infrared.main",
            Self::MicrosoftWeatherRadarMain => "microsoft.weather.radar.main",
            Self::MicrosoftDark => "microsoft.base.darkgrey",
        }
    }

    fn max_zoom(self) -> u32 {
        match self {
            Self::MicrosoftImagery => 19,
            Self::MicrosoftWeatherInfraredMain | Self::MicrosoftWeatherRadarMain => 15,
            _ => 22,
        }
    }
}

/// Options for the Azure Maps tile provider.
#[derive(Clone, Debug)]
pub struct AzureMapsOptions {
    /// Azure Maps subscription key.
    pub subscription_key: String,
    /// Tile set / style to request.
    pub style: AzureMapsStyle,
    /// Tile size in pixels — `256` or `512`.
    pub tile_size: u32,
    /// Optional language code for map labels (e.g. `"en-US"`).
    pub language: Option<String>,
}

/// A raster overlay backed by the Azure Maps Render v2 tile service.
///
/// Tiles are fetched via:
/// `https://atlas.microsoft.com/map/tile?api-version=2022-08-01&tilesetId={id}&zoom={z}&x={x}&y={y}&tileSize={size}`
/// The subscription key is sent as the `Ocp-Apim-Subscription-Key` request header.
pub struct AzureMapsProvider {
    options: AzureMapsOptions,
    accessor: Arc<dyn AssetAccessor>,
    ctx: Context,
    /// Validated tile size (256 or 512; defaults to 256 for any other value).
    tile_size: u32,
}

impl AzureMapsProvider {
    pub fn new(
        options: AzureMapsOptions,
        accessor: Arc<dyn AssetAccessor>,
        ctx: Context,
    ) -> Arc<Self> {
        let tile_size = if options.tile_size == 256 || options.tile_size == 512 {
            options.tile_size
        } else {
            log::warn!(
                "Azure Maps: unsupported tile_size {}; falling back to 256",
                options.tile_size
            );
            256
        };
        Arc::new(Self {
            options,
            accessor,
            ctx,
            tile_size,
        })
    }

    fn build_url(&self, x: u32, y: u32, level: u32) -> String {
        let tileset_id = self.options.style.tile_set_id();
        let tile_size = self.tile_size;
        let mut url = format!(
            "https://atlas.microsoft.com/map/tile\
             ?api-version=2022-08-01\
             &tilesetId={tileset_id}\
             &zoom={level}\
             &x={x}\
             &y={y}\
             &tileSize={tile_size}"
        );
        if let Some(ref lang) = self.options.language {
            url.push_str("&language=");
            url.push_str(&url_encode_simple(lang));
        }
        url
    }
}

impl RasterOverlayTileProvider for AzureMapsProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        let url = self.build_url(x, y, level);
        let bounds = terra::GlobeRectangle::WEB_MERCATOR;
        let rect = compute_tile_rectangle(x, y, level, &bounds, OverlayProjection::WebMercator);
        let headers = vec![(
            "Ocp-Apim-Subscription-Key".to_string(),
            self.options.subscription_key.clone(),
        )];
        super::fetch_and_decode_tile(
            &self.accessor,
            self.ctx.clone(),
            &url,
            &headers,
            rect,
            OverlayProjection::WebMercator,
        )
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        terra::GlobeRectangle::WEB_MERCATOR
    }

    fn maximum_level(&self) -> u32 {
        self.options.style.max_zoom()
    }

    fn minimum_level(&self) -> u32 {
        0
    }

    fn projection(&self) -> OverlayProjection {
        OverlayProjection::WebMercator
    }

    fn credits(&self) -> Vec<Credit> {
        vec![Credit::new("© Microsoft")]
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }
}

/// User-facing overlay source implementing [`RasterOverlay`].
///
/// Add this to an [`OverlayEngine`](crate::OverlayEngine) to stream Azure Maps
/// raster tiles.
pub struct AzureMapsRasterOverlay {
    options: AzureMapsOptions,
}

impl AzureMapsRasterOverlay {
    pub fn new(options: AzureMapsOptions) -> Self {
        Self { options }
    }
}

impl RasterOverlay for AzureMapsRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let provider =
            AzureMapsProvider::new(self.options.clone(), Arc::clone(accessor), context.clone());
        orkester::resolved(provider as Arc<dyn RasterOverlayTileProvider>)
    }
}

/// Percent-encode a string for use in a URL query parameter value.
///
/// Passes through unreserved characters (`A-Z a-z 0-9 - _ . ~`) unchanged and
/// percent-encodes everything else as `%XX` using the UTF-8 code units.
fn url_encode_simple(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, 'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~') {
            out.push(c);
        } else {
            let mut buf = [0u8; 4];
            let encoded = c.encode_utf8(&mut buf);
            for &b in encoded.as_bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

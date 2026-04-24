//! Concrete raster overlay tile provider implementations.

pub mod arcgis_map_server;
pub mod azure_maps;
pub mod bing_maps;
pub mod google_maps;
pub(crate) mod image_decode;
pub mod ion;
pub(crate) mod tms;
pub(crate) mod url_template;
pub mod vekuta;
pub(crate) mod wms;
pub(crate) mod wmts;

pub use arcgis_map_server::ArcGisMapServerRasterOverlay;
pub use azure_maps::{AzureMapsOptions, AzureMapsProvider, AzureMapsRasterOverlay, AzureMapsStyle};
pub use bing_maps::{BingMapsRasterOverlay, BingMapsStyle};
pub use google_maps::{
    GoogleMapsMapType, GoogleMapsRasterOverlay, GoogleMapsTilesOptions, GoogleMapsTilesProvider,
};
pub use ion::IonRasterOverlay;
pub use tms::TileMapServiceRasterOverlay;
pub use url_template::UrlTemplateRasterOverlay;
pub use vekuta::VekutaRasterOverlay;
pub use wms::WebMapServiceRasterOverlay;
pub use wmts::WebMapTileServiceRasterOverlay;

use std::sync::Arc;

use courtier::{AssetAccessor, RequestPriority};
use orkester::{Context, Task};

use crate::overlay::{OverlayProjection, RasterOverlayTile, TileFetchError};

/// Fetch an encoded tile from `url` via `accessor`, decode it on a
/// background worker, and assemble the final [`RasterOverlayTile`].
///
/// All raster providers funnel through this helper so PNG/JPEG/WebP
/// decompression never runs on the thread that completes HTTP requests
/// (which may be the main thread in single-threaded deployments such as
/// WebAssembly). The decode step is chained with [`Task::then`] on `ctx`,
/// so it runs on `ctx`'s executor - typically the background thread pool.
///
/// Failures (transport errors, non-2xx responses, corrupt images) are
/// returned as [`TileFetchError`] in the task payload so that one bad tile
/// never panics the worker thread.
pub(crate) fn fetch_and_decode_tile(
    accessor: &Arc<dyn AssetAccessor>,
    ctx: Context,
    url: &str,
    headers: &[(String, String)],
    rectangle: terra::GlobeRectangle,
    projection: OverlayProjection,
) -> Task<Result<RasterOverlayTile, TileFetchError>> {
    accessor
        .get(url, headers, RequestPriority::NORMAL, None)
        .then(&ctx, move |result| {
            let resp = result.map_err(TileFetchError::Fetch)?;
            resp.check_status().map_err(TileFetchError::Fetch)?;
            let decoded = image_decode::decode_image_to_rgba(&resp.data)
                .map_err(|e| TileFetchError::Decode(e.to_string().into()))?;
            Ok(RasterOverlayTile {
                pixels: Arc::from(decoded.pixels),
                width: decoded.width,
                height: decoded.height,
                rectangle,
                projection,
            })
        })
}

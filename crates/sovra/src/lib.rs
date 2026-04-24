mod basemaps;
mod compositing;
pub mod credit;
mod engine;
mod event;
pub mod gltf;
mod hierarchy;
mod overlay;
pub mod providers;
pub mod upsample;

pub use basemaps::Basemap;
pub use compositing::composite_overlay_tiles;
pub use credit::Credit;
pub use engine::{
    DEFAULT_TARGET_TEXELS_PER_RADIAN, OverlayEngine, OverlayEngineOptions, OverlayTileInfo,
    OverlayViewInfo,
};
pub use event::OverlayEvent;
pub use gltf::{
    apply_raster_overlay, compute_overlay_uvs, compute_overlay_uvs_from_positions,
    encode_overlay_png, extract_ecef_positions,
};
pub use hierarchy::OverlayHierarchy;
pub use overlay::{
    OverlayCollection, OverlayId, OverlayProjection, RasterOverlay, RasterOverlayTile,
    RasterOverlayTileProvider, TileFetchError, get_tiles_for_extent,
};
pub use providers::tms::TmsOptions;
pub use providers::url_template::UrlTemplateOptions;
pub use providers::wms::WmsOptions;
pub use providers::wmts::WmtsOptions;
pub use providers::{
    ArcGisMapServerRasterOverlay, AzureMapsOptions, AzureMapsProvider, AzureMapsRasterOverlay,
    AzureMapsStyle, BingMapsRasterOverlay, BingMapsStyle, GoogleMapsMapType,
    GoogleMapsRasterOverlay, GoogleMapsTilesOptions, GoogleMapsTilesProvider, IonRasterOverlay,
    TileMapServiceRasterOverlay, UrlTemplateRasterOverlay, VekutaRasterOverlay,
    WebMapServiceRasterOverlay, WebMapTileServiceRasterOverlay,
};
pub use upsample::{UpsampleChildId, upsample_gltf_for_overlay};

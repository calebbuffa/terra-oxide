//! ArcGIS REST API service clients.

pub mod basemaps;
pub mod feature_server;
pub mod image_server;
pub mod map_server;
pub mod scene_server;
pub mod types;

pub use basemaps::{
    Basemap, WORLD_ELEVATION_URL, WORLD_HILLSHADE_URL, WORLD_IMAGERY_URL, WORLD_OCEANS_URL,
    basemap_url,
};
pub use feature_server::{FeatureServerClient, QueryGeometry, QueryParams};
pub use image_server::ImageServerClient;
pub use map_server::MapServerClient;
pub use scene_server::{SceneServerClient, SceneServerInfo};
pub use types::{
    Extent, ImageServerMetadata, LodEntry, MapServerMetadata, SpatialReference, TileInfo,
    TileOrigin, TileRange, TilemapResponse,
};

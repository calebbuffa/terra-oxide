//! Well-known ArcGIS basemap service URLs.

/// ArcGIS World Imagery tile service (MapServer).
pub const WORLD_IMAGERY_URL: &str =
    "https://ibasemaps-api.arcgis.com/arcgis/rest/services/World_Imagery/MapServer";

/// ArcGIS World Hillshade tile service (MapServer).
pub const WORLD_HILLSHADE_URL: &str =
    "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Elevation/World_Hillshade/MapServer";

/// ArcGIS World Oceans tile service (MapServer).
pub const WORLD_OCEANS_URL: &str =
    "https://ibasemaps-api.arcgis.com/arcgis/rest/services/Ocean/World_Ocean_Base/MapServer";

/// ArcGIS World Elevation 3D terrain service (ImageServer).
pub const WORLD_ELEVATION_URL: &str =
    "https://elevation3d.arcgis.com/arcgis/rest/services/WorldElevation3D/Terrain3D/ImageServer";

/// Well-known ArcGIS basemap variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Basemap {
    Satellite,
    Hillshade,
    Oceans,
}

/// Returns the canonical URL for the given well-known basemap.
pub fn basemap_url(b: Basemap) -> &'static str {
    match b {
        Basemap::Satellite => WORLD_IMAGERY_URL,
        Basemap::Hillshade => WORLD_HILLSHADE_URL,
        Basemap::Oceans => WORLD_OCEANS_URL,
    }
}

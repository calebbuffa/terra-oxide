//! ArcGIS REST API types shared across service clients.

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpatialReference {
    pub wkid: Option<u32>,
    pub latest_wkid: Option<u32>,
}

impl SpatialReference {
    /// Returns the effective WKID (preferring `latest_wkid`).
    pub fn effective_wkid(&self) -> Option<u32> {
        self.latest_wkid.or(self.wkid)
    }

    /// Returns true if this spatial reference is Web Mercator
    /// (WKID 102100, 102113, or 3857).
    pub fn is_web_mercator(&self) -> bool {
        matches!(
            self.effective_wkid(),
            Some(102100) | Some(102113) | Some(3857)
        )
    }

    /// Returns true if this spatial reference is WGS84 geographic (WKID 4326).
    pub fn is_geographic(&self) -> bool {
        matches!(self.effective_wkid(), Some(4326))
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Extent {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
    pub spatial_reference: Option<SpatialReference>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LodEntry {
    pub level: u32,
    pub resolution: f64,
    pub scale: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TileInfo {
    pub rows: u32,
    pub cols: u32,
    pub format: Option<String>,
    pub lods: Vec<LodEntry>,
    pub spatial_reference: SpatialReference,
    pub origin: Option<TileOrigin>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TileOrigin {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MapServerMetadata {
    pub tile_info: Option<TileInfo>,
    pub full_extent: Option<Extent>,
    pub initial_extent: Option<Extent>,
    pub copyright_text: Option<String>,
    pub capabilities: Option<String>,
    pub export_tiles_allowed: Option<bool>,
    pub max_export_tiles_count: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageServerMetadata {
    pub tile_info: Option<TileInfo>,
    pub extent: Option<Extent>,
    pub spatial_reference: Option<SpatialReference>,
    pub min_values: Option<Vec<f64>>,
    pub max_values: Option<Vec<f64>>,
    pub band_count: Option<u32>,
    pub capabilities: Option<String>,
    pub copyright_text: Option<String>,
}

impl ImageServerMetadata {
    /// Returns true if the server supports the Tilemap capability.
    pub fn supports_tilemap(&self) -> bool {
        self.capabilities
            .as_deref()
            .map(|c| c.contains("Tilemap"))
            .unwrap_or(false)
    }

    /// Returns the height (elevation) range from min/max values.
    pub fn height_range(&self) -> Option<(f64, f64)> {
        let min = self.min_values.as_ref()?.first().copied()?;
        let max = self.max_values.as_ref()?.first().copied()?;
        Some((min, max))
    }
}

/// Raw tilemap availability response bytes.
#[derive(Debug)]
pub struct TilemapResponse {
    pub data: Vec<u8>,
}

/// A rectangular range of available tiles at a given zoom level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileRange {
    pub start_x: u32,
    pub start_y: u32,
    pub end_x: u32,
    pub end_y: u32,
}

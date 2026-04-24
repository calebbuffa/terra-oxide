//! ArcGIS MapServer client — imagery tile provider.

use super::types::MapServerMetadata;
use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use crate::rest::fetch_json;
use orkester::Task;
use std::sync::Arc;

/// Client for an ArcGIS MapServer REST endpoint.
///
/// Provides metadata, tile URL construction, and export/identify URL builders.
///
/// Auth is handled at the accessor level — no token field here.
///
/// # WKID mapping
/// - 102100 / 102113 / 3857 -> Web Mercator
/// - 4326 -> Geographic
pub struct MapServerClient {
    base_url: String,
    accessor: Arc<dyn AssetAccessor>,
}

impl MapServerClient {
    pub fn new(base_url: impl Into<String>, accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            accessor,
        }
    }

    /// Fetch `{base_url}?f=json` and parse as [`MapServerMetadata`].
    pub fn metadata(&self) -> Task<Result<MapServerMetadata, FetchError>> {
        fetch_json(self, "?f=json")
    }

    /// Returns the pre-cached tile URL for the given level/row/col.
    ///
    /// ArcGIS tile URL scheme: `tile/{level}/{row}/{col}` (note: row before col).
    pub fn tile_url(&self, level: u32, row: u32, col: u32) -> String {
        format!("{}/tile/{level}/{row}/{col}", self.base_url)
    }

    /// Returns an `export` URL for a dynamic map image.
    ///
    /// `bbox` is `[west, south, east, north]` in `wkid` CRS.
    /// `size` is `[width_px, height_px]`.
    pub fn export_url(&self, bbox: [f64; 4], size: [u32; 2], wkid: u32) -> String {
        format!(
            "{}/export?bbox={},{},{},{}&bboxSR={wkid}&size={},{}&format=png32\
             &transparent=true&f=image",
            self.base_url, bbox[0], bbox[1], bbox[2], bbox[3], size[0], size[1],
        )
    }

    /// Returns an `identify` URL for feature picking at a map point.
    ///
    /// `point` is `[x, y]`, `sr` is the spatial reference WKID,
    /// `map_extent` is `[xmin, ymin, xmax, ymax]`, `size` is `[w, h]` in pixels.
    pub fn identify_url(
        &self,
        point: [f64; 2],
        sr: u32,
        map_extent: [f64; 4],
        size: [u32; 2],
    ) -> String {
        format!(
            "{}/identify?f=json&tolerance=2&geometryType=esriGeometryPoint\
             &geometry={},{}&sr={sr}&mapExtent={},{},{},{}&imageDisplay={},{},96",
            self.base_url,
            point[0],
            point[1],
            map_extent[0],
            map_extent[1],
            map_extent[2],
            map_extent[3],
            size[0],
            size[1],
        )
    }
}

impl Client for MapServerClient {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}

impl MapServerClient {
    /// Fetch a raw tile by URL and return its bytes.
    pub fn fetch_tile(&self, level: u32, row: u32, col: u32) -> Task<Result<Vec<u8>, FetchError>> {
        let url = self.tile_url(level, row, col);
        self.accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    Ok(resp.data)
                })
            })
    }
}

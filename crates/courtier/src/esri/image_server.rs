//! ArcGIS ImageServer client — elevation and geoid tile provider.

use super::types::{ImageServerMetadata, TileRange};
use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use crate::rest::fetch_json;
use orkester::Task;
use std::sync::Arc;

/// Client for an ArcGIS ImageServer REST endpoint.
///
/// Used for elevation terrain tiles (LERC/raw encoding) and as the geoid
/// (EGM2008) service for I3S scene height correction.
pub struct ImageServerClient {
    base_url: String,
    accessor: Arc<dyn AssetAccessor>,
}

impl ImageServerClient {
    pub fn new(base_url: impl Into<String>, accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            accessor,
        }
    }

    /// Fetch `{base_url}?f=pjson` and parse as [`ImageServerMetadata`].
    pub fn metadata(&self) -> Task<Result<ImageServerMetadata, FetchError>> {
        fetch_json(self, "?f=pjson")
    }

    /// Returns the tile URL for the given level/row/col.
    pub fn tile_url(&self, level: u32, row: u32, col: u32) -> String {
        format!("{}/tile/{level}/{row}/{col}", self.base_url)
    }

    /// Fetch a raw tile and return its bytes.
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

    /// Fetch tilemap availability for a 128×128 block at the given offset.
    ///
    /// Returns a list of rectangular tile ranges decoded by the flood-fill
    /// algorithm (ported from `ArcGISTiledElevationTerrainProvider.js`).
    pub fn tilemap(
        &self,
        level: u32,
        x_offset: u32,
        y_offset: u32,
    ) -> Task<Result<Vec<TileRange>, FetchError>> {
        const DIM: u32 = 128;
        let url = format!(
            "{}/tilemap/{level}/{y_offset}/{x_offset}/{DIM}/{DIM}",
            self.base_url
        );
        let accessor = Arc::clone(&self.accessor);
        accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .map(move |result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    Ok(compute_availability(
                        x_offset, y_offset, DIM, DIM, &resp.data,
                    ))
                })
            })
    }
}

impl Client for ImageServerClient {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}

// ---------------------------------------------------------------------------
// Tilemap availability — flood-fill range decode
// Ported from ArcGISTiledElevationTerrainProvider.js::computeAvailability
// ---------------------------------------------------------------------------

fn compute_availability(
    x_off: u32,
    y_off: u32,
    width: u32,
    height: u32,
    data: &[u8],
) -> Vec<TileRange> {
    let mut ranges = Vec::new();
    let mut visited = vec![false; (width * height) as usize];

    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if visited[idx] || idx >= data.len() || data[idx] == 0 {
                visited[idx] = true;
                continue;
            }
            // Flood-fill to find the maximal rectangle rooted at (x, y).
            let range = find_range(x, y, width, height, data, &mut visited);
            ranges.push(TileRange {
                start_x: x_off + range.0,
                start_y: y_off + range.1,
                end_x: x_off + range.2,
                end_y: y_off + range.3,
            });
        }
    }

    ranges
}

/// Find the maximal rectangular run of available tiles starting at `(ox, oy)`.
///
/// Returns `(start_x, start_y, end_x, end_y)` in block-local coordinates.
fn find_range(
    ox: u32,
    oy: u32,
    width: u32,
    height: u32,
    data: &[u8],
    visited: &mut [bool],
) -> (u32, u32, u32, u32) {
    // Extend right as far as tiles are available.
    let mut end_x = ox;
    while end_x + 1 < width {
        let next_idx = (oy * width + end_x + 1) as usize;
        if next_idx >= data.len() || data[next_idx] == 0 {
            break;
        }
        end_x += 1;
    }

    // Extend down while the entire row [ox..=end_x] is available.
    let mut end_y = oy;
    'outer: while end_y + 1 < height {
        for x in ox..=end_x {
            let idx = ((end_y + 1) * width + x) as usize;
            if idx >= data.len() || data[idx] == 0 {
                break 'outer;
            }
        }
        end_y += 1;
    }

    // Mark all cells in the found rectangle as visited.
    for y in oy..=end_y {
        for x in ox..=end_x {
            let idx = (y * width + x) as usize;
            if idx < visited.len() {
                visited[idx] = true;
            }
        }
    }

    (ox, oy, end_x, end_y)
}

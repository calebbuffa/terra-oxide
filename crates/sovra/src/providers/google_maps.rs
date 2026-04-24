//! Google Maps Platform Tiles API provider.
//!
//! Uses the Google Map Tiles API v1 with session token management.
//! See: <https://developers.google.com/maps/documentation/tile/create-renderer>

use std::sync::Arc;

use courtier::{AssetAccessor, RequestPriority};
use orkester::{Context, LoadOnce, Task};
use serde::Deserialize;

use super::url_template::compute_tile_rectangle;
use crate::credit::Credit;
use crate::overlay::{
    OverlayProjection, RasterOverlay, RasterOverlayTile, RasterOverlayTileProvider, TileFetchError,
    get_tiles_for_extent,
};

/// Map type for the Google Maps Tiles API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GoogleMapsMapType {
    /// Standard road map.
    Roadmap,
    /// Satellite imagery.
    Satellite,
    /// Terrain view.
    Terrain,
    /// Satellite with road overlay.
    Hybrid,
}

impl GoogleMapsMapType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Roadmap => "roadmap",
            Self::Satellite => "satellite",
            Self::Terrain => "terrain",
            // Hybrid uses satellite imagery with a road layer overlay.
            Self::Hybrid => "satellite",
        }
    }
}

/// Options for the Google Maps Tiles API provider.
#[derive(Clone, Debug)]
pub struct GoogleMapsTilesOptions {
    /// Google Maps Platform API key.
    pub api_key: String,
    /// Map type to request.
    pub map_type: GoogleMapsMapType,
    /// IETF language tag for labels (e.g. `"en-US"`).
    pub language: Option<String>,
    /// CLDR region code (e.g. `"US"`).
    pub region: Option<String>,
    /// Request high-DPI tiles (512 px instead of 256 px).
    pub high_dpi: bool,
}

/// Session state for the Google Maps Tiles API.
#[derive(Clone)]
struct GoogleMapsSession {
    session_token: String,
    /// Expiry timestamp as Unix seconds.
    expiry: u64,
}

/// A raster overlay backed by the Google Maps Platform Tiles API.
///
/// Manages a session token automatically: the first `get_tile` call
/// (or any call after session expiry) triggers an async `createSession`
/// POST before fetching the tile.
pub struct GoogleMapsTilesProvider {
    options: GoogleMapsTilesOptions,
    accessor: Arc<dyn AssetAccessor>,
    /// Stored context for chaining the session-creation task with tile fetches.
    ctx: Context,
    /// Deduplicating loader: at most one `createSession` POST in flight at a time.
    session_loader: Arc<LoadOnce<(), GoogleMapsSession>>,
    /// Cache of the most recently established session (avoids re-POSTing while
    /// the session is still valid).
    session_cache: Arc<std::sync::Mutex<Option<GoogleMapsSession>>>,
}

impl GoogleMapsTilesProvider {
    pub fn new(
        options: GoogleMapsTilesOptions,
        accessor: Arc<dyn AssetAccessor>,
        ctx: Context,
    ) -> Arc<Self> {
        Arc::new(Self {
            options,
            accessor,
            ctx,
            session_loader: Arc::new(LoadOnce::new()),
            session_cache: Arc::new(std::sync::Mutex::new(None)),
        })
    }

    /// Build the JSON body for the `createSession` endpoint.
    fn build_session_body(&self) -> String {
        let map_type = self.options.map_type.as_str();
        let is_hybrid = self.options.map_type == GoogleMapsMapType::Hybrid;
        let mut obj = format!(r#"{{"mapType":"{map_type}","language":""#);
        if let Some(ref lang) = self.options.language {
            obj.push_str(lang);
        } else {
            obj.push_str("en-US");
        }
        obj.push_str(r#"","region":""#);
        if let Some(ref region) = self.options.region {
            obj.push_str(region);
        } else {
            obj.push_str("US");
        }
        obj.push_str(r#"","imageFormat":"jpeg""#);
        // P06: Hybrid requires layerRoadmap overlay; all types require explicit overlay flag.
        if is_hybrid {
            obj.push_str(r#","layerTypes":["layerRoadmap"],"overlay":true"#);
        } else {
            obj.push_str(r#","overlay":false"#);
        }
        // P17: scale factor for high-DPI tiles.
        if self.options.high_dpi {
            obj.push_str(r#","scale":"scaleFactor2x""#);
        } else {
            obj.push_str(r#","scale":"scaleFactor1x""#);
        }
        obj.push('}');
        obj
    }

    /// Return a [`Handle<String>`] that resolves to the current session token.
    ///
    /// * Fast path: if `session_cache` holds a session with more than 60 s
    ///   remaining, return an already-resolved handle.
    /// * Slow path: delegate to `session_loader` so that at most one
    ///   `createSession` POST is in flight at any time.  All concurrent callers
    ///   share the same underlying handle and receive the token once the POST
    ///   completes.
    fn ensure_session(&self) -> orkester::Handle<String> {
        // Fast path: valid cached session.
        {
            let guard = self.session_cache.lock().unwrap();
            if let Some(s) = guard.as_ref() {
                if s.expiry > now_secs() + 60 {
                    return orkester::resolved(s.session_token.clone()).share();
                }
            }
        }

        // Slow path: deduplicated POST via LoadOnce.
        let accessor = Arc::clone(&self.accessor);
        let api_key = self.options.api_key.clone();
        let body = self.build_session_body();
        let session_cache = Arc::clone(&self.session_cache);

        let session_handle = self.session_loader.get_or_load((), move |_| {
            let body_bytes = body.into_bytes();
            let session_url = format!(
                "https://tile.googleapis.com/v1/createSession?key={}",
                api_key
            );
            let headers = vec![("Content-Type".to_string(), "application/json".to_string())];

            // Create a (resolver, task) pair so the HTTP response callback can
            // asynchronously resolve the GoogleMapsSession task.
            let (resolver, session_task) = orkester::pair::<GoogleMapsSession>();

            let _ = accessor
                .request(
                    "POST",
                    &session_url,
                    &headers,
                    Some(&body_bytes),
                    RequestPriority::HIGH,
                    None,
                )
                .map(move |result| match result {
                    Ok(resp) if resp.status >= 200 && resp.status < 300 => {
                        match parse_new_session(&resp.data) {
                            Ok(session) => resolver.resolve(session),
                            Err(e) => {
                                log::warn!("Google Maps createSession parse error: {e}");
                                resolver.reject(e);
                            }
                        }
                    }
                    Ok(resp) => {
                        let msg = format!("Google Maps createSession HTTP {}", resp.status);
                        log::warn!("{msg}");
                        resolver.reject(msg);
                    }
                    Err(e) => {
                        log::warn!("Google Maps createSession fetch failed: {e}");
                        resolver.reject(e.to_string());
                    }
                });

            session_task
        });

        // Chain: store the new session in the cache and return the token.
        session_handle
            .map(move |session| {
                let token = session.session_token.clone();
                *session_cache.lock().unwrap() = Some(session);
                token
            })
            .share()
    }
}

impl RasterOverlayTileProvider for GoogleMapsTilesProvider {
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>> {
        let handle = self.ensure_session();
        let accessor = Arc::clone(&self.accessor);
        let api_key = self.options.api_key.clone();
        let ctx = self.ctx.clone();

        handle
            .then(&self.ctx, move |token| {
                let url = format!(
                    "https://tile.googleapis.com/v1/2dtiles/{}/{}/{}?session={}&key={}",
                    level, x, y, token, api_key
                );
                let bounds = web_mercator_bounds();
                let rect =
                    compute_tile_rectangle(x, y, level, &bounds, OverlayProjection::WebMercator);
                super::fetch_and_decode_tile(
                    &accessor,
                    ctx,
                    &url,
                    &[],
                    rect,
                    OverlayProjection::WebMercator,
                )
            })
            .or_else(|e| -> Result<RasterOverlayTile, TileFetchError> {
                Err(TileFetchError::Decode(e.to_string().into()))
            })
    }

    fn bounds(&self) -> terra::GlobeRectangle {
        web_mercator_bounds()
    }

    fn maximum_level(&self) -> u32 {
        20
    }

    fn minimum_level(&self) -> u32 {
        0
    }

    fn projection(&self) -> OverlayProjection {
        OverlayProjection::WebMercator
    }

    fn credits(&self) -> Vec<Credit> {
        vec![Credit::new("© Google")]
    }

    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)> {
        get_tiles_for_extent(self, extent, target_screen_pixels)
    }
}

/// User-facing overlay source that implements [`RasterOverlay`].
///
/// Add this to an [`OverlayEngine`](crate::OverlayEngine) to stream Google
/// Maps Platform tiles.
pub struct GoogleMapsRasterOverlay {
    options: GoogleMapsTilesOptions,
}

impl GoogleMapsRasterOverlay {
    pub fn new(options: GoogleMapsTilesOptions) -> Self {
        Self { options }
    }
}

impl RasterOverlay for GoogleMapsRasterOverlay {
    fn create_tile_provider(
        &self,
        context: &Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>> {
        let provider = GoogleMapsTilesProvider::new(
            self.options.clone(),
            Arc::clone(accessor),
            context.clone(),
        );
        orkester::resolved(provider as Arc<dyn RasterOverlayTileProvider>)
    }
}

// helpers

fn web_mercator_bounds() -> terra::GlobeRectangle {
    terra::GlobeRectangle::from_degrees(-180.0, -85.051_128_78, 180.0, 85.051_128_78)
}

#[derive(Deserialize)]
struct SessionResponse {
    session: String,
    #[serde(rename = "expiryTime")]
    expiry_time: String,
}

fn parse_new_session(data: &[u8]) -> Result<GoogleMapsSession, String> {
    let resp: SessionResponse = serde_json::from_slice(data).map_err(|e| e.to_string())?;
    let expiry = parse_rfc3339_to_unix(&resp.expiry_time).unwrap_or_else(|| now_secs() + 30 * 60);
    Ok(GoogleMapsSession {
        session_token: resp.session,
        expiry,
    })
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Parse an RFC 3339 / ISO 8601 timestamp (`YYYY-MM-DDTHH:MM:SSZ`) to a Unix
/// timestamp (seconds since 1970-01-01T00:00:00Z).
///
/// Returns `None` on any parse error so callers can supply a safe fallback.
fn parse_rfc3339_to_unix(s: &str) -> Option<u64> {
    let s = if s.ends_with('Z') {
        &s[..s.len() - 1]
    } else {
        s
    };
    let (date, time) = s.split_once('T')?;

    let mut d = date.split('-');
    let year: u64 = d.next()?.parse().ok()?;
    let month: u64 = d.next()?.parse().ok()?;
    let day: u64 = d.next()?.parse().ok()?;

    let mut t = time.split(':');
    let hour: u64 = t.next()?.parse().ok()?;
    let min: u64 = t.next()?.parse().ok()?;
    // Accept whole or fractional seconds.
    let sec: u64 = t.next()?.split('.').next()?.parse().ok()?;

    let days = days_since_epoch(year, month, day)?;
    Some(days * 86400 + hour * 3600 + min * 60 + sec)
}

fn is_leap_year(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0)
}

fn days_since_epoch(year: u64, month: u64, day: u64) -> Option<u64> {
    if year < 1970 || month < 1 || month > 12 || day < 1 {
        return None;
    }
    let leap = is_leap_year(year);
    let days_per_month: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if day > days_per_month[(month - 1) as usize] {
        return None;
    }
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap_year(y) { 366 } else { 365 };
    }
    for m in 1..month {
        days += days_per_month[(m - 1) as usize];
    }
    days += day - 1;
    Some(days)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rfc3339_known_timestamp() {
        // 2025-01-01T12:00:00Z -> 2025-01-01 is 20089 days after epoch
        // 20089 * 86400 + 12*3600 = 1735,732,800
        assert_eq!(
            parse_rfc3339_to_unix("2025-01-01T12:00:00Z"),
            Some(1_735_732_800)
        );
    }

    #[test]
    fn parse_rfc3339_invalid_returns_none() {
        assert_eq!(parse_rfc3339_to_unix("not-a-date"), None);
    }
}

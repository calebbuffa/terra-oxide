//! Cesium Ion REST API types.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Authentication mode reported by the Ion server's `/appData` endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthenticationMode {
    /// Standard OAuth2 with ion.cesium.com accounts.
    CesiumIon,
    /// OAuth2 with an external SAML identity provider (self-hosted ion).
    Saml,
    /// Self-hosted ion without authentication (single-user mode).
    /// In this mode `/me` and `/tokens` are unavailable.
    SingleUser,
}

/// Data retrieved from `/appData` — describes the Ion server configuration.
#[derive(Debug, Clone)]
pub struct ApplicationData {
    /// How the Ion server authenticates users.
    pub authentication_mode: AuthenticationMode,
    /// The type of file storage used by this server (e.g. `"S3"`, `"FILE_SYSTEM"`).
    pub data_store_type: String,
    /// Attribution HTML for this Ion server.
    pub attribution: String,
}

impl ApplicationData {
    /// Returns `true` if this server requires OAuth2 authentication.
    pub fn needs_oauth_authentication(&self) -> bool {
        self.authentication_mode != AuthenticationMode::SingleUser
    }
}

/// A JWT-based login token obtained from the OAuth2 authorization flow.
///
/// These tokens expire (typically after 1 hour) and can be refreshed using a
/// refresh token.  Distinct from a Cesium Ion [`Token`], which is valid until
/// explicitly revoked by the user and is scoped to specific assets/endpoints.
#[derive(Debug, Clone)]
pub struct LoginToken {
    token: String,
    expiration: Option<i64>, // Unix timestamp seconds
}

impl LoginToken {
    /// Parse a JWT string, extracting the `exp` claim from the payload.
    pub fn parse(token_string: impl Into<String>) -> Self {
        let token = token_string.into();
        let expiration = parse_jwt_expiry(&token);
        Self { token, expiration }
    }

    /// Create a token that is treated as never-expiring (no `exp` check).
    pub fn never_expires(token_string: impl Into<String>) -> Self {
        Self {
            token: token_string.into(),
            expiration: None,
        }
    }

    /// The raw token string.
    pub fn token(&self) -> &str {
        &self.token
    }

    /// Returns `true` if the token has not yet expired.
    pub fn is_valid(&self) -> bool {
        match self.expiration {
            None => true,
            Some(exp) => {
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
                now < exp
            }
        }
    }

    /// The expiration time as a Unix timestamp, if present.
    pub fn expiration(&self) -> Option<i64> {
        self.expiration
    }
}

fn parse_jwt_expiry(token: &str) -> Option<i64> {
    use base64::Engine as _;
    let payload_b64 = token.split('.').nth(1)?;
    // base64url, no padding
    let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload_b64)
        .ok()?;
    let json: serde_json::Value = serde_json::from_slice(&payload).ok()?;
    json.get("exp")?.as_i64()
}

/// A Cesium Ion asset.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Asset {
    pub id: u64,
    pub name: String,
    #[serde(rename = "type")]
    pub asset_type: String,
    pub bytes: Option<i64>,
    pub status: Option<String>,
    pub attribution: Option<String>,
    pub description: Option<String>,
    pub date_added: Option<String>,
    pub percent_complete: Option<i8>,
}

/// Resolved streaming endpoint for a Cesium Ion asset.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssetEndpoint {
    pub url: String,
    pub access_token: String,
    pub attribution: Option<String>,
}

/// Pagination wrapper returned by Ion list endpoints.
#[derive(Debug, Clone, Deserialize)]
pub struct PagedIonResponse<T> {
    pub items: Vec<T>,
}

/// Options for paginated list requests.
#[derive(Debug, Default, Clone, Serialize)]
pub struct ListOptions {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub search: Option<String>,
}

/// A Cesium Ion API access token (distinct from a [`LoginToken`]).
///
/// These tokens are valid until revoked and are scoped to specific assets
/// and/or origin URLs.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Token {
    pub id: String,
    pub name: String,
    pub token: String,
    pub date_added: Option<String>,
    pub date_modified: Option<String>,
    pub date_last_used: Option<String>,
    pub is_default: bool,
    /// `None` means the token allows access to all assets.
    pub asset_ids: Option<Vec<i64>>,
    /// `None` means the token can be accessed from any URL.
    pub allowed_urls: Option<Vec<String>>,
    pub scopes: Vec<String>,
}

/// A page of [`Token`] results, with optional pagination cursors.
#[derive(Debug, Clone)]
pub struct TokenPage {
    pub items: Vec<Token>,
    /// URL for the next page of results (pass to [`Connection::tokens_page`]).
    pub next_page_url: Option<String>,
    /// URL for the previous page of results.
    pub previous_page_url: Option<String>,
}

/// Sort direction for token list requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Options for [`Connection::tokens`].
#[derive(Debug, Default, Clone)]
pub struct ListTokensOptions {
    pub limit: Option<i32>,
    pub page: Option<i32>,
    pub search: Option<String>,
    /// Valid values: `"NAME"`, `"LAST_USED"`.
    pub sort_by: Option<String>,
    pub sort_order: Option<SortOrder>,
}

/// A valueless response, for use with operations that return no body on success.
#[derive(Debug, Clone)]
pub struct NoValue;

/// Storage quota information for a Cesium Ion account.
#[derive(Debug, Clone, Default)]
pub struct ProfileStorage {
    /// Bytes currently in use.
    pub used: i64,
    /// Bytes available for additional uploads.
    pub available: i64,
    /// Total byte quota for this account.
    pub total: i64,
}

/// Profile information for the authenticated Cesium Ion user.
#[derive(Debug, Clone)]
pub struct Profile {
    pub id: i64,
    /// OAuth2 scopes granted to the current token.
    pub scopes: Vec<String>,
    pub username: String,
    pub email: String,
    pub email_verified: bool,
    /// URL to the user's avatar image.
    pub avatar: String,
    pub storage: ProfileStorage,
}

/// The default asset IDs for imagery, terrain, and buildings.
#[derive(Debug, Clone, Default)]
pub struct DefaultAssets {
    pub imagery: i64,
    pub terrain: i64,
    pub buildings: i64,
}

/// A raster overlay that can be combined with a quick-add asset.
#[derive(Debug, Clone)]
pub struct QuickAddRasterOverlay {
    pub name: String,
    pub asset_id: i64,
    /// `true` if the authenticated user is subscribed to this asset.
    pub subscribed: bool,
}

/// A curated asset recommended by Cesium Ion.
#[derive(Debug, Clone)]
pub struct QuickAddAsset {
    pub name: String,
    /// Name of the primary (non-imagery) asset.
    pub object_name: String,
    pub description: String,
    pub asset_id: i64,
    /// Asset type, e.g. `"3DTILES"`, `"TERRAIN"`, `"IMAGERY"`.
    pub asset_type: String,
    /// `true` if the authenticated user is subscribed.
    pub subscribed: bool,
    pub raster_overlays: Vec<QuickAddRasterOverlay>,
}

/// Response from `/v1/defaults` — default and recommended Ion assets.
#[derive(Debug, Clone)]
pub struct Defaults {
    pub default_assets: DefaultAssets,
    pub quick_add_assets: Vec<QuickAddAsset>,
}

/// Which geocoder provider to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeocoderProviderType {
    /// Google geocoder (for use with Google data).
    Google,
    /// Bing geocoder (for use with Bing data).
    Bing,
    /// The server's default geocoder.
    Default,
}

/// Type of geocoder request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeocoderRequestType {
    /// Full search from a complete query.
    Search,
    /// Quick search from partial input (e.g. while the user is typing).
    Autocomplete,
}

/// The geographic destination for a geocoder result.
#[derive(Debug, Clone)]
pub enum GeocoderDestination {
    /// A bounding rectangle (west, south, east, north in radians).
    Rectangle {
        west: f64,
        south: f64,
        east: f64,
        north: f64,
    },
    /// A single point (longitude, latitude in radians).
    Point { longitude: f64, latitude: f64 },
}

impl GeocoderDestination {
    /// Returns the west/south/east/north bounding rectangle.
    /// For a point, returns a zero-area rectangle at that location.
    pub fn to_rectangle(&self) -> (f64, f64, f64, f64) {
        match self {
            GeocoderDestination::Rectangle {
                west,
                south,
                east,
                north,
            } => (*west, *south, *east, *north),
            GeocoderDestination::Point {
                longitude,
                latitude,
            } => (*longitude, *latitude, *longitude, *latitude),
        }
    }

    /// Returns the center point as (longitude, latitude) in radians.
    pub fn center(&self) -> (f64, f64) {
        match self {
            GeocoderDestination::Rectangle {
                west,
                south,
                east,
                north,
            } => ((*west + *east) * 0.5, (*south + *north) * 0.5),
            GeocoderDestination::Point {
                longitude,
                latitude,
            } => (*longitude, *latitude),
        }
    }
}

/// A single feature (location or region) from a geocoder result.
#[derive(Debug, Clone)]
pub struct GeocoderFeature {
    /// User-friendly display name.
    pub display_name: String,
    /// The geographic extent or point for this result.
    pub destination: GeocoderDestination,
}

/// Attribution information for a geocoder result.
#[derive(Debug, Clone)]
pub struct GeocoderAttribution {
    /// HTML string with attribution text.
    pub html: String,
    /// If `true`, should be shown prominently; otherwise can be in a popover.
    pub show_on_screen: bool,
}

/// Response from the Ion geocoder API.
#[derive(Debug, Clone)]
pub struct GeocoderResult {
    pub attributions: Vec<GeocoderAttribution>,
    pub features: Vec<GeocoderFeature>,
}

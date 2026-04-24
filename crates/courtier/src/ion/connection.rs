//! Cesium Ion API connection.

use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use orkester::Task;
use serde_json::Value;

use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use crate::rest::parse_json;

use super::types::{
    ApplicationData, Asset, AssetEndpoint, AuthenticationMode, DefaultAssets, Defaults,
    GeocoderAttribution, GeocoderDestination, GeocoderFeature, GeocoderProviderType,
    GeocoderRequestType, GeocoderResult, ListOptions, ListTokensOptions, LoginToken, NoValue,
    PagedIonResponse, Profile, ProfileStorage, QuickAddAsset, QuickAddRasterOverlay, SortOrder,
    Token, TokenPage,
};

const DEFAULT_API_URL: &str = "https://api.cesium.com";

struct LoginDetails {
    access_token: LoginToken,
    refresh_token: String,
}

enum AuthMode {
    /// Auth is handled by the accessor (e.g. wrapped with `AuthenticatedAccessor`).
    External,
    /// Connection holds a static Bearer token (non-expiring API tokens).
    Static(String),
    /// Connection holds an OAuth2 login token and can refresh it.
    Login {
        details: Arc<Mutex<LoginDetails>>,
        api_url: String,
        client_id: i64,
        redirect_path: String,
    },
    /// Ion self-hosted SingleUser mode — no authentication required.
    SingleUser,
}

/// A connection to the Cesium Ion REST API.
///
/// # Authentication
///
/// Three authentication modes are supported:
///
/// - **External** (`Connection::new`): auth is already baked into the accessor
///   (e.g. via `AuthenticatedAccessor::new(http, BearerTokenAuth::new(token))`).
/// - **Static token** (`Connection::with_token`): the connection injects a
///   Bearer token on every request.  Use this for non-expiring API tokens.
/// - **OAuth2 login** (`Connection::with_login` or `Connection::authorize`):
///   the connection holds a JWT access token, checks expiry before each request,
///   and uses a refresh token to obtain a new access token when needed.
pub struct Connection {
    pub(crate) base_url: String,
    pub(crate) accessor: Arc<dyn AssetAccessor>,
    auth_mode: AuthMode,
    pub app_data: Option<ApplicationData>,
}

impl Connection {
    /// Create a connection where auth is delegated to the accessor.
    pub fn new(accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: DEFAULT_API_URL.to_owned(),
            accessor,
            auth_mode: AuthMode::External,
            app_data: None,
        }
    }

    /// Returns the accessor backing this connection.
    pub fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }

    /// Create a connection with a static Bearer token.
    ///
    /// Use for non-expiring Cesium Ion API tokens.  The token is injected as
    /// `Authorization: Bearer {token}` on every request.
    pub fn with_token(token: impl Into<String>, accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: DEFAULT_API_URL.to_owned(),
            accessor,
            auth_mode: AuthMode::Static(token.into()),
            app_data: None,
        }
    }

    /// Create a connection using a login token with optional refresh capability.
    ///
    /// Pass an empty `refresh_token` for tokens that cannot be refreshed.
    pub fn with_login(
        access_token: LoginToken,
        refresh_token: impl Into<String>,
        client_id: i64,
        redirect_path: impl Into<String>,
        app_data: ApplicationData,
        accessor: Arc<dyn AssetAccessor>,
        api_url: impl Into<String>,
    ) -> Self {
        let details = Arc::new(Mutex::new(LoginDetails {
            access_token,
            refresh_token: refresh_token.into(),
        }));
        let mut api = api_url.into();
        if api.is_empty() {
            api = DEFAULT_API_URL.to_owned();
        }
        let base_url = api.trim_end_matches('/').to_owned();
        Self {
            base_url: base_url.clone(),
            accessor,
            auth_mode: AuthMode::Login {
                details,
                api_url: base_url,
                client_id,
                redirect_path: redirect_path.into(),
            },
            app_data: Some(app_data),
        }
    }

    /// Create a connection for Ion self-hosted SingleUser mode (no auth).
    pub fn single_user(accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: DEFAULT_API_URL.to_owned(),
            accessor,
            auth_mode: AuthMode::SingleUser,
            app_data: None,
        }
    }

    /// Override the Ion API base URL (for private deployments).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        let url = url.into().trim_end_matches('/').to_owned();
        if let AuthMode::Login { api_url, .. } = &mut self.auth_mode {
            *api_url = url.clone();
        }
        self.base_url = url;
        self
    }

    /// Fetch application data from `/appData` without requiring a connection.
    ///
    /// Use this to detect whether the Ion server requires OAuth authentication
    /// and to obtain the `ApplicationData` needed for other constructors.
    pub fn app_data_request(
        accessor: &Arc<dyn AssetAccessor>,
        api_url: &str,
    ) -> Task<Result<ApplicationData, FetchError>> {
        let url = format!("{}/appData", api_url.trim_end_matches('/'));
        let accessor = Arc::clone(accessor);
        accessor
            .get(
                &url,
                &[("Accept".to_owned(), "application/json".to_owned())],
                RequestPriority::NORMAL,
                None,
            )
            .map(|result| {
                let resp = result?;
                resp.check_status()?;
                let json: Value = serde_json::from_slice(&resp.data)
                    .map_err(|e| FetchError::Json(e.to_string()))?;
                let mode_str = json_str(&json, "applicationMode").unwrap_or("cesium-ion");
                let authentication_mode = match mode_str {
                    "single-user" => AuthenticationMode::SingleUser,
                    "saml" => AuthenticationMode::Saml,
                    _ => AuthenticationMode::CesiumIon,
                };
                Ok(ApplicationData {
                    authentication_mode,
                    data_store_type: json_str(&json, "dataStoreType").unwrap_or("S3").to_owned(),
                    attribution: json_str(&json, "attribution").unwrap_or("").to_owned(),
                })
            })
    }

    /// Attempt to discover the Ion REST API URL for a given Ion instance URL.
    ///
    /// Self-hosted instances may serve a `config.json` at their root that
    /// contains an `apiHostname` key.  Returns `None` if no config is found
    /// (which is the case for `ion.cesium.com`).
    pub fn get_api_url(
        accessor: &Arc<dyn AssetAccessor>,
        ion_url: &str,
    ) -> Task<Result<Option<String>, FetchError>> {
        let config_url = format!("{}/config.json", ion_url.trim_end_matches('/'));
        let ion_url = ion_url.to_owned();
        let accessor = Arc::clone(accessor);
        accessor
            .get(&config_url, &[], RequestPriority::NORMAL, None)
            .map(move |result| match result {
                Ok(resp) if resp.status >= 200 && resp.status < 300 => {
                    if let Ok(json) = serde_json::from_slice::<Value>(&resp.data) {
                        if let Some(hostname) = json.get("apiHostname").and_then(|v| v.as_str()) {
                            return Ok(Some(format!("https://{hostname}/")));
                        }
                    }
                    Ok(derive_api_url(&ion_url))
                }
                _ => Ok(derive_api_url(&ion_url)),
            })
    }

    /// Extract the token ID (`jti` claim) from a JWT token string.
    ///
    /// Returns `None` if the JWT cannot be parsed or has no `jti` claim.
    pub fn get_id_from_token(token_str: &str) -> Option<String> {
        use base64::Engine as _;
        let payload_b64 = token_str.split('.').nth(1)?;
        let payload = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload_b64)
            .ok()?;
        let json: Value = serde_json::from_slice(&payload).ok()?;
        json.get("jti")?.as_str().map(str::to_owned)
    }

    /// Authorize access to Cesium Ion on behalf of a user via OAuth2 PKCE.
    ///
    /// This opens a browser window to the Ion authorization page, listens on a
    /// local HTTP port for the redirect, exchanges the authorization code for
    /// tokens, and returns a ready-to-use [`Connection`].
    ///
    /// # Parameters
    /// - `accessor` — raw HTTP accessor (should **not** have auth baked in)
    /// - `friendly_name` — shown in the browser tab after authorization
    /// - `client_id` — OAuth2 client ID registered with Cesium Ion
    /// - `redirect_path` — path component only (e.g. `"callback"`); must match
    ///   the registered redirect URI without scheme/host/port
    /// - `scopes` — list of OAuth2 scopes to request
    /// - `open_url` — callback that opens the given URL in the user's browser
    /// - `app_data` — obtained from [`Connection::app_data_request`]
    /// - `ion_api_url` — base URL of the Ion REST API (usually `https://api.cesium.com`)
    /// - `ion_authorize_url` — OAuth2 authorization URL (usually `https://ion.cesium.com/oauth`)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn authorize(
        accessor: Arc<dyn AssetAccessor>,
        friendly_name: impl Into<String>,
        client_id: i64,
        redirect_path: impl Into<String>,
        scopes: &[impl AsRef<str>],
        open_url: impl Fn(String) + Send + 'static,
        app_data: ApplicationData,
        ion_api_url: impl Into<String>,
        ion_authorize_url: impl Into<String>,
    ) -> Task<Result<Connection, FetchError>> {
        use std::net::TcpListener;

        let friendly_name = friendly_name.into();
        let redirect_path = redirect_path.into();
        let ion_api_url = ion_api_url.into().trim_end_matches('/').to_owned();
        let ion_authorize_url = ion_authorize_url.into();
        let scopes_str: Vec<String> = scopes.iter().map(|s| s.as_ref().to_owned()).collect();

        let (verifier, challenge) = match generate_pkce() {
            Ok(p) => p,
            Err(e) => return orkester::resolved(Err(FetchError::Network(e))),
        };

        // Bind to a random localhost port.
        let listener = match TcpListener::bind("127.0.0.1:0") {
            Ok(l) => l,
            Err(e) => return orkester::resolved(Err(FetchError::Io(e))),
        };
        let port = match listener.local_addr() {
            Ok(addr) => addr.port(),
            Err(e) => return orkester::resolved(Err(FetchError::Io(e))),
        };
        let redirect_uri = format!("http://127.0.0.1:{port}/{redirect_path}");

        // Build the authorization URL.
        let auth_url = {
            let scopes_joined = scopes_str.join(" ");
            format!(
                "{}?response_type=code&client_id={}&redirect_uri={}&code_challenge={}&code_challenge_method=S256&scope={}",
                ion_authorize_url,
                client_id,
                url_encode(&redirect_uri),
                url_encode(&challenge),
                url_encode(&scopes_joined),
            )
        };

        // Open the browser.
        open_url(auth_url);

        // Spawn a thread to accept the redirect and exchange the code.
        let (resolver, task) = orkester::pair::<Result<Connection, FetchError>>();
        let token_url = format!("{ion_api_url}/oauth/token");
        let redirect_uri_clone = redirect_uri.clone();
        let verifier_clone = verifier.clone();
        let api_url_clone = ion_api_url.clone();
        let redirect_path_clone = redirect_path.clone();

        std::thread::spawn(move || {
            // Accept one connection from the browser.
            let (mut stream, _) = match listener.accept() {
                Ok(s) => s,
                Err(e) => {
                    resolver.resolve(Err(FetchError::Io(e)));
                    return;
                }
            };

            // Read the request line to get the auth code.
            let mut buf = [0u8; 8192];
            let n = stream.read(&mut buf).unwrap_or(0);
            let request_text = std::str::from_utf8(&buf[..n]).unwrap_or("");
            let code = parse_auth_code_from_request(request_text);

            // Send a simple success page back to the browser.
            let body = format!(
                "Authorization complete. You may close this tab and return to {friendly_name}.",
            );
            let _ = stream.write_all(
                format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            );
            drop(stream);

            let code = match code {
                Some(c) => c,
                None => {
                    resolver.resolve(Err(FetchError::Network(
                        "no authorization code in redirect request".into(),
                    )));
                    return;
                }
            };

            // Exchange code for tokens via POST.
            let form_body = format!(
                "grant_type=authorization_code&code={}&redirect_uri={}&client_id={}&code_verifier={}",
                url_encode(&code),
                url_encode(&redirect_uri_clone),
                url_encode(&client_id.to_string()),
                url_encode(&verifier_clone),
            );
            let headers = vec![
                (
                    "Content-Type".to_owned(),
                    "application/x-www-form-urlencoded".to_owned(),
                ),
                ("Accept".to_owned(), "application/json".to_owned()),
            ];

            let task_result = accessor
                .request(
                    "POST",
                    &token_url,
                    &headers,
                    Some(form_body.as_bytes()),
                    RequestPriority::NORMAL,
                    None,
                )
                .block();

            let result = (|| {
                let resp = task_result.map_err(|e| FetchError::Network(e.to_string()))??;
                resp.check_status()?;
                let json: Value = serde_json::from_slice(&resp.data)
                    .map_err(|e| FetchError::Json(e.to_string()))?;
                let access_str = json["access_token"]
                    .as_str()
                    .ok_or_else(|| FetchError::Json("missing access_token".into()))?;
                let refresh_str = json
                    .get("refresh_token")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                let conn = Connection::with_login(
                    LoginToken::parse(access_str),
                    refresh_str,
                    client_id,
                    redirect_path_clone,
                    app_data,
                    accessor,
                    api_url_clone,
                );
                Ok(conn)
            })();

            resolver.resolve(result);
        });

        task
    }

    /// Returns the `Authorization: Bearer …` header value (or empty string for
    /// External/SingleUser auth modes where the accessor handles auth).
    ///
    /// For `Login` mode with an expired token, performs an OAuth2 token refresh.
    fn ensure_valid_token(&self) -> Task<Result<String, FetchError>> {
        match &self.auth_mode {
            AuthMode::External => orkester::resolved(Ok(String::new())),
            AuthMode::SingleUser => orkester::resolved(Ok(String::new())),
            AuthMode::Static(token) => orkester::resolved(Ok(format!("Bearer {token}"))),
            AuthMode::Login {
                details,
                api_url,
                client_id,
                redirect_path,
            } => {
                // Fast path: token is still valid.
                {
                    let guard = details.lock().unwrap();
                    if guard.access_token.is_valid() {
                        let bearer = format!("Bearer {}", guard.access_token.token());
                        return orkester::resolved(Ok(bearer));
                    }
                    if guard.refresh_token.is_empty() {
                        return orkester::resolved(Err(FetchError::Network(
                            "Cesium Ion access token expired; no refresh token available".into(),
                        )));
                    }
                }

                // Token expired — capture state without holding the lock during I/O.
                let refresh_token = details.lock().unwrap().refresh_token.clone();
                let token_url = format!("{}/oauth/token", api_url.trim_end_matches('/'));
                let client_id_str = client_id.to_string();
                let redirect_uri = redirect_path.clone();
                let details_arc = Arc::clone(details);

                let form_body = format!(
                    "grant_type=refresh_token&refresh_token={}&client_id={}&redirect_uri={}",
                    url_encode(&refresh_token),
                    url_encode(&client_id_str),
                    url_encode(&redirect_uri),
                );
                let headers = vec![
                    (
                        "Content-Type".to_owned(),
                        "application/x-www-form-urlencoded".to_owned(),
                    ),
                    ("Accept".to_owned(), "application/json".to_owned()),
                ];

                self.accessor
                    .request(
                        "POST",
                        &token_url,
                        &headers,
                        Some(form_body.as_bytes()),
                        RequestPriority::NORMAL,
                        None,
                    )
                    .map(move |result| {
                        let resp = result?;
                        resp.check_status()?;
                        let json: Value = serde_json::from_slice(&resp.data)
                            .map_err(|e| FetchError::Json(e.to_string()))?;
                        let access_str = json["access_token"].as_str().ok_or_else(|| {
                            FetchError::Json("missing access_token in refresh response".into())
                        })?;
                        let new_refresh = json
                            .get("refresh_token")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        let login_token = LoginToken::parse(access_str);
                        let bearer = format!("Bearer {}", login_token.token());
                        let mut guard = details_arc.lock().unwrap();
                        guard.access_token = login_token;
                        guard.refresh_token = new_refresh;
                        Ok(bearer)
                    })
            }
        }
    }

    /// Authenticated GET, returns parsed JSON.
    fn get_json<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        url: String,
    ) -> Task<Result<T, FetchError>> {
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<T, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let headers = json_headers_with_auth(&bearer);
                        accessor
                            .get(&url, &headers, RequestPriority::NORMAL, None)
                            .map(|r| {
                                r.and_then(|resp| {
                                    resp.check_status()?;
                                    parse_json(resp)
                                })
                            })
                    }
                }
            })
    }

    /// Authenticated POST with a JSON body, returns parsed JSON.
    fn post_json<T: serde::de::DeserializeOwned + Send + 'static>(
        &self,
        url: String,
        body: Vec<u8>,
    ) -> Task<Result<T, FetchError>> {
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<T, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let mut headers = vec![
                            ("Content-Type".to_owned(), "application/json".to_owned()),
                            ("Accept".to_owned(), "application/json".to_owned()),
                        ];
                        if !bearer.is_empty() {
                            headers.push(("Authorization".to_owned(), bearer));
                        }
                        accessor
                            .request(
                                "POST",
                                &url,
                                &headers,
                                Some(&body),
                                RequestPriority::NORMAL,
                                None,
                            )
                            .map(|r| {
                                r.and_then(|resp| {
                                    resp.check_status()?;
                                    parse_json(resp)
                                })
                            })
                    }
                }
            })
    }

    /// Authenticated PATCH with a JSON body, returns `()` on success.
    fn patch_no_body(&self, url: String, body: Vec<u8>) -> Task<Result<NoValue, FetchError>> {
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<NoValue, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let mut headers = vec![
                            ("Content-Type".to_owned(), "application/json".to_owned()),
                            ("Accept".to_owned(), "application/json".to_owned()),
                        ];
                        if !bearer.is_empty() {
                            headers.push(("Authorization".to_owned(), bearer));
                        }
                        accessor
                            .request(
                                "PATCH",
                                &url,
                                &headers,
                                Some(&body),
                                RequestPriority::NORMAL,
                                None,
                            )
                            .map(|r| {
                                r.and_then(|resp| {
                                    resp.check_status()?;
                                    Ok(NoValue)
                                })
                            })
                    }
                }
            })
    }

    /// Authenticated GET, returns a `TokenPage` (parses JSON + Link header).
    fn get_token_page(&self, url: String) -> Task<Result<TokenPage, FetchError>> {
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<TokenPage, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let headers = json_headers_with_auth(&bearer);
                        accessor
                            .get(&url, &headers, RequestPriority::NORMAL, None)
                            .map(|result| {
                                let resp = result?;
                                resp.check_status()?;
                                let (next, prev) = resp
                                    .header("link")
                                    .map(parse_link_header)
                                    .unwrap_or((None, None));
                                let json: Value = serde_json::from_slice(&resp.data)
                                    .map_err(|e| FetchError::Json(e.to_string()))?;
                                let items = parse_token_list(&json);
                                Ok(TokenPage {
                                    items,
                                    next_page_url: next,
                                    previous_page_url: prev,
                                })
                            })
                    }
                }
            })
    }

    /// Retrieve profile information for the authenticated user.
    ///
    /// In SingleUser mode returns a synthetic profile without making a request.
    pub fn me(&self) -> Task<Result<Profile, FetchError>> {
        if matches!(self.auth_mode, AuthMode::SingleUser) {
            return orkester::resolved(Ok(Profile {
                id: 0,
                scopes: vec![
                    "assets:read".into(),
                    "assets:list".into(),
                    "assets:write".into(),
                    "profile:read".into(),
                    "tokens:read".into(),
                    "tokens:write".into(),
                ],
                username: "ion-user".into(),
                email: "none@example.com".into(),
                email_verified: true,
                avatar: "https://www.gravatar.com/avatar/4f14cc6c584f41d89ef1d34c8986ebfb.jpg?d=mp"
                    .into(),
                storage: ProfileStorage {
                    used: 0,
                    available: i64::MAX,
                    total: i64::MAX,
                },
            }));
        }

        let url = format!("{}/v1/me", self.base_url);
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<Profile, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let headers = json_headers_with_auth(&bearer);
                        accessor
                            .get(&url, &headers, RequestPriority::NORMAL, None)
                            .map(|result| {
                                let resp = result?;
                                resp.check_status()?;
                                let json: Value = serde_json::from_slice(&resp.data)
                                    .map_err(|e| FetchError::Json(e.to_string()))?;
                                let storage = json.get("storage");
                                Ok(Profile {
                                    id: json_i64(&json, "id").unwrap_or(-1),
                                    scopes: json_strings(&json, "scopes"),
                                    username: json_str(&json, "username").unwrap_or("").to_owned(),
                                    email: json_str(&json, "email").unwrap_or("").to_owned(),
                                    email_verified: json
                                        .get("emailVerified")
                                        .and_then(|v| v.as_bool())
                                        .unwrap_or(false),
                                    avatar: json_str(&json, "avatar").unwrap_or("").to_owned(),
                                    storage: ProfileStorage {
                                        used: storage
                                            .and_then(|s| s.get("used"))
                                            .and_then(|v| v.as_i64())
                                            .unwrap_or(0),
                                        available: storage
                                            .and_then(|s| s.get("available"))
                                            .and_then(|v| v.as_i64())
                                            .unwrap_or(0),
                                        total: storage
                                            .and_then(|s| s.get("total"))
                                            .and_then(|v| v.as_i64())
                                            .unwrap_or(0),
                                    },
                                })
                            })
                    }
                }
            })
    }

    /// Retrieve default imagery, terrain, and building asset IDs, plus curated
    /// quick-add assets.
    pub fn defaults(&self) -> Task<Result<Defaults, FetchError>> {
        let url = format!("{}/v1/defaults", self.base_url);
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token()
            .map(move |auth_result| -> Task<Result<Defaults, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let headers = json_headers_with_auth(&bearer);
                        accessor
                            .get(&url, &headers, RequestPriority::NORMAL, None)
                            .map(|result| {
                                let resp = result?;
                                resp.check_status()?;
                                let json: Value = serde_json::from_slice(&resp.data)
                                    .map_err(|e| FetchError::Json(e.to_string()))?;
                                Ok(parse_defaults(&json))
                            })
                    }
                }
            })
    }

    /// List assets with optional pagination/search.
    pub fn assets(&self, opts: ListOptions) -> Task<Result<PagedIonResponse<Asset>, FetchError>> {
        let url = self.list_url("/v1/assets", &opts);
        self.get_json(url)
    }

    /// Fetch a single asset by ID.
    pub fn asset(&self, id: u64) -> Task<Result<Asset, FetchError>> {
        let url = format!("{}/v1/assets/{id}", self.base_url);
        self.get_json(url)
    }

    /// Resolve the streaming endpoint for an asset (needed by tile loaders).
    pub fn asset_endpoint(&self, id: u64) -> Task<Result<AssetEndpoint, FetchError>> {
        let url = format!("{}/v1/assets/{id}/endpoint", self.base_url);
        self.get_json(url)
    }

    /// List tokens with optional pagination and sorting.
    ///
    /// Returns `TokenPage` which includes pagination cursor URLs.
    /// In SingleUser mode returns an empty page without making a request.
    pub fn tokens(&self, opts: ListTokensOptions) -> Task<Result<TokenPage, FetchError>> {
        if matches!(self.auth_mode, AuthMode::SingleUser) {
            return orkester::resolved(Ok(TokenPage {
                items: vec![],
                next_page_url: None,
                previous_page_url: None,
            }));
        }

        let mut url = format!("{}/v2/tokens", self.base_url);
        let mut sep = '?';
        if let Some(limit) = opts.limit {
            url.push_str(&format!("{sep}limit={limit}"));
            sep = '&';
        }
        if let Some(page) = opts.page {
            url.push_str(&format!("{sep}page={page}"));
            sep = '&';
        }
        if let Some(search) = &opts.search {
            url.push_str(&format!("{sep}search={}", url_encode(search)));
            sep = '&';
        }
        if let Some(sort_by) = &opts.sort_by {
            url.push_str(&format!("{sep}sortBy={sort_by}"));
            sep = '&';
        }
        if let Some(order) = opts.sort_order {
            let dir = if order == SortOrder::Ascending {
                "ASC"
            } else {
                "DESC"
            };
            url.push_str(&format!("{sep}sortOrder={dir}"));
        }

        self.get_token_page(url)
    }

    /// Fetch a single token by its ID.
    pub fn token(&self, token_id: &str) -> Task<Result<Token, FetchError>> {
        let url = format!("{}/v2/tokens/{token_id}", self.base_url);
        self.get_json(url)
    }

    /// Fetch the account's default token.
    pub fn default_token(&self) -> Task<Result<Token, FetchError>> {
        let url = format!("{}/v1/tokens/default", self.base_url);
        self.get_json(url)
    }

    /// Fetch the next page of token results.
    pub fn next_page(&self, page: &TokenPage) -> Task<Result<TokenPage, FetchError>> {
        match &page.next_page_url {
            Some(url) => self.get_token_page(url.clone()),
            None => orkester::resolved(Err(FetchError::Network("no next page available".into()))),
        }
    }

    /// Fetch the previous page of token results.
    pub fn previous_page(&self, page: &TokenPage) -> Task<Result<TokenPage, FetchError>> {
        match &page.previous_page_url {
            Some(url) => self.get_token_page(url.clone()),
            None => orkester::resolved(Err(FetchError::Network(
                "no previous page available".into(),
            ))),
        }
    }

    /// Create a new access token.
    ///
    /// - `asset_ids`: if `None`, the token allows access to all assets.
    /// - `allowed_urls`: if `None`, the token can be used from any URL.
    pub fn create_token(
        &self,
        name: &str,
        scopes: &[impl AsRef<str>],
        asset_ids: Option<&[u64]>,
        allowed_urls: Option<&[impl AsRef<str>]>,
    ) -> Task<Result<Token, FetchError>> {
        let url = format!("{}/v2/tokens", self.base_url);
        let body = build_token_body(name, scopes, asset_ids, allowed_urls);
        self.post_json(url, body)
    }

    /// Modify an existing token.
    pub fn modify_token(
        &self,
        token_id: &str,
        new_name: &str,
        new_scopes: &[impl AsRef<str>],
        new_asset_ids: Option<&[u64]>,
        new_allowed_urls: Option<&[impl AsRef<str>]>,
    ) -> Task<Result<NoValue, FetchError>> {
        let url = format!("{}/v2/tokens/{token_id}", self.base_url);
        let body = build_modify_token_body(new_name, new_scopes, new_asset_ids, new_allowed_urls);
        self.patch_no_body(url, body)
    }

    /// Geocode a query using the Ion geocoding service.
    pub fn geocode(
        &self,
        provider: GeocoderProviderType,
        request_type: GeocoderRequestType,
        query: &str,
    ) -> Task<Result<GeocoderResult, FetchError>> {
        let endpoint = if request_type == GeocoderRequestType::Autocomplete {
            "v1/geocode/autocomplete"
        } else {
            "v1/geocode/search"
        };
        let mut url = format!("{}/{endpoint}?text={}", self.base_url, url_encode(query));
        match provider {
            GeocoderProviderType::Bing => url.push_str("&geocoder=bing"),
            GeocoderProviderType::Google => url.push_str("&geocoder=google"),
            GeocoderProviderType::Default => {}
        }
        let accessor = Arc::clone(&self.accessor);
        self.ensure_valid_token().map(
            move |auth_result| -> Task<Result<GeocoderResult, FetchError>> {
                match auth_result {
                    Err(e) => orkester::resolved(Err(e)),
                    Ok(bearer) => {
                        let headers = json_headers_with_auth(&bearer);
                        accessor
                            .get(&url, &headers, RequestPriority::NORMAL, None)
                            .map(|result| {
                                let resp = result?;
                                resp.check_status()?;
                                let json: Value = serde_json::from_slice(&resp.data)
                                    .map_err(|e| FetchError::Json(e.to_string()))?;
                                Ok(parse_geocoder_result(&json))
                            })
                    }
                }
            },
        )
    }

    fn list_url(&self, path: &str, opts: &ListOptions) -> String {
        let mut url = format!("{}{path}", self.base_url);
        let mut sep = '?';
        if let Some(page) = opts.page {
            url.push_str(&format!("{sep}page={page}"));
            sep = '&';
        }
        if let Some(limit) = opts.limit {
            url.push_str(&format!("{sep}limit={limit}"));
            sep = '&';
        }
        if let Some(search) = &opts.search {
            url.push_str(&format!("{sep}search={}", url_encode(search)));
            let _ = sep;
        }
        url
    }
}

impl Client for Connection {
    fn base_url(&self) -> &str {
        &self.base_url
    }
    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}

fn json_str<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key)?.as_str()
}

fn json_i64(v: &Value, key: &str) -> Option<i64> {
    v.get(key)?.as_i64()
}

fn json_strings(v: &Value, key: &str) -> Vec<String> {
    v.get(key)
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| s.as_str())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn parse_token_list(json: &Value) -> Vec<Token> {
    json.get("items")
        .and_then(|a| a.as_array())
        .map(|arr| arr.iter().filter_map(|v| parse_token(v)).collect())
        .unwrap_or_default()
}

fn parse_token(json: &Value) -> Option<Token> {
    let id = json.get("id")?.as_str()?.to_owned();
    let name = json_str(json, "name").unwrap_or("").to_owned();
    let token = json_str(json, "token").unwrap_or("").to_owned();

    let asset_ids = json.get("assetIds").and_then(|v| {
        if v.is_null() {
            None
        } else {
            v.as_array()
                .map(|a| a.iter().filter_map(|n| n.as_i64()).collect())
        }
    });
    let allowed_urls = json.get("allowedUrls").and_then(|v| {
        if v.is_null() {
            None
        } else {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_owned))
                    .collect()
            })
        }
    });

    Some(Token {
        id,
        name,
        token,
        date_added: json_str(json, "dateAdded").map(str::to_owned),
        date_modified: json_str(json, "dateModified").map(str::to_owned),
        date_last_used: json_str(json, "dateLastUsed").map(str::to_owned),
        is_default: json
            .get("isDefault")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        asset_ids,
        allowed_urls,
        scopes: json_strings(json, "scopes"),
    })
}

fn parse_defaults(json: &Value) -> Defaults {
    let da = json.get("defaultAssets");
    let default_assets = DefaultAssets {
        imagery: da.and_then(|d| json_i64(d, "imagery")).unwrap_or(-1),
        terrain: da.and_then(|d| json_i64(d, "terrain")).unwrap_or(-1),
        buildings: da.and_then(|d| json_i64(d, "buildings")).unwrap_or(-1),
    };
    let quick_add_assets = json
        .get("quickAddAssets")
        .and_then(|a| a.as_array())
        .map(|arr| arr.iter().map(parse_quick_add_asset).collect())
        .unwrap_or_default();
    Defaults {
        default_assets,
        quick_add_assets,
    }
}

fn parse_quick_add_asset(json: &Value) -> QuickAddAsset {
    let raster_overlays = json
        .get("rasterOverlays")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| QuickAddRasterOverlay {
                    name: json_str(v, "name").unwrap_or("").to_owned(),
                    asset_id: json_i64(v, "assetId").unwrap_or(-1),
                    subscribed: v
                        .get("subscribed")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false),
                })
                .collect()
        })
        .unwrap_or_default();
    QuickAddAsset {
        name: json_str(json, "name").unwrap_or("").to_owned(),
        object_name: json_str(json, "objectName").unwrap_or("").to_owned(),
        description: json_str(json, "description").unwrap_or("").to_owned(),
        asset_id: json_i64(json, "assetId").unwrap_or(-1),
        asset_type: json_str(json, "type").unwrap_or("").to_owned(),
        subscribed: json
            .get("subscribed")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        raster_overlays,
    }
}

fn parse_geocoder_result(json: &Value) -> GeocoderResult {
    let features = json
        .get("features")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|feature| {
                    let label = feature
                        .pointer("/properties/label")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    // Try bbox first, then geometry/coordinates.
                    if let Some(bbox) = feature.get("bbox").and_then(|v| v.as_array()) {
                        if bbox.len() == 4 {
                            let vals: Vec<f64> = bbox.iter().filter_map(|v| v.as_f64()).collect();
                            if vals.len() == 4 {
                                return Some(GeocoderFeature {
                                    display_name: label,
                                    destination: GeocoderDestination::Rectangle {
                                        west: vals[0].to_radians(),
                                        south: vals[1].to_radians(),
                                        east: vals[2].to_radians(),
                                        north: vals[3].to_radians(),
                                    },
                                });
                            }
                        }
                    }
                    if let Some(coords) = feature
                        .pointer("/geometry/coordinates")
                        .and_then(|v| v.as_array())
                    {
                        if coords.len() == 2 {
                            let lon = coords[0].as_f64()?;
                            let lat = coords[1].as_f64()?;
                            return Some(GeocoderFeature {
                                display_name: label,
                                destination: GeocoderDestination::Point {
                                    longitude: lon.to_radians(),
                                    latitude: lat.to_radians(),
                                },
                            });
                        }
                    }
                    None
                })
                .collect()
        })
        .unwrap_or_default();

    let attributions = json
        .get("attributions")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|v| GeocoderAttribution {
                    html: json_str(v, "html").unwrap_or("").to_owned(),
                    show_on_screen: !v
                        .get("collapsible")
                        .and_then(|c| c.as_bool())
                        .unwrap_or(false),
                })
                .collect()
        })
        .unwrap_or_default();

    GeocoderResult {
        attributions,
        features,
    }
}

fn build_token_body(
    name: &str,
    scopes: &[impl AsRef<str>],
    asset_ids: Option<&[u64]>,
    allowed_urls: Option<&[impl AsRef<str>]>,
) -> Vec<u8> {
    let scopes_json: Vec<Value> = scopes
        .iter()
        .map(|s| Value::String(s.as_ref().to_owned()))
        .collect();
    let asset_ids_json = match asset_ids {
        Some(ids) => Value::Array(ids.iter().map(|&n| Value::Number(n.into())).collect()),
        None => Value::Null,
    };
    let allowed_urls_json = match allowed_urls {
        Some(urls) => Value::Array(
            urls.iter()
                .map(|u| Value::String(u.as_ref().to_owned()))
                .collect(),
        ),
        None => Value::Null,
    };
    let body = serde_json::json!({
        "name": name,
        "scopes": scopes_json,
        "assetIds": asset_ids_json,
        "allowedUrls": allowed_urls_json,
    });
    serde_json::to_vec(&body).unwrap_or_default()
}

fn build_modify_token_body(
    new_name: &str,
    new_scopes: &[impl AsRef<str>],
    new_asset_ids: Option<&[u64]>,
    new_allowed_urls: Option<&[impl AsRef<str>]>,
) -> Vec<u8> {
    let scopes_json: Vec<Value> = new_scopes
        .iter()
        .map(|s| Value::String(s.as_ref().to_owned()))
        .collect();
    let asset_ids_json = match new_asset_ids {
        Some(ids) => Value::Array(ids.iter().map(|&n| Value::Number(n.into())).collect()),
        None => Value::Null,
    };
    let allowed_urls_json = match new_allowed_urls {
        Some(urls) => Value::Array(
            urls.iter()
                .map(|u| Value::String(u.as_ref().to_owned()))
                .collect(),
        ),
        None => Value::Null,
    };
    let body = serde_json::json!({
        "name": new_name,
        "scopes": scopes_json,
        "assetIds": asset_ids_json,
        "newAllowedUrls": allowed_urls_json,
    });
    serde_json::to_vec(&body).unwrap_or_default()
}

fn json_headers_with_auth(bearer: &str) -> Vec<(String, String)> {
    let mut h = vec![("Accept".to_owned(), "application/json".to_owned())];
    if !bearer.is_empty() {
        h.push(("Authorization".to_owned(), bearer.to_owned()));
    }
    h
}

/// Parse `Link: <url>; rel="next", <url>; rel="prev"` -> (next, prev).
fn parse_link_header(header: &str) -> (Option<String>, Option<String>) {
    let mut next = None;
    let mut prev = None;
    for part in header.split(',') {
        let part = part.trim();
        if let Some((url_part, rel_part)) = part.split_once(';') {
            let url = url_part
                .trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_owned();
            let rel = rel_part.trim().to_lowercase();
            if rel.contains("\"next\"") || rel == "rel=next" {
                next = Some(url);
            } else if rel.contains("\"prev\"") || rel.contains("\"previous\"") || rel == "rel=prev"
            {
                prev = Some(url);
            }
        }
    }
    (next, prev)
}

/// Derive an API URL from an Ion instance URL: `https://ion.example.com` -> `https://api.example.com/`.
fn derive_api_url(ion_url: &str) -> Option<String> {
    let url = ion_url.trim_end_matches('/');
    // Strip scheme.
    let without_scheme = url.split_once("://").map(|(_, rest)| rest)?;
    // Replace leading hostname with "api.{host}".
    let host = without_scheme.split('/').next()?;
    Some(format!("https://api.{host}/"))
}

/// Percent-encode a string for use in URL query parameters.
pub(crate) fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Parse the auth code from the first line of an HTTP GET request.
///
/// Looks for `GET /{path}?code=xxx` or `GET /?code=xxx`.
fn parse_auth_code_from_request(request: &str) -> Option<String> {
    let first_line = request.lines().next()?;
    // e.g. "GET /callback?code=abc123&state=xyz HTTP/1.1"
    let path_part = first_line.split_whitespace().nth(1)?;
    let query = path_part.split_once('?').map(|(_, q)| q)?;
    for pair in query.split('&') {
        if let Some((key, value)) = pair.split_once('=') {
            if key == "code" {
                return Some(value.to_owned());
            }
        }
    }
    None
}

/// Generate a PKCE (code_verifier, code_challenge) pair using OS randomness.
#[cfg(not(target_arch = "wasm32"))]
fn generate_pkce() -> Result<(String, String), String> {
    use base64::Engine as _;
    use sha2::{Digest, Sha256};

    let mut verifier_bytes = [0u8; 32];
    getrandom::getrandom(&mut verifier_bytes)
        .map_err(|e| format!("failed to generate PKCE verifier: {e}"))?;

    let verifier = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(hasher.finalize());

    Ok((verifier, challenge))
}

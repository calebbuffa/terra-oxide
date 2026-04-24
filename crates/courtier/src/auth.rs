//! Authentication providers for injecting credentials into requests.

use crate::fetch::FetchError;
use orkester::Task;

/// Provides authentication credentials for outgoing requests.
///
/// Implementations hold the credential (API key, Bearer token, etc.) and know
/// how to inject it into a URL or header list.
pub trait AuthProvider: Send + Sync + 'static {
    /// Returns the current credential string (synchronous — tokens are assumed stable).
    fn get_token(&self) -> String;

    /// Injects the credential into the request, returning `(url, headers)`.
    ///
    /// `url` may have a query parameter appended (API key style).
    /// `headers` may have an `Authorization` header prepended (Bearer style).
    fn inject(
        &self,
        token: &str,
        url: &str,
        headers: &[(String, String)],
    ) -> (String, Vec<(String, String)>);

    /// Refresh the credential and return the new value.
    ///
    /// Default implementation returns the current token unchanged (no-op for
    /// static credentials). Override for OAuth2 or other refreshable tokens.
    fn refresh(&self) -> Task<Result<String, FetchError>> {
        let token = self.get_token();
        orkester::resolved(Ok(token))
    }
}

/// API-key style authentication — appends `?{param}={key}` to the URL.
pub struct ApiKeyAuth {
    param_name: String,
    key: String,
}

impl ApiKeyAuth {
    /// Inject the key as a query parameter (e.g., ArcGIS `?token=…`).
    pub fn query_param(param_name: impl Into<String>, key: impl Into<String>) -> Self {
        Self {
            param_name: param_name.into(),
            key: key.into(),
        }
    }
}

impl AuthProvider for ApiKeyAuth {
    fn get_token(&self) -> String {
        self.key.clone()
    }

    fn inject(
        &self,
        token: &str,
        url: &str,
        headers: &[(String, String)],
    ) -> (String, Vec<(String, String)>) {
        let sep = if url.contains('?') { '&' } else { '?' };
        let new_url = format!("{url}{sep}{}={token}", self.param_name);
        (new_url, headers.to_vec())
    }
}

/// Bearer-token style authentication — injects `Authorization: Bearer {token}`.
pub struct BearerTokenAuth {
    token: String,
}

impl BearerTokenAuth {
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

impl AuthProvider for BearerTokenAuth {
    fn get_token(&self) -> String {
        self.token.clone()
    }

    fn inject(
        &self,
        token: &str,
        url: &str,
        headers: &[(String, String)],
    ) -> (String, Vec<(String, String)>) {
        let mut h = headers.to_vec();
        h.push(("Authorization".to_owned(), format!("Bearer {token}")));
        (url.to_owned(), h)
    }
}

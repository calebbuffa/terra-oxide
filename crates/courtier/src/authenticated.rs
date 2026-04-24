//! Authentication-injecting decorator accessor.

use std::sync::Arc;

use crate::auth::AuthProvider;
use crate::fetch::{AssetAccessor, AssetResponse, FetchError, RequestPriority};
use orkester::{CancellationToken, Task};

/// Decorator that injects authentication credentials into every request.
///
/// Auth is injected transparently — callers of `request()`, `get()`, and
/// `get_range()` need not know about credentials. On a 401 response, the
/// auth provider's `refresh()` is called once and the request is retried.
///
/// # Example
/// ```rust,ignore
/// let accessor = Arc::new(AuthenticatedAccessor::new(
///     HttpAccessor::new(bg_ctx.clone()),
///     ApiKeyAuth::query_param("token", "my_arcgis_token"),
///     bg_ctx,
/// ));
/// let client = MapServerClient::new(url, accessor);
/// ```
pub struct AuthenticatedAccessor<A, P> {
    inner: Arc<A>,
    auth: Arc<P>,
    ctx: orkester::Context,
}

impl<A: AssetAccessor, P: AuthProvider> AuthenticatedAccessor<A, P> {
    /// Create a new `AuthenticatedAccessor`.
    ///
    /// `ctx` is used to schedule the 401-refresh continuation on a background
    /// worker.  Pass the same background context used by your HTTP accessor.
    pub fn new(inner: A, auth: P, ctx: orkester::Context) -> Self {
        Self {
            inner: Arc::new(inner),
            auth: Arc::new(auth),
            ctx,
        }
    }
}

impl<A: AssetAccessor, P: AuthProvider> AssetAccessor for AuthenticatedAccessor<A, P> {
    fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        let cred = self.auth.get_token();
        let (authed_url, authed_headers) = self.auth.inject(&cred, url, headers);
        let first = self
            .inner
            .request(method, &authed_url, &authed_headers, body, priority, token);

        // Clone everything needed by the 401-retry closure.
        let auth = Arc::clone(&self.auth);
        let inner = Arc::clone(&self.inner);
        let method = method.to_owned();
        let url = url.to_owned();
        let headers = headers.to_vec();
        let body = body.map(|b| b.to_vec());
        let token = token.cloned();
        let ctx1 = self.ctx.clone();
        let ctx2 = self.ctx.clone();

        first.then(&ctx1, move |result| {
            let is_401 = matches!(&result, Ok(r) if r.status == 401);
            if is_401 {
                // Refresh token, then retry once.
                auth.refresh().then(&ctx2, move |_| {
                    let new_cred = auth.get_token();
                    let (url2, headers2) = auth.inject(&new_cred, &url, &headers);
                    inner.request(
                        &method,
                        &url2,
                        &headers2,
                        body.as_deref(),
                        priority,
                        token.as_ref(),
                    )
                })
            } else {
                orkester::resolved(result)
            }
        })
    }
}

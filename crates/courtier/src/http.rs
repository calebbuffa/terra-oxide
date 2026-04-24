//! Native HTTP/HTTPS accessor backed by [`ureq`].

use crate::fetch::{
    AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority, cancelled_error,
};
use orkester::{CancellationToken, Task};
use std::io;
use std::sync::Arc;

/// Retry configuration for [`HttpAccessor`].
///
/// Controls the number of attempts and the exponential back-off delays.
/// Applied only to idempotent methods (GET, HEAD) and to retryable server
/// responses (429, 500, 502, 503, 504) — never to cancellations or
/// non-idempotent methods with transport errors.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Debug)]
pub struct RetryConfig {
    /// Total number of attempts (including the first). Default: 3.
    pub max_attempts: u32,
    /// Delay before the second attempt in milliseconds. Default: 500.
    pub initial_delay_ms: u64,
    /// Maximum delay cap in milliseconds. Default: 2000.
    pub max_delay_ms: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 500,
            max_delay_ms: 2000,
        }
    }
}

/// Compute exponential backoff delay for a retry attempt.
///
/// Uses saturating arithmetic and caps the shift at 62 to prevent `1u64 << 63`
/// from panicking in debug builds when `attempt >= 63`.
#[cfg(not(target_arch = "wasm32"))]
fn backoff_millis(attempt: u32, cfg: &RetryConfig) -> u64 {
    let shift = attempt.min(62); // prevent overflow: 1u64 << 63 panics in debug
    cfg.initial_delay_ms
        .saturating_mul(1u64 << shift)
        .min(cfg.max_delay_ms)
}

/// Blocking HTTP/HTTPS accessor backed by `ureq`, dispatched on background workers.
///
/// Each call issues a ureq request on the orkester background thread pool so
/// the main thread is never blocked.
#[cfg(not(target_arch = "wasm32"))]
pub struct HttpAccessor {
    ctx: orkester::Context,
    default_headers: Arc<[(String, String)]>,
    timeout: Option<std::time::Duration>,
    retry_config: RetryConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl HttpAccessor {
    /// Create an accessor with no default headers and no timeout.
    pub fn new(ctx: orkester::Context) -> Self {
        Self {
            ctx,
            default_headers: Arc::from([]),
            timeout: None,
            retry_config: RetryConfig::default(),
        }
    }

    /// Create an accessor with default headers applied to every request.
    pub fn with_headers(
        ctx: orkester::Context,
        headers: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        Self {
            ctx,
            default_headers: headers.into_iter().collect(),
            timeout: None,
            retry_config: RetryConfig::default(),
        }
    }

    /// Set a timeout that applies to every request.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Override the retry configuration.
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }

    fn build_request(
        method: &str,
        url: &str,
        default_headers: &[(String, String)],
        extra_headers: &[(String, String)],
        timeout: Option<std::time::Duration>,
    ) -> ureq::Request {
        let mut req = ureq::request(method, url);
        for (k, v) in default_headers.iter().chain(extra_headers.iter()) {
            req = req.set(k, v);
        }
        if let Some(t) = timeout {
            req = req.timeout(t);
        }
        req
    }

    fn read_response(
        response: ureq::Response,
        token: Option<&CancellationToken>,
    ) -> Result<AssetResponse, FetchError> {
        use io::Read;
        let status = response.status();
        let headers: Vec<(String, String)> = response
            .headers_names()
            .into_iter()
            .filter_map(|name| {
                response
                    .header(&name)
                    .map(|v| (name.to_ascii_lowercase(), v.to_owned()))
            })
            .collect();
        let mut buf = Vec::new();
        let mut reader = response.into_reader();
        let mut chunk = [0u8; 64 * 1024];
        loop {
            if token.is_some_and(|t| t.is_cancelled()) {
                return Err(cancelled_error());
            }
            let n = reader
                .read(&mut chunk)
                .map_err(|e| FetchError::Network(e.to_string()))?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&chunk[..n]);
        }
        Ok(AssetResponse {
            status,
            headers,
            data: buf,
            content_encoding: ContentEncoding::None,
        })
    }

    fn map_ureq_error(e: ureq::Error) -> FetchError {
        match e {
            ureq::Error::Status(status, resp) => FetchError::Http {
                status,
                message: resp.status_text().to_owned(),
            },
            ureq::Error::Transport(t) => FetchError::Network(t.to_string()),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetAccessor for HttpAccessor {
    fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        _priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        if token.is_some_and(|t| t.is_cancelled()) {
            return orkester::resolved(Err(cancelled_error()));
        }
        let method = method.to_owned();
        let url = url.to_owned();
        let default_headers = Arc::clone(&self.default_headers);
        let extra_headers = headers.to_vec();
        let body = body.map(|b| b.to_vec());
        let timeout = self.timeout;
        let token = token.cloned();
        let retry_config = self.retry_config.clone();
        self.ctx.run(move || {
            let is_idempotent = method == "GET" || method == "HEAD";

            for attempt in 0..retry_config.max_attempts {
                if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                    return Err(cancelled_error());
                }

                let req =
                    Self::build_request(&method, &url, &default_headers, &extra_headers, timeout);
                let result = if let Some(ref body_bytes) = body {
                    req.send_bytes(body_bytes)
                } else {
                    req.call()
                };

                let is_last = attempt + 1 >= retry_config.max_attempts;

                match result {
                    Ok(resp) => return Self::read_response(resp, token.as_ref()),
                    Err(ureq::Error::Status(status, resp)) => {
                        let is_retryable =
                            !is_last && matches!(status, 429 | 500 | 502 | 503 | 504);
                        if !is_retryable {
                            let message = resp.into_string().unwrap_or_default();
                            return Err(FetchError::Http { status, message });
                        }
                        // Respect Retry-After header on 429 (capped at 10 s).
                        let delay = if status == 429 {
                            resp.header("retry-after")
                                .and_then(|v| v.parse::<u64>().ok())
                                .map(|n| std::time::Duration::from_secs(n.min(10)))
                                .unwrap_or_else(|| {
                                    std::time::Duration::from_millis(backoff_millis(
                                        attempt,
                                        &retry_config,
                                    ))
                                })
                        } else {
                            std::time::Duration::from_millis(backoff_millis(attempt, &retry_config))
                        };
                        std::thread::sleep(delay);
                    }
                    Err(ureq::Error::Transport(t)) => {
                        if is_last || !is_idempotent {
                            return Err(FetchError::Network(t.to_string()));
                        }
                        std::thread::sleep(std::time::Duration::from_millis(backoff_millis(
                            attempt,
                            &retry_config,
                        )));
                    }
                }
            }

            // Unreachable: the loop always returns on the last attempt.
            Err(FetchError::Network("retry loop exhausted".into()))
        })
    }
}

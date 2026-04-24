//! Core I/O trait, error type, and response types.

use orkester::{CancellationToken, Task};

/// Content-encoding of the raw bytes in an [`AssetResponse`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ContentEncoding {
    #[default]
    None,
    Gzip,
}

/// Response from an asset request.
#[derive(Debug)]
pub struct AssetResponse {
    /// HTTP status code (or equivalent for non-HTTP sources).
    pub status: u16,
    /// Response headers (lower-cased names).
    pub headers: Vec<(String, String)>,
    /// Response body (may be compressed — see [`content_encoding`](AssetResponse::content_encoding)).
    pub data: Vec<u8>,
    /// Content encoding of [`data`](AssetResponse::data).
    pub content_encoding: ContentEncoding,
}

impl AssetResponse {
    /// Case-insensitive lookup of a single response header.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Case-insensitive lookup of the `Content-Type` header.
    #[inline]
    pub fn content_type(&self) -> Option<&str> {
        self.header("content-type")
    }

    /// Returns `Ok(())` for 2xx status codes, `Err(FetchError::Http)` otherwise.
    #[inline]
    pub fn check_status(&self) -> Result<(), FetchError> {
        if self.status >= 200 && self.status < 300 {
            Ok(())
        } else {
            Err(FetchError::Http {
                status: self.status,
                message: String::new(),
            })
        }
    }

    /// Borrow the response bytes, asserting they are already decompressed.
    #[inline]
    pub fn decompressed_data(&self) -> &[u8] {
        debug_assert_eq!(
            self.content_encoding,
            ContentEncoding::None,
            "AssetResponse::decompressed_data called on a gzip-compressed response \
             — wrap your accessor with GunzipAccessor first"
        );
        &self.data
    }

    /// Consume the response and return the raw bytes, asserting they are already decompressed.
    #[inline]
    pub fn into_decompressed_data(self) -> Vec<u8> {
        debug_assert_eq!(
            self.content_encoding,
            ContentEncoding::None,
            "AssetResponse::into_decompressed_data called on a gzip-compressed response \
             — wrap your accessor with GunzipAccessor first"
        );
        self.data
    }
}

/// Error type for asset fetching operations.
///
/// Replaces `io::Error` to preserve HTTP status codes (enabling 401 -> token-refresh).
#[derive(Debug, thiserror::Error)]
pub enum FetchError {
    #[error("HTTP {status}: {message}")]
    Http { status: u16, message: String },
    #[error("network error: {0}")]
    Network(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(String),
    #[error("request cancelled")]
    Cancelled,
}

impl FetchError {
    /// Returns the HTTP status code if this is an HTTP error.
    pub fn status(&self) -> Option<u16> {
        match self {
            Self::Http { status, .. } => Some(*status),
            _ => None,
        }
    }

    pub fn is_unauthorized(&self) -> bool {
        matches!(self, Self::Http { status: 401, .. })
    }

    pub fn is_not_found(&self) -> bool {
        matches!(self, Self::Http { status: 404, .. })
    }
}

/// Fetch priority hint passed to [`AssetAccessor::request`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct RequestPriority(pub u8);

impl RequestPriority {
    pub const LOW: Self = Self(0);
    pub const NORMAL: Self = Self(128);
    pub const HIGH: Self = Self(255);
}

/// Error shorthand for cancelled I/O.
#[inline]
pub(crate) fn cancelled_error() -> FetchError {
    FetchError::Cancelled
}

/// Async asset accessor for fetching data from network, file, or cache.
///
/// Implementations handle the actual I/O (HTTP, file system, SLPK archive, etc.).
///
/// The single required method is [`request`](AssetAccessor::request).
/// Convenience methods [`get`] and [`get_range`] have default implementations
/// that delegate to `request`.
pub trait AssetAccessor: Send + Sync + 'static {
    /// Issue an HTTP-like request with the given method, URL, headers, and optional body.
    fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>>;

    /// Fetch the asset at the given URL via GET.
    fn get(
        &self,
        url: &str,
        headers: &[(String, String)],
        priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        self.request("GET", url, headers, None, priority, token)
    }

    /// Fetch a byte range of the asset at the given URL.
    fn get_range(
        &self,
        url: &str,
        headers: &[(String, String)],
        priority: RequestPriority,
        offset: u64,
        length: u64,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        let last = offset
            .checked_add(length)
            .and_then(|e| e.checked_sub(1))
            .unwrap_or(u64::MAX);
        let mut h = headers.to_vec();
        h.push(("Range".to_owned(), format!("bytes={offset}-{last}")));
        self.request("GET", url, &h, None, priority, token)
    }
}

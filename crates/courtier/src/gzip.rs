use crate::fetch::{AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority};
use flate2::read::GzDecoder;
use orkester::{CancellationToken, Task};
use std::io::Read;

/// Decorator that decompresses gzip-encoded responses from an inner accessor.
pub struct GunzipAccessor<A> {
    inner: A,
}

impl<A: AssetAccessor> GunzipAccessor<A> {
    pub fn new(inner: A) -> Self {
        Self { inner }
    }
}

fn gunzip_response(mut resp: AssetResponse) -> Result<AssetResponse, FetchError> {
    if resp.content_encoding == ContentEncoding::Gzip {
        let mut decoder = GzDecoder::new(resp.data.as_slice());
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        resp.data = decompressed;
        resp.content_encoding = ContentEncoding::None;
    }
    Ok(resp)
}

impl<A: AssetAccessor> AssetAccessor for GunzipAccessor<A> {
    fn request(
        &self,
        method: &str,
        url: &str,
        headers: &[(String, String)],
        body: Option<&[u8]>,
        priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        self.inner
            .request(method, url, headers, body, priority, token)
            .map(|result| result.and_then(gunzip_response))
    }

    fn get_range(
        &self,
        url: &str,
        headers: &[(String, String)],
        priority: RequestPriority,
        offset: u64,
        length: u64,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        self.inner
            .get_range(url, headers, priority, offset, length, token)
            .map(|result| result.and_then(gunzip_response))
    }
}

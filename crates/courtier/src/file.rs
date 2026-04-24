use crate::fetch::{
    AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority, cancelled_error,
};
use orkester::{CancellationToken, Task};
use std::io::{self, Read};
use std::path::PathBuf;

const MAX_FILE_RANGE: u64 = 512 * 1024 * 1024;

/// Synchronous `file://` accessor for local filesystem paths.
///
/// Resolves `file:///absolute/path` or bare paths by reading from disk
/// on a background worker thread.
#[cfg(not(target_arch = "wasm32"))]
pub struct FileAccessor {
    ctx: orkester::Context,
}

#[cfg(not(target_arch = "wasm32"))]
impl FileAccessor {
    pub fn new(ctx: orkester::Context) -> Self {
        Self { ctx }
    }

    fn url_to_path(url: &str) -> PathBuf {
        let stripped = if let Some(rest) = url.strip_prefix("file:///") {
            rest
        } else if let Some(rest) = url.strip_prefix("file://") {
            rest
        } else {
            url
        };
        let stripped = stripped.split('?').next().unwrap_or(stripped);
        let stripped = stripped.split('#').next().unwrap_or(stripped);
        let normalized;
        let stripped = if stripped.contains('\\') && stripped.contains('/') {
            normalized = stripped.replace('/', "\\");
            normalized.as_str()
        } else {
            stripped
        };
        PathBuf::from(stripped)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetAccessor for FileAccessor {
    fn request(
        &self,
        _method: &str,
        url: &str,
        _headers: &[(String, String)],
        _body: Option<&[u8]>,
        _priority: RequestPriority,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        if token.is_some_and(|t| t.is_cancelled()) {
            return orkester::resolved(Err(cancelled_error()));
        }

        // Check if this is a range request from the Range header (injected by get_range default impl).
        // File accessor reads the whole file and slices for range requests.
        let path = Self::url_to_path(url);
        let token = token.cloned();
        self.ctx.run(move || {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                return Err(cancelled_error());
            }
            let data = std::fs::read(&path)
                .map_err(|e| io::Error::new(e.kind(), format!("{e} (path: {})", path.display())))?;
            Ok(AssetResponse {
                status: 200,
                headers: Vec::new(),
                data,
                content_encoding: ContentEncoding::None,
            })
        })
    }

    fn get_range(
        &self,
        url: &str,
        _headers: &[(String, String)],
        _priority: RequestPriority,
        offset: u64,
        length: u64,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        debug_assert!(length > 0, "get_range called with length=0");
        if length > MAX_FILE_RANGE {
            return orkester::resolved(Err(FetchError::Io(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("get_range: requested length {length} exceeds limit ({MAX_FILE_RANGE})"),
            ))));
        }
        if token.is_some_and(|t| t.is_cancelled()) {
            return orkester::resolved(Err(cancelled_error()));
        }
        let path = Self::url_to_path(url);
        let token = token.cloned();
        self.ctx.run(move || {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                return Err(cancelled_error());
            }
            let mut file = std::fs::File::open(&path)
                .map_err(|e| io::Error::new(e.kind(), format!("{e} (path: {})", path.display())))?;
            io::Seek::seek(&mut file, io::SeekFrom::Start(offset))?;
            let mut buf = vec![0u8; length as usize];
            let n = file.read(&mut buf)?;
            buf.truncate(n);
            Ok(AssetResponse {
                status: 206,
                headers: Vec::new(),
                data: buf,
                content_encoding: ContentEncoding::None,
            })
        })
    }
}

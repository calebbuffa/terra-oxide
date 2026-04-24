//! ZIP/SLPK/3TZ archive accessor.

use crate::fetch::{
    AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority, cancelled_error,
};
use flate2::read::GzDecoder;
use orkester::{CancellationToken, Task};
use std::collections::HashMap;
use std::io::{self, Read};
use std::sync::Arc;

/// Self-contained accessor for `.zip`, `.slpk`, and `.3tz` local archives.
///
/// The ZIP central directory is read once at construction; thereafter each
/// lookup is O(1) via hash map. Entries whose names end in `.gz` are
/// decompressed with gzip after extraction.
#[cfg(not(target_arch = "wasm32"))]
pub struct ArchiveAccessor {
    index: Arc<HashMap<String, zip::CompressionMethod>>,
    data: Arc<[u8]>,
    ctx: orkester::Context,
}

#[cfg(not(target_arch = "wasm32"))]
impl ArchiveAccessor {
    /// Open and index an archive from disk.
    pub fn open(path: impl AsRef<std::path::Path>, ctx: orkester::Context) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        Self::from_bytes(data, ctx)
    }

    /// Create from an already-loaded byte buffer.
    pub fn from_bytes(data: Vec<u8>, ctx: orkester::Context) -> io::Result<Self> {
        let cursor = io::Cursor::new(&data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut index = HashMap::with_capacity(archive.len());
        for i in 0..archive.len() {
            let file = archive
                .by_index_raw(i)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            index.insert(file.name().to_owned(), file.compression());
        }

        Ok(Self {
            index: Arc::new(index),
            data: Arc::from(data.into_boxed_slice()),
            ctx,
        })
    }

    fn read_entry_sync(
        data: &[u8],
        index: &HashMap<String, zip::CompressionMethod>,
        name: &str,
    ) -> io::Result<Vec<u8>> {
        let key = name.trim_start_matches('/');

        let method = *index.get(key).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("entry not found: {key}"))
        })?;

        let cursor = io::Cursor::new(data);
        let mut archive = zip::ZipArchive::new(cursor)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut file = archive
            .by_name(key)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut buf = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut buf)?;

        let bytes = if key.ends_with(".gz") && method == zip::CompressionMethod::Stored {
            let mut dec = GzDecoder::new(buf.as_slice());
            let mut out = Vec::new();
            dec.read_to_end(&mut out)?;
            out
        } else {
            buf
        };

        Ok(bytes)
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn archive_url_path(url: &str) -> &str {
    if let Some(pos) = url.find("://") {
        let after_scheme = &url[pos + 3..];
        after_scheme
            .find('/')
            .map_or(after_scheme, |i| &after_scheme[i..])
    } else {
        url
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl AssetAccessor for ArchiveAccessor {
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
        let path = archive_url_path(url).to_owned();
        let data = Arc::clone(&self.data);
        let index = Arc::clone(&self.index);
        let token = token.cloned();
        self.ctx.run(move || {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                return Err(cancelled_error());
            }
            Self::read_entry_sync(&data, &index, &path)
                .map_err(FetchError::Io)
                .map(|data| AssetResponse {
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
        if token.is_some_and(|t| t.is_cancelled()) {
            return orkester::resolved(Err(cancelled_error()));
        }
        let path = archive_url_path(url).to_owned();
        let data = Arc::clone(&self.data);
        let index = Arc::clone(&self.index);
        let token = token.cloned();
        self.ctx.run(move || {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                return Err(cancelled_error());
            }
            Self::read_entry_sync(&data, &index, &path)
                .map_err(FetchError::Io)
                .and_then(|entry_data| {
                    let start = offset as usize;
                    let end = (offset + length) as usize;
                    if end > entry_data.len() {
                        return Err(FetchError::Io(io::Error::new(
                            io::ErrorKind::UnexpectedEof,
                            format!(
                                "get_range out-of-bounds: offset={offset} length={length} entry_len={}",
                                entry_data.len()
                            ),
                        )));
                    }
                    Ok(AssetResponse {
                        status: 206,
                        headers: Vec::new(),
                        data: entry_data[start..end].to_vec(),
                        content_encoding: ContentEncoding::None,
                    })
                })
        })
    }
}

//! WASM-native HTTP accessor using the browser Fetch API.

use crate::fetch::{
    AssetAccessor, AssetResponse, ContentEncoding, FetchError, RequestPriority, cancelled_error,
};
use js_sys::{ArrayBuffer, Uint8Array};
use orkester::{CancellationToken, Task, resolved};
use std::sync::Arc;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{AbortController, RequestInit, RequestMode, Response};

/// HTTP accessor that uses the browser's native `fetch()` API.
///
/// This is the WASM equivalent of `HttpAccessor`. Both implement
/// `AssetAccessor` so the rest of the codebase is target-agnostic.
#[derive(Clone)]
pub struct WasmFetchAccessor {
    default_headers: Arc<[(String, String)]>,
}

impl WasmFetchAccessor {
    pub fn new() -> Self {
        Self {
            default_headers: Arc::from([]),
        }
    }

    pub fn with_headers(headers: impl IntoIterator<Item = (String, String)>) -> Self {
        Self {
            default_headers: headers.into_iter().collect(),
        }
    }
}

impl Default for WasmFetchAccessor {
    fn default() -> Self {
        Self::new()
    }
}

fn js_err_to_fetch(e: wasm_bindgen::JsValue) -> FetchError {
    let msg = e.as_string().unwrap_or_else(|| format!("{e:?}"));
    FetchError::Network(msg)
}

async fn do_fetch(
    method: String,
    url: String,
    default_headers: Arc<[(String, String)]>,
    extra_headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    token: Option<CancellationToken>,
) -> Result<AssetResponse, FetchError> {
    let window =
        web_sys::window().ok_or_else(|| FetchError::Network("no browser window".into()))?;

    let mut init = RequestInit::new();
    init.set_method(&method);
    init.set_mode(RequestMode::Cors);

    if let Some(body_bytes) = body {
        let uint8 = js_sys::Uint8Array::from(body_bytes.as_slice());
        init.set_body(&uint8.buffer());
    }

    let headers = web_sys::Headers::new().map_err(js_err_to_fetch)?;
    for (k, v) in default_headers.iter().chain(extra_headers.iter()) {
        headers.set(k, v).map_err(js_err_to_fetch)?;
    }
    init.set_headers(&headers);

    let controller = AbortController::new().map_err(js_err_to_fetch)?;
    init.set_signal(Some(&controller.signal()));
    let _reg = token.as_ref().map(|t| {
        let c = controller.clone();
        t.on_cancel(move || c.abort())
    });

    let request = web_sys::Request::new_with_str_and_init(&url, &init).map_err(js_err_to_fetch)?;

    if token.as_ref().is_some_and(|t| t.is_cancelled()) {
        return Err(cancelled_error());
    }

    let resp_value = JsFuture::from(window.fetch_with_request(&request))
        .await
        .map_err(|e| {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                cancelled_error()
            } else {
                js_err_to_fetch(e)
            }
        })?;

    let resp: Response = resp_value.dyn_into().map_err(js_err_to_fetch)?;
    let status = resp.status();
    let headers = snapshot_headers(&resp);

    let array_buffer = JsFuture::from(resp.array_buffer().map_err(js_err_to_fetch)?)
        .await
        .map_err(|e| {
            if token.as_ref().is_some_and(|t| t.is_cancelled()) {
                cancelled_error()
            } else {
                js_err_to_fetch(e)
            }
        })?;

    let array_buffer: ArrayBuffer = array_buffer.dyn_into().map_err(js_err_to_fetch)?;
    let data = Uint8Array::new(&array_buffer).to_vec();

    Ok(AssetResponse {
        status,
        headers,
        data,
        content_encoding: ContentEncoding::None,
    })
}

fn snapshot_headers(resp: &Response) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let iter: js_sys::Iterator = match resp.headers().entries() {
        Ok(v) => v,
        Err(_) => return out,
    };
    loop {
        let next = match iter.next() {
            Ok(n) => n,
            Err(_) => break,
        };
        if next.done() {
            break;
        }
        let pair = js_sys::Array::from(&next.value());
        if pair.length() == 2 {
            let k = pair.get(0).as_string().unwrap_or_default();
            let v = pair.get(1).as_string().unwrap_or_default();
            out.push((k.to_ascii_lowercase(), v));
        }
    }
    out
}

impl AssetAccessor for WasmFetchAccessor {
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
            return resolved(Err(cancelled_error()));
        }
        let method = method.to_owned();
        let url = url.to_owned();
        let default_headers = Arc::clone(&self.default_headers);
        let extra_headers = headers.to_vec();
        let body = body.map(|b| b.to_vec());
        let token = token.cloned();

        let future = do_fetch(method, url, default_headers, extra_headers, body, token);
        let (resolver, task) = orkester::pair();
        wasm_bindgen_futures::spawn_local(async move {
            resolver.resolve(future.await);
        });
        task
    }

    fn get_range(
        &self,
        url: &str,
        headers: &[(String, String)],
        _priority: RequestPriority,
        offset: u64,
        length: u64,
        token: Option<&CancellationToken>,
    ) -> Task<Result<AssetResponse, FetchError>> {
        if length == 0 {
            return resolved(Err(FetchError::Network(
                "get_range called with length=0".into(),
            )));
        }
        if token.is_some_and(|t| t.is_cancelled()) {
            return resolved(Err(cancelled_error()));
        }

        let last = offset
            .checked_add(length)
            .and_then(|e| e.checked_sub(1))
            .unwrap_or(u64::MAX);
        let mut merged_headers = headers.to_vec();
        merged_headers.push(("Range".to_owned(), format!("bytes={offset}-{last}")));

        let url = url.to_owned();
        let default_headers = Arc::clone(&self.default_headers);
        let token = token.cloned();

        let future = do_fetch(
            "GET".into(),
            url,
            default_headers,
            merged_headers,
            None,
            token,
        );
        let (resolver, task) = orkester::pair();
        wasm_bindgen_futures::spawn_local(async move {
            let mut resp = future.await;
            if let Ok(r) = &mut resp {
                if r.status == 200 {
                    r.status = 206;
                }
            }
            resolver.resolve(resp);
        });
        task
    }
}

//! JSON helpers and generic REST utilities.

use crate::client::Client;
use crate::fetch::{AssetResponse, FetchError, RequestPriority};
use orkester::Task;
use serde::{Serialize, de::DeserializeOwned};

/// Deserialize a response body as JSON.
pub fn parse_json<T: DeserializeOwned>(resp: AssetResponse) -> Result<T, FetchError> {
    serde_json::from_slice(&resp.data).map_err(|e| FetchError::Json(e.to_string()))
}

/// Paginated list that covers both page-number and skip/top pagination styles.
pub struct PagedList<T> {
    pub items: Vec<T>,
    pub next_url: Option<String>,
}

/// Fetch a URL and deserialize the response as JSON.
///
/// Checks HTTP status before deserializing.
pub fn fetch_json<T: Send + DeserializeOwned>(
    client: &impl Client,
    path: &str,
) -> Task<Result<T, FetchError>> {
    let url = format!("{}{}", client.base_url(), path);
    client
        .accessor()
        .get(&url, &[], RequestPriority::NORMAL, None)
        .map(|result| {
            result.and_then(|resp| {
                resp.check_status()?;
                parse_json(resp)
            })
        })
}

/// POST a JSON body and deserialize the response as JSON.
pub fn post_json<B: Serialize, T: Send + DeserializeOwned>(
    client: &impl Client,
    path: &str,
    body: &B,
) -> Task<Result<T, FetchError>> {
    let url = format!("{}{}", client.base_url(), path);
    let body_bytes = match serde_json::to_vec(body) {
        Ok(b) => b,
        Err(e) => return orkester::resolved(Err(FetchError::Json(e.to_string()))),
    };
    let headers = vec![("Content-Type".to_owned(), "application/json".to_owned())];
    client
        .accessor()
        .request(
            "POST",
            &url,
            &headers,
            Some(&body_bytes),
            RequestPriority::NORMAL,
            None,
        )
        .map(|result| {
            result.and_then(|resp| {
                resp.check_status()?;
                parse_json(resp)
            })
        })
}

/// Try to parse an ArcGIS or Ion error response body into a human-readable message.
///
/// ArcGIS error schema: `{ "error": { "message": "…", "code": 400 } }`
/// Ion error schema:    `{ "message": "…" }`
pub fn parse_error_response(body: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    if let Some(msg) = v.pointer("/error/message").and_then(|m| m.as_str()) {
        return Some(msg.to_owned());
    }
    if let Some(msg) = v.get("message").and_then(|m| m.as_str()) {
        return Some(msg.to_owned());
    }
    None
}

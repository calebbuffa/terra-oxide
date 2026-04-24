//! ArcGIS FeatureServer client — vector feature queries.

use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use orkester::Task;
use std::sync::Arc;

/// Spatial filter geometry for feature queries.
#[derive(Debug, Clone)]
pub enum QueryGeometry {
    /// Bounding box `[xmin, ymin, xmax, ymax]`.
    Envelope([f64; 4]),
    /// Point `[x, y]`.
    Point([f64; 2]),
}

impl QueryGeometry {
    fn to_query_params(&self) -> (String, &'static str) {
        match self {
            Self::Envelope([xmin, ymin, xmax, ymax]) => (
                format!("{xmin},{ymin},{xmax},{ymax}"),
                "esriGeometryEnvelope",
            ),
            Self::Point([x, y]) => (format!("{x},{y}"), "esriGeometryPoint"),
        }
    }
}

/// Query parameters for a FeatureServer layer query.
#[derive(Debug, Default, Clone)]
pub struct QueryParams {
    /// SQL WHERE clause (e.g. `"population > 1000"`). Defaults to `1=1`.
    pub where_clause: Option<String>,
    /// Fields to return. `None` -> `*` (all fields).
    pub out_fields: Option<Vec<String>>,
    /// Spatial filter geometry.
    pub geometry: Option<QueryGeometry>,
    /// Spatial reference WKID for the geometry filter.
    pub geometry_sr: Option<u32>,
    /// Pagination offset.
    pub result_offset: Option<u32>,
    /// Maximum number of records to return.
    pub result_record_count: Option<u32>,
}

/// Client for an ArcGIS FeatureServer REST endpoint.
pub struct FeatureServerClient {
    base_url: String,
    accessor: Arc<dyn AssetAccessor>,
}

impl FeatureServerClient {
    pub fn new(base_url: impl Into<String>, accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            accessor,
        }
    }

    /// Query features from a layer, returning a GeoJSON [`serde_json::Value`].
    pub fn query(
        &self,
        layer_id: u32,
        params: QueryParams,
    ) -> Task<Result<serde_json::Value, FetchError>> {
        let url = self.build_query_url(layer_id, &params);
        self.accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    serde_json::from_slice(&resp.data).map_err(|e| FetchError::Json(e.to_string()))
                })
            })
    }

    /// Fetch metadata for a layer.
    pub fn metadata(&self, layer_id: u32) -> Task<Result<serde_json::Value, FetchError>> {
        let url = format!("{}/{layer_id}?f=json", self.base_url);
        self.accessor
            .get(&url, &[], RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    serde_json::from_slice(&resp.data).map_err(|e| FetchError::Json(e.to_string()))
                })
            })
    }

    fn build_query_url(&self, layer_id: u32, params: &QueryParams) -> String {
        let where_clause = params.where_clause.as_deref().unwrap_or("1=1");
        let out_fields = params
            .out_fields
            .as_ref()
            .map(|f| f.join(","))
            .unwrap_or_else(|| "*".to_owned());

        let mut url = format!(
            "{}/{layer_id}/query?where={}&outFields={}&f=geojson",
            self.base_url,
            urlencoding::encode(where_clause),
            out_fields,
        );

        if let Some(geom) = &params.geometry {
            let (geom_str, geom_type) = geom.to_query_params();
            url.push_str(&format!(
                "&geometryType={geom_type}&geometry={}",
                urlencoding::encode(&geom_str)
            ));
            if let Some(sr) = params.geometry_sr {
                url.push_str(&format!("&inSR={sr}"));
            }
        }
        if let Some(offset) = params.result_offset {
            url.push_str(&format!("&resultOffset={offset}"));
        }
        if let Some(count) = params.result_record_count {
            url.push_str(&format!("&resultRecordCount={count}"));
        }

        url
    }
}

impl Client for FeatureServerClient {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}

// Minimal URL encoding for query values — only encode chars that break URLs.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut out = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                ' ' => out.push('+'),
                '&' | '=' | '+' | '%' | '#' => {
                    out.push('%');
                    let b = c as u8;
                    out.push(char::from_digit((b >> 4) as u32, 16).unwrap());
                    out.push(char::from_digit((b & 0xf) as u32, 16).unwrap());
                }
                other => out.push(other),
            }
        }
        out
    }
}

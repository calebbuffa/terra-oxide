//! Bentley iTwin API connection.

use super::types::{IModelMeshExport, ITwin, RealityData};
use crate::client::Client;
use crate::fetch::{AssetAccessor, FetchError, RequestPriority};
use crate::rest::parse_json;
use orkester::Task;
use serde::Deserialize;
use std::sync::Arc;

const DEFAULT_API_URL: &str = "https://api.bentley.com";

#[derive(Deserialize)]
struct ITwinsResponse {
    #[serde(rename = "iTwins")]
    itwins: Vec<ITwin>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RealityDataResponse {
    reality_data: Vec<RealityData>,
}

#[derive(Deserialize)]
struct MeshExportsResponse {
    exports: Vec<IModelMeshExport>,
}

/// Connection to the Bentley iTwin REST API.
///
/// Auth (Bearer token via OAuth2) is handled at the accessor level — inject it
/// via `AuthenticatedAccessor::new(accessor, BearerTokenAuth::new(token))`.
pub struct Connection {
    base_url: String,
    accessor: Arc<dyn AssetAccessor>,
}

impl Connection {
    /// Create a connection with the default iTwin API URL.
    pub fn new(accessor: Arc<dyn AssetAccessor>) -> Self {
        Self {
            base_url: DEFAULT_API_URL.to_owned(),
            accessor,
        }
    }

    /// Override the iTwin API base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into().trim_end_matches('/').to_owned();
        self
    }

    /// List iTwin projects accessible to the authenticated user.
    pub fn itwins(&self) -> Task<Result<Vec<ITwin>, FetchError>> {
        let url = format!("{}/itwins?subClass=Project", self.base_url);
        self.accessor
            .get(&url, &self.json_headers(), RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    let r: ITwinsResponse = parse_json(resp)?;
                    Ok(r.itwins)
                })
            })
    }

    /// List reality data assets for an iTwin project.
    pub fn reality_data(&self, itwin_id: &str) -> Task<Result<Vec<RealityData>, FetchError>> {
        let url = format!(
            "{}/reality-management/reality-data?iTwinId={itwin_id}",
            self.base_url
        );
        self.accessor
            .get(&url, &self.json_headers(), RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    let r: RealityDataResponse = parse_json(resp)?;
                    Ok(r.reality_data)
                })
            })
    }

    /// List mesh export jobs for an iModel.
    pub fn imodel_exports(
        &self,
        imodel_id: &str,
    ) -> Task<Result<Vec<IModelMeshExport>, FetchError>> {
        let url = format!("{}/mesh-export?iModelId={imodel_id}", self.base_url);
        self.accessor
            .get(&url, &self.json_headers(), RequestPriority::NORMAL, None)
            .map(|result| {
                result.and_then(|resp| {
                    resp.check_status()?;
                    let r: MeshExportsResponse = parse_json(resp)?;
                    Ok(r.exports)
                })
            })
    }

    fn json_headers(&self) -> Vec<(String, String)> {
        vec![(
            "Accept".to_owned(),
            "application/vnd.bentley.itwin-client.v1+json".to_owned(),
        )]
    }
}

impl Client for Connection {
    fn base_url(&self) -> &str {
        &self.base_url
    }

    fn accessor(&self) -> &Arc<dyn AssetAccessor> {
        &self.accessor
    }
}

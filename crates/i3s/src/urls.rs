//! URL construction helpers for I3S REST endpoints.
//!
//! All functions take a `base_url` that is the root of the scene layer
//! (i.e., the URL that returns the `SceneLayerInfo` JSON when you
//! append `/layers/0`).  Trailing slashes are normalised.

/// URL for the SceneLayerInfo of layer 0.
///
/// ```
/// # use i3s::layer_url;
/// assert_eq!(layer_url("https://example.com/sls"), "https://example.com/sls/layers/0");
/// ```
pub fn layer_url(base_url: &str) -> String {
    format!("{}/layers/0", base_url.trim_end_matches('/'))
}

/// URL for the NodeIndexDocument page at `page_id`.
///
/// ```
/// # use i3s::node_page_url;
/// assert_eq!(node_page_url("https://example.com/sls", 3), "https://example.com/sls/layers/0/nodepages/3");
/// ```
pub fn node_page_url(base_url: &str, page_id: u64) -> String {
    format!(
        "{}/layers/0/nodepages/{page_id}",
        base_url.trim_end_matches('/')
    )
}

/// URL for a geometry resource.
///
/// `node_id` is the node index (used to derive the node path).
/// `resource` is the value of `MeshGeometry::resource` for the node.
/// `buf_idx` is the index of the chosen `GeometryBuffer` within its
/// `GeometryDefinition`.
///
/// ```
/// # use i3s::geometry_url;
/// assert_eq!(
///     geometry_url("https://example.com/sls", 5, 5, 1),
///     "https://example.com/sls/layers/0/nodes/5/geometries/1"
/// );
/// ```
pub fn geometry_url(base_url: &str, node_id: u64, _resource: u64, buf_idx: usize) -> String {
    format!(
        "{}/layers/0/nodes/{node_id}/geometries/{buf_idx}",
        base_url.trim_end_matches('/')
    )
}

/// URL for an attribute resource.
///
/// `node_id` is the node index.
/// `attribute_index` is the zero-based index of the attribute in
/// `SceneLayerInfo::attribute_storage_info`.
///
/// ```
/// # use i3s::attribute_url;
/// assert_eq!(
///     attribute_url("https://example.com/sls", 5, 0),
///     "https://example.com/sls/layers/0/nodes/5/attributes/f_0/0"
/// );
/// ```
pub fn attribute_url(base_url: &str, node_id: u64, attribute_index: usize) -> String {
    format!(
        "{}/layers/0/nodes/{node_id}/attributes/f_{attribute_index}/0",
        base_url.trim_end_matches('/')
    )
}

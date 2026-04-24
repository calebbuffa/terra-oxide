//! Error types for GeoJSON parsing and loading.

/// Error loading a GeoJSON document (JSON parse or fatal structure error).
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parse error: {0}")]
    Parse(String),
}

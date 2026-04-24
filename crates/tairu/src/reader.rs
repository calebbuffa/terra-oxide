//! [`TilesetReader`] - deserialize a `tileset.json` byte payload into a [`Tileset`].
//!
//! # Example
//!
//! ```no_run
//! use tairu::TilesetReader;
//!
//! let json = br#"{"asset":{"version":"1.1"},"geometricError":0,"root":{"boundingVolume":{"sphere":[0,0,0,1]},"geometricError":0}}"#;
//! match TilesetReader::read_from_slice(json) {
//!     Ok(tileset) => println!("{}", tileset.asset.version),
//!     Err(e) => eprintln!("failed to read tileset: {e}"),
//! }
//! ```

use crate::generated::Tileset;

/// Error returned by [`TilesetReader`].
#[derive(Debug, thiserror::Error)]
pub enum TileParseError {
    /// The JSON payload could not be parsed.
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    /// The tileset failed schema validation.
    #[error("validation error: {0}")]
    Validation(String),
}

/// Parses 3D Tiles `tileset.json` payloads.
pub struct TilesetReader;

impl TilesetReader {
    /// Parse a tileset from a raw JSON byte slice.
    ///
    /// Non-fatal issues (e.g. negative geometric error, extension mismatches) are
    /// emitted as `log::warn!` rather than returned as errors.
    pub fn read_from_slice(data: &[u8]) -> Result<Tileset, TileParseError> {
        let tileset = serde_json::from_slice::<Tileset>(data)?;
        validate(&tileset)?;
        Ok(tileset)
    }

    /// Parse a tileset from a JSON string slice.
    pub fn read_from_str(s: &str) -> Result<Tileset, TileParseError> {
        Self::read_from_slice(s.as_bytes())
    }
}

/// Validate a successfully-parsed [`Tileset`], returning fatal errors and emitting
/// non-fatal issues as warnings.
fn validate(tileset: &Tileset) -> Result<(), TileParseError> {
    if tileset.asset.version.is_empty() {
        return Err(TileParseError::Validation(
            "asset.version is required and must not be empty".into(),
        ));
    }

    if tileset.geometric_error < 0.0 {
        log::warn!(
            "geometricError is negative ({}); expected >= 0",
            tileset.geometric_error
        );
    }

    if tileset.root.geometric_error < 0.0 {
        log::warn!(
            "root.geometricError is negative ({}); expected >= 0",
            tileset.root.geometric_error
        );
    }

    for ext in &tileset.extensions_required {
        if !tileset.extensions_used.contains(ext) {
            log::warn!(
                "extensionsRequired contains '{}' which is not listed in extensionsUsed",
                ext
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_json() -> &'static [u8] {
        br#"{
            "asset": { "version": "1.1" },
            "geometricError": 100.0,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 1000] },
                "geometricError": 100.0,
                "refine": "ADD"
            }
        }"#
    }

    #[test]
    fn parses_minimal_tileset() {
        let ts = TilesetReader::read_from_slice(minimal_json()).expect("should parse");
        assert_eq!(ts.asset.version, "1.1");
        assert_eq!(ts.geometric_error, 100.0);
    }

    #[test]
    fn warns_on_negative_geometric_error() {
        // Non-fatal warning; parsing should still succeed.
        let json = br#"{
            "asset": { "version": "1.1" },
            "geometricError": -1.0,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 1] },
                "geometricError": 0.0
            }
        }"#;
        assert!(TilesetReader::read_from_slice(json).is_ok());
    }

    #[test]
    fn errors_on_invalid_json() {
        let err = TilesetReader::read_from_slice(b"not json");
        assert!(matches!(err, Err(TileParseError::Json(_))));
    }

    #[test]
    fn errors_on_empty_version() {
        let json = br#"{
            "asset": { "version": "" },
            "geometricError": 0.0,
            "root": {
                "boundingVolume": { "sphere": [0, 0, 0, 1] },
                "geometricError": 0.0
            }
        }"#;
        let err = TilesetReader::read_from_slice(json);
        assert!(matches!(err, Err(TileParseError::Validation(_))));
    }
}

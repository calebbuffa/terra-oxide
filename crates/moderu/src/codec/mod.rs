//! Codec implementations for glTF 2.0 models.
//!
//! Two levels of API:
//!
//! ### Buffer-level (format-agnostic)
//! Operate directly on raw bytes - no glTF model required.
//!
//! | Function | Input | Output |
//! |---|---|---|
//! | [`draco::decode_buffer`] | compressed bytes + attr id map | [`draco::DecodedMesh`] |
//! | [`meshopt::decode_vertex_buffer`] | compressed bytes | `Vec<u8>` |
//! | [`meshopt::decode_index_buffer`] | compressed bytes | `Vec<u32>` |
//! | [`spz::decode_buffer`] | SPZ bytes | [`spz::DecodedSplats`] |
//! | [`ktx2::decode_buffer`] | KTX2 bytes | `moderu::ImageData` |
//! | [`image::decode_buffer`] | PNG/JPEG/WebP bytes | `moderu::ImageData` |
//!
//! ### Model-level via [`CodecRegistry`]
//! Register decoders/encoders and run them all at once:
//!
//! ```ignore
//! // Default registry: all built-in codecs enabled by feature flags.
//! let mut reg = CodecRegistry::default();
//!
//! // Or start empty and register what you want:
//! let mut reg = CodecRegistry::empty();
//! reg.register_decoder::<draco::DracoDecoder>()
//!    .register_decoder::<MyCustomDecoder>();
//!
//! let warnings = reg.decode_all(&mut model);
//! ```
//!
//! ### Implementing a custom codec
//! ```ignore
//! struct MyDecoder;
//! impl CodecDecoder for MyDecoder {
//!     const EXT_NAME: &'static str = "MY_custom_compression";
//!     type Error = MyError;
//!     fn decode_primitive(model, mesh_idx, prim_idx, ext) -> Result<(), MyError> { ... }
//! }
//! registry.register_decoder::<MyDecoder>();
//! ```

mod pipeline;
use std::marker::PhantomData;

use super::GltfModel;

pub use pipeline::decode_model;

/// Result of checking if a codec is applicable to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplicabilityResult {
    /// Codec can decode this model; proceed with decoding.
    Applicable,
    /// Codec cannot decode this model; skip it.
    NotApplicable,
}

impl From<bool> for ApplicabilityResult {
    fn from(b: bool) -> Self {
        if b {
            Self::Applicable
        } else {
            Self::NotApplicable
        }
    }
}

/// Trait for standardized codec decompression.
///
/// Implement this trait to add a custom decoder that integrates with
/// [`CodecRegistry`].
pub trait CodecDecoder: Sized + Send + Sync + 'static {
    /// The registered glTF extension name (e.g. `"KHR_draco_mesh_compression"`).
    const EXT_NAME: &'static str;

    /// The error type returned by decode operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Check if this codec can decode the model.
    ///
    /// Default: checks `model.extensions_used` for `EXT_NAME`.
    fn can_decode(model: &GltfModel) -> ApplicabilityResult {
        model
            .extensions_used
            .iter()
            .any(|e| e == Self::EXT_NAME)
            .into()
    }

    /// Decode the contents of a mesh primitive in-place.
    fn decode_primitive(
        model: &mut GltfModel,
        mesh_idx: usize,
        prim_idx: usize,
        ext: &serde_json::Value,
    ) -> Result<(), Self::Error> {
        let _ = (model, mesh_idx, prim_idx, ext);
        Ok(())
    }

    /// Decode the contents of a buffer view in-place.
    fn decode_view(
        model: &mut GltfModel,
        bv_idx: usize,
        ext: &serde_json::Value,
    ) -> Result<(), Self::Error> {
        let _ = (model, bv_idx, ext);
        Ok(())
    }

    /// Full model-level decode. Default: iterates primitives via
    /// [`decode_primitives`] then buffer views via [`decode_buffer_views`].
    /// Override for codecs with custom dispatch logic (e.g. prefix-matched
    /// extension names).
    fn decode_model(model: &mut GltfModel) -> Vec<String> {
        let mut w = decode_primitives::<Self>(model);
        w.extend(decode_buffer_views::<Self>(model));
        w
    }
}

/// Trait for standardized codec compression.
pub trait CodecEncoder: Sized + Send + Sync + 'static {
    /// The registered glTF extension name (e.g. `"KHR_draco_mesh_compression"`).
    const EXT_NAME: &'static str;

    /// The error type returned by encode operations.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Encode the model using this codec.
    fn encode_model(model: &mut GltfModel) -> Result<(), Self::Error> {
        let _ = model;
        Ok(())
    }
}

/// Run `C::decode_primitive` over all mesh primitives that carry the extension.
pub fn decode_primitives<C: CodecDecoder>(model: &mut GltfModel) -> Vec<String> {
    if C::can_decode(model) == ApplicabilityResult::NotApplicable {
        return vec![];
    }
    let mut warnings = Vec::new();
    for mesh_idx in 0..model.meshes.len() {
        for prim_idx in 0..model.meshes[mesh_idx].primitives.len() {
            let ext_value = model.meshes[mesh_idx].primitives[prim_idx]
                .extensions
                .get(C::EXT_NAME)
                .cloned();
            let Some(ext) = ext_value else { continue };
            if let Err(e) = C::decode_primitive(model, mesh_idx, prim_idx, &ext) {
                warnings.push(format!(
                    "mesh[{mesh_idx}].primitive[{prim_idx}] {}: {e}",
                    C::EXT_NAME
                ));
            }
        }
    }
    warnings
}

/// Run `C::decode_view` over all buffer views that carry the extension.
pub fn decode_buffer_views<C: CodecDecoder>(model: &mut GltfModel) -> Vec<String> {
    if C::can_decode(model) == ApplicabilityResult::NotApplicable {
        return vec![];
    }
    let mut warnings = Vec::new();
    for bv_idx in 0..model.buffer_views.len() {
        let ext_value = model.buffer_views[bv_idx]
            .extensions
            .get(C::EXT_NAME)
            .cloned();
        let Some(ext) = ext_value else { continue };
        if let Err(e) = C::decode_view(model, bv_idx, &ext) {
            warnings.push(format!("bufferView[{bv_idx}] {}: {e}", C::EXT_NAME));
        }
    }
    warnings
}

trait ErasedDecoder: Send + Sync {
    fn ext_name(&self) -> &'static str;
    fn decode(&self, model: &mut GltfModel) -> Vec<String>;
}

trait ErasedEncoder: Send + Sync {
    fn ext_name(&self) -> &'static str;
    fn encode(&self, model: &mut GltfModel) -> Result<(), String>;
}

struct DecoderWrapper<C: CodecDecoder>(PhantomData<C>);
struct EncoderWrapper<C: CodecEncoder>(PhantomData<C>);

// SAFETY: PhantomData<C> is zero-sized; C: Send+Sync is required by the bounds.
unsafe impl<C: CodecDecoder> Send for DecoderWrapper<C> {}
unsafe impl<C: CodecDecoder> Sync for DecoderWrapper<C> {}
unsafe impl<C: CodecEncoder> Send for EncoderWrapper<C> {}
unsafe impl<C: CodecEncoder> Sync for EncoderWrapper<C> {}

impl<C: CodecDecoder> ErasedDecoder for DecoderWrapper<C> {
    fn ext_name(&self) -> &'static str {
        C::EXT_NAME
    }
    fn decode(&self, model: &mut GltfModel) -> Vec<String> {
        C::decode_model(model)
    }
}

impl<C: CodecEncoder> ErasedEncoder for EncoderWrapper<C> {
    fn ext_name(&self) -> &'static str {
        C::EXT_NAME
    }
    fn encode(&self, model: &mut GltfModel) -> Result<(), String> {
        C::encode_model(model).map_err(|e| e.to_string())
    }
}

/// Registry of decoders and encoders.
///
/// Use [`CodecRegistry::default()`] to get all built-in codecs (controlled by
/// feature flags), or [`CodecRegistry::empty()`] to opt-in manually.
///
/// Decoders and encoders run in insertion order.
pub struct CodecRegistry {
    decoders: Vec<Box<dyn ErasedDecoder>>,
    encoders: Vec<Box<dyn ErasedEncoder>>,
}

impl CodecRegistry {
    /// Empty registry - no codecs registered.
    pub fn empty() -> Self {
        Self {
            decoders: vec![],
            encoders: vec![],
        }
    }

    /// Registry pre-populated with all built-in codecs enabled at compile time.
    ///
    /// The order matches the correct decode sequence:
    /// image -> draco -> meshopt -> spz -> ktx2
    pub fn with_defaults() -> Self {
        let mut r = Self::empty();
        // image must run before ktx2 (populates pixel buffers from data URIs)
        #[cfg(feature = "image")]
        r.register_decoder::<image::ImageDecoder>();
        #[cfg(feature = "draco")]
        r.register_decoder::<draco::DracoDecoder>();
        #[cfg(feature = "meshopt")]
        r.register_decoder::<meshopt::MeshoptDecoder>();
        #[cfg(feature = "spz")]
        r.register_decoder::<spz::SpzDecoder>();
        #[cfg(feature = "ktx2")]
        r.register_decoder::<ktx2::Ktx2Decoder>();

        // Encoders (all disabled / no-op by default)
        #[cfg(feature = "draco")]
        r.register_encoder::<draco::DracoEncoder>();
        #[cfg(feature = "meshopt")]
        r.register_encoder::<meshopt::MeshoptEncoder>();
        #[cfg(feature = "spz")]
        r.register_encoder::<spz::SpzEncoder>();
        #[cfg(feature = "ktx2")]
        r.register_encoder::<ktx2::Ktx2Encoder>();
        #[cfg(feature = "image")]
        r.register_encoder::<image::ImageEncoder>();
        r
    }

    /// Register a custom decoder. Returns `&mut self` for chaining.
    pub fn register_decoder<C: CodecDecoder>(&mut self) -> &mut Self {
        self.decoders
            .push(Box::new(DecoderWrapper::<C>(PhantomData)));
        self
    }

    /// Register a custom encoder. Returns `&mut self` for chaining.
    pub fn register_encoder<C: CodecEncoder>(&mut self) -> &mut Self {
        self.encoders
            .push(Box::new(EncoderWrapper::<C>(PhantomData)));
        self
    }

    /// Remove a decoder by extension name. Returns `true` if it was present.
    pub fn remove_decoder(&mut self, ext_name: &str) -> bool {
        let before = self.decoders.len();
        self.decoders.retain(|d| d.ext_name() != ext_name);
        self.decoders.len() != before
    }

    /// Remove an encoder by extension name. Returns `true` if it was present.
    pub fn remove_encoder(&mut self, ext_name: &str) -> bool {
        let before = self.encoders.len();
        self.encoders.retain(|e| e.ext_name() != ext_name);
        self.encoders.len() != before
    }

    /// Run all registered decoders in insertion order. Collects non-fatal
    /// warnings from each decoder.
    pub fn decode_all(&self, model: &mut GltfModel) -> Vec<String> {
        let mut warnings = Vec::new();
        for decoder in &self.decoders {
            warnings.extend(decoder.decode(model));
        }
        warnings
    }

    /// Run all registered encoders in insertion order.
    ///
    /// Returns the first fatal error, if any. Non-fatal warnings are not
    /// currently surfaced from encoders (they return `Result`).
    pub fn encode_all(&self, model: &mut GltfModel) -> Result<(), String> {
        for encoder in &self.encoders {
            encoder.encode(model)?;
        }
        Ok(())
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

impl std::fmt::Debug for CodecRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodecRegistry")
            .field(
                "decoders",
                &self
                    .decoders
                    .iter()
                    .map(|d| d.ext_name())
                    .collect::<Vec<_>>(),
            )
            .field(
                "encoders",
                &self
                    .encoders
                    .iter()
                    .map(|e| e.ext_name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

#[cfg(feature = "draco")]
pub mod draco;

#[cfg(feature = "meshopt")]
pub mod meshopt;

#[cfg(feature = "spz")]
pub mod spz;

#[cfg(feature = "ktx2")]
pub mod ktx2;

#[cfg(feature = "image")]
pub mod image;

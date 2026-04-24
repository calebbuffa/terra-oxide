//! glTF 2.0 reader with full codec pipeline.
//!
//! Parses GLB / glTF JSON and runs every post-processing step:
//!
//! 1. **Data URI decoding** - `data:` buffer/image URIs -> raw bytes
//! 2. **Image decoding** - PNG / JPEG / WebP -> `Image` pixels *(feature: `image`)*
//! 3. **Draco** - `KHR_draco_mesh_compression` *(feature: `draco`)*
//! 4. **meshopt** - `EXT_meshopt_compression` *(feature: `meshopt`)*
//! 5. **SPZ** - `KHR_gaussian_splatting_compression_spz` *(feature: `spz`)*
//! 6. **Dequantization** - `KHR_mesh_quantization`
//! 7. **Texture transform** - `KHR_texture_transform`
//!
//! All codecs are behind feature flags so you only link what you need.
//!
//! ## Quick start
//!
//! ```ignore
//! let model = GltfReader::default().read_file("model.glb")?.model;
//! for img in &model.images {
//!     upload_to_gpu(img.pixels.data.as_slice(), img.pixels.width, img.pixels.height);
//! }
//! ```
mod error;
mod external_refs;
mod glb;
mod image_ops;
mod uri;

mod dequantize;
mod khr_texture_transform;

pub use dequantize::dequantize;
pub use error::{GltfError, Warning, Warnings};
pub use external_refs::resolve_external_refs;
pub use image_ops::blit_image;
pub use khr_texture_transform::apply_texture_transforms;
pub use uri::decode_data_urls;

use crate::GltfModel;

/// Image processing options for data URIs and embedded images.
#[derive(Clone, Debug)]
pub struct ImageProcessingOptions {
    /// Resolve external (non-`data:`) buffer and image URIs from the filesystem.
    ///
    /// Requires `base_path` to be set. Automatically set to `true` when using
    /// [`GltfReader::read_file`].
    pub resolve_external_references: bool,
    /// Base directory used to resolve relative URIs when
    /// `resolve_external_references` is `true`. Automatically derived from the
    /// file path when using [`GltfReader::read_file`].
    pub base_path: Option<std::path::PathBuf>,
    /// Decode `data:` URIs in buffers and images.
    pub decode_data_urls: bool,
    /// Clear the URI string after decoding a data URL (saves memory).
    pub clear_decoded_data_urls: bool,
    /// Decode embedded images (PNG / JPEG / WebP) to pixel data.
    pub decode_embedded_images: bool,
}

impl Default for ImageProcessingOptions {
    fn default() -> Self {
        Self {
            resolve_external_references: true,
            base_path: None,
            decode_data_urls: true,
            clear_decoded_data_urls: true,
            decode_embedded_images: true,
        }
    }
}

/// Mesh codec decompression options.
///
/// These boolean flags have been replaced by [`moderu::codec::CodecRegistry`].
/// Use [`GltfReader::codec_registry`] to control which codecs run.
///
/// This struct is retained only for the non-codec processing options
/// (`dequantize`, `apply_texture_transform`).
#[derive(Clone, Debug)]
pub struct MeshProcessingOptions {
    /// Dequantize `KHR_mesh_quantization` attributes to float.
    pub dequantize: bool,
    /// Apply `KHR_texture_transform` to UV coordinates.
    pub apply_texture_transform: bool,
}

impl Default for MeshProcessingOptions {
    fn default() -> Self {
        Self {
            dequantize: true,
            apply_texture_transform: true,
        }
    }
}

/// Options controlling which post-processing steps run.
///
/// Codec selection is now handled by [`GltfReader::codec_registry`].
/// This struct covers image processing and mesh post-processing.
#[derive(Clone, Debug)]
pub struct GltfReaderOptions {
    /// Image processing settings (data URIs, embedded images).
    pub images: ImageProcessingOptions,
    /// Mesh post-processing settings (quantization, texture transforms).
    pub mesh: MeshProcessingOptions,
}

impl Default for GltfReaderOptions {
    fn default() -> Self {
        Self {
            images: ImageProcessingOptions::default(),
            mesh: MeshProcessingOptions::default(),
        }
    }
}

impl GltfReaderOptions {
    /// All post-processing disabled - parse JSON/GLB only. No codecs.
    pub fn minimal() -> Self {
        Self {
            images: ImageProcessingOptions {
                resolve_external_references: false,
                base_path: None,
                decode_data_urls: false,
                clear_decoded_data_urls: false,
                decode_embedded_images: false,
            },
            mesh: MeshProcessingOptions {
                dequantize: false,
                apply_texture_transform: false,
            },
        }
    }
}

/// glTF 2.0 reader. Parses GLB or glTF JSON and runs the codec pipeline.
#[derive(Debug)]
pub struct GltfReader {
    pub options: GltfReaderOptions,
}

impl Default for GltfReader {
    fn default() -> Self {
        Self {
            options: GltfReaderOptions::default(),
        }
    }
}

impl GltfReader {
    pub fn new(options: GltfReaderOptions) -> Self {
        Self { options }
    }

    /// Read from a file path. Detects GLB vs glTF JSON automatically.
    ///
    /// Automatically sets `options.images.base_path` to the file's parent
    /// directory so that external buffer and image URIs can be resolved.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<GltfModel, GltfError> {
        let path = path.as_ref();
        let data = std::fs::read(path)?;
        self.read_bytes(&data)
    }

    /// Parse binary GLB or JSON glTF from raw bytes.
    ///
    /// Automatically detects GLB by the magic header. Falls back to JSON parse.
    pub fn read_bytes(&self, data: &[u8]) -> Result<GltfModel, GltfError> {
        if glb::is_glb(data) {
            self.parse_glb(data)
        } else {
            self.parse_json(data)
        }
    }

    /// Parse a binary GLB container from raw bytes.
    fn parse_glb(&self, data: &[u8]) -> Result<GltfModel, GltfError> {
        let (mut model, bin_chunk) = glb::parse_glb(data)?;
        if let Some(bin) = bin_chunk {
            if let Some(b) = model.buffers.first_mut() {
                b.data = bin.to_vec();
            }
        }

        Ok(model)
    }

    /// Parse glTF JSON from raw bytes.
    fn parse_json(&self, data: &[u8]) -> Result<GltfModel, GltfError> {
        let model = serde_json::from_slice::<GltfModel>(data).map_err(GltfError::JsonParse)?;
        Ok(model)
    }
}

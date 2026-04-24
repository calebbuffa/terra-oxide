//! Main glTF writer API.

use super::error::WriteResult;
use super::glb::{GlbHeader, write_bin_chunk, write_json_chunk};
use crate::GltfModel;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;

/// Options for how to write a glTF (formatting, alignment, etc.).
///
/// Codec selection for encoding is handled by [`GltfWriter::codec_registry`].
#[derive(Clone, Debug)]
pub struct GltfWriterOptions {
    /// Save as glb
    pub glb: bool,

    /// Enable pretty-printing JSON output.
    pub pretty_print: bool,

    /// Byte alignment for GLB chunks.
    ///
    /// The GLB spec requires 4-byte alignment. Some extensions (e.g.
    /// `EXT_mesh_features`) require 8-byte alignment. Defaults to `4`.
    pub binary_chunk_byte_alignment: usize,
}

impl Default for GltfWriterOptions {
    fn default() -> Self {
        GltfWriterOptions {
            glb: false,
            pretty_print: true,
            binary_chunk_byte_alignment: 4,
        }
    }
}

/// Main glTF writer for saving models to JSON and GLB formats.
///
/// # Encoding
/// The writer owns a [`moderu::codec::CodecRegistry`] that is run by
/// [`GltfWriter::encode`]. By default the registry is *empty* (no encoding) -
/// encoding is always opt-in. Register encoders explicitly:
///
/// ```ignore
/// use moderu::{GltfWriter, GltfWriterOptions};
/// let options = GltfWriterOptions::default();
/// let writer = GltfWriter::new(options);
/// let result = writer.write(&mut model);
/// ```
pub struct GltfWriter {
    pub options: GltfWriterOptions,
}

impl Default for GltfWriter {
    fn default() -> Self {
        Self {
            options: GltfWriterOptions::default(),
        }
    }
}

impl GltfWriter {
    pub fn new(options: GltfWriterOptions) -> Self {
        Self { options }
    }

    /// Run all registered encoders against the model in-place.
    ///
    /// # Errors
    /// Returns `WriteError::Codec` if any encoder fails.
    pub fn write(&self, model: &GltfModel) -> WriteResult<Vec<u8>> {
        let json_str = if self.options.pretty_print {
            serde_json::to_string_pretty(model)?
        } else {
            serde_json::to_string(model)?
        };
        if !self.options.glb {
            let json_bytes = json_str.as_bytes();
            return Ok(json_bytes.to_vec());
        } else {
            let mut glb = Vec::new();
            self.write_to(model, &mut glb)?;
            return Ok(glb);
        }
    }

    pub fn write_to<W: Write>(&self, model: &GltfModel, mut writer: W) -> WriteResult<()> {
        let json_str = if self.options.pretty_print {
            serde_json::to_string_pretty(model)?
        } else {
            serde_json::to_string(model)?
        };
        if !self.options.glb {
            let json_bytes = json_str.as_bytes();
            writer.write_all(json_bytes)?;
            return Ok(());
        } else {
            self.write_glb_to(model, writer)?;
            return Ok(());
        }
    }

    fn write_glb_to<W: Write>(&self, model: &GltfModel, mut writer: W) -> WriteResult<()> {
        // Use the first runtime buffer as the GLB BIN chunk.
        let bin_data: &[u8] = model
            .buffers
            .first()
            .map(|b| b.data.as_slice())
            .unwrap_or(&[]);

        // Calculate file length: 12 (header) + json chunk + bin chunk
        let align = self.options.binary_chunk_byte_alignment.max(1);
        let json_str = if self.options.pretty_print {
            serde_json::to_string_pretty(model)?
        } else {
            serde_json::to_string(model)?
        };
        let json_bytes = json_str.as_bytes();
        let json_chunk_size = ((json_bytes.len() + align - 1) / align) * align + 8; // padded + header
        let bin_chunk_size = if !bin_data.is_empty() {
            ((bin_data.len() + align - 1) / align) * align + 8 // padded + header
        } else {
            0
        };
        let total_length = 12 + json_chunk_size + bin_chunk_size;

        // Write GLB header
        let header = GlbHeader::new(total_length as u32);
        header.write(&mut writer)?;

        // Write JSON chunk
        write_json_chunk(&mut writer, json_bytes, align)?;

        // Write binary chunk if present
        if !bin_data.is_empty() {
            write_bin_chunk(&mut writer, bin_data, align)?;
        }

        Ok(())
    }
}

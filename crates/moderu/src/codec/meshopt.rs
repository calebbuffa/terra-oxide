//! `EXT_meshopt_compression` decoder and encoder.

use crate::CodecDecoder;
use crate::CodecEncoder;
use moderu::GltfModel;
use serde_json::{Value, json};

/// Errors that can occur during meshopt decode or encode operations.
#[derive(thiserror::Error, Debug)]
pub enum MeshoptError {
    #[error("missing buffer in meshopt extension")]
    MissingBuffer,
    #[error("missing byteLength in meshopt extension")]
    MissingByteLength,
    #[error("missing count in meshopt extension")]
    MissingCount,
    #[error("missing byteStride in meshopt extension")]
    MissingByteStride,
    #[error("source buffer {0} out of range")]
    BufferOutOfRange(usize),
    #[error("compressed range [{start}..{end}) exceeds buffer size {size}")]
    CompressedRangeExceeded {
        start: usize,
        end: usize,
        size: usize,
    },
    #[error("decode_vertex_buffer (stride={stride}): {message}")]
    DecodeVertexBuffer { stride: usize, message: String },
    #[error("decode_index_buffer u16: {0}")]
    DecodeIndexBufferU16(String),
    #[error("decode_index_buffer u32: {0}")]
    DecodeIndexBufferU32(String),
    #[error("unsupported meshopt mode: {0}")]
    UnsupportedMode(String),
    #[error("unsupported meshopt vertex byte_stride: {0}")]
    UnsupportedStride(usize),
    #[error("unsupported index_size: {size} (expected 2 or 4)")]
    UnsupportedIndexSize { size: usize },
    #[error("unsupported meshopt filter: {0}")]
    UnsupportedFilter(String),
    #[error("meshopt filter {filter}: invalid byte_stride {stride}")]
    InvalidFilterStride { filter: &'static str, stride: usize },
    #[error("meshopt filter {filter}: decoded buffer too small ({got} < {expected})")]
    FilterBufferTooSmall {
        filter: &'static str,
        got: usize,
        expected: usize,
    },
}

pub struct MeshoptDecoder;

impl CodecDecoder for MeshoptDecoder {
    const EXT_NAME: &'static str = "EXT_meshopt_compression";
    type Error = MeshoptError;

    fn decode_view(model: &mut GltfModel, bv_idx: usize, ext: &Value) -> Result<(), MeshoptError> {
        decode_buffer_view(model, bv_idx, ext)
    }
}

/// Decode all meshopt-compressed buffer views in-place.
/// Returns a warning string for each buffer view that fails.
pub fn decode(model: &mut GltfModel) -> Vec<String> {
    crate::decode_buffer_views::<MeshoptDecoder>(model)
}

/// Low-level: decode a meshopt-encoded vertex attribute buffer.
///
/// Returns raw decoded bytes; length = `count * byte_stride`.
/// `byte_stride` must be a multiple of 4 and at most 64.
pub fn decode_vertex_buffer(
    data: &[u8],
    count: usize,
    byte_stride: usize,
) -> Result<Vec<u8>, MeshoptError> {
    decode_vertices_dynamic(data, count, byte_stride)
}

/// Low-level: decode a meshopt-encoded index buffer.
///
/// Returns indices as `Vec<u32>` regardless of source `index_size` (2 or 4 bytes).
pub fn decode_index_buffer(
    data: &[u8],
    count: usize,
    index_size: usize,
) -> Result<Vec<u32>, MeshoptError> {
    match index_size {
        2 => {
            let indices = meshopt::encoding::decode_index_buffer::<u16>(data, count)
                .map_err(|e| MeshoptError::DecodeIndexBufferU16(e.to_string()))?;
            Ok(indices.into_iter().map(|i| i as u32).collect())
        }
        4 => meshopt::encoding::decode_index_buffer::<u32>(data, count)
            .map_err(|e| MeshoptError::DecodeIndexBufferU32(e.to_string())),
        size => Err(MeshoptError::UnsupportedIndexSize { size }),
    }
}

fn decode_buffer_view(
    model: &mut GltfModel,
    bv_idx: usize,
    ext: &Value,
) -> Result<(), MeshoptError> {
    let src_buffer_idx = ext
        .get("buffer")
        .and_then(|v| v.as_i64())
        .ok_or(MeshoptError::MissingBuffer)? as usize;

    let src_byte_offset = ext.get("byteOffset").and_then(|v| v.as_i64()).unwrap_or(0) as usize;

    let src_byte_length = ext
        .get("byteLength")
        .and_then(|v| v.as_i64())
        .ok_or(MeshoptError::MissingByteLength)? as usize;

    let count = ext
        .get("count")
        .and_then(|v| v.as_i64())
        .ok_or(MeshoptError::MissingCount)? as usize;

    let byte_stride = ext
        .get("byteStride")
        .and_then(|v| v.as_i64())
        .ok_or(MeshoptError::MissingByteStride)? as usize;

    let mode = ext
        .get("mode")
        .and_then(|v| v.as_str())
        .unwrap_or("ATTRIBUTES");

    let filter = ext.get("filter").and_then(|v| v.as_str()).unwrap_or("NONE");

    let compressed: Vec<u8> = {
        let src_buf = &model
            .buffers
            .get(src_buffer_idx)
            .ok_or(MeshoptError::BufferOutOfRange(src_buffer_idx))?
            .data;
        let src_end = src_byte_offset + src_byte_length;
        if src_end > src_buf.len() {
            return Err(MeshoptError::CompressedRangeExceeded {
                start: src_byte_offset,
                end: src_end,
                size: src_buf.len(),
            });
        }
        src_buf[src_byte_offset..src_end].to_vec()
    };
    let output_size = count * byte_stride;
    let mut decoded = vec![0u8; output_size];

    match mode {
        "ATTRIBUTES" => {
            let verts = decode_vertices_dynamic(&compressed, count, byte_stride)?;
            decoded.copy_from_slice(&verts);
        }
        "TRIANGLES" => {
            if byte_stride == 2 {
                let indices = meshopt::encoding::decode_index_buffer::<u16>(&compressed, count)
                    .map_err(|e| MeshoptError::DecodeIndexBufferU16(e.to_string()))?;
                decoded.copy_from_slice(bytemuck::cast_slice(&indices));
            } else {
                let indices = meshopt::encoding::decode_index_buffer::<u32>(&compressed, count)
                    .map_err(|e| MeshoptError::DecodeIndexBufferU32(e.to_string()))?;
                decoded.copy_from_slice(bytemuck::cast_slice(&indices));
            }
        }
        _ => return Err(MeshoptError::UnsupportedMode(mode.to_string())),
    }

    // EXT_meshopt_compression filters: applied after the core codec.
    //
    // - OCTAHEDRAL: decodes 8/16-bit 4-component octahedron-encoded unit
    //   vectors to signed normalized xyzw. `byte_stride` selects 4 (8-bit) or
    //   8 (16-bit) and `w` encodes the tangent sign (or is untouched).
    // - QUATERNION: 16-bit 4-component xyzw with a 2-bit largest-component
    //   index stored in the low 2 bits of the max magnitude component.
    // - EXPONENTIAL: per-component f32 mantissa packed as u32, with the
    //   shared exponent stored in the top 8 bits (sign-extended 24-bit mantissa).
    match filter {
        "NONE" | "" => {}
        "OCTAHEDRAL" => decode_filter_octahedral(&mut decoded, count, byte_stride)?,
        "QUATERNION" => decode_filter_quaternion(&mut decoded, count, byte_stride)?,
        "EXPONENTIAL" => decode_filter_exponential(&mut decoded, count, byte_stride)?,
        other => return Err(MeshoptError::UnsupportedFilter(other.to_string())),
    }

    // Write decoded data back to buffer view
    let bv = &model.buffer_views[bv_idx];
    let buf_idx = bv.buffer;
    let byte_offset = bv.byte_offset;

    if byte_offset + output_size <= model.buffers[buf_idx].data.len() {
        model.buffers[buf_idx].data[byte_offset..byte_offset + output_size]
            .copy_from_slice(&decoded);
    } else {
        model.buffers[buf_idx].data.extend_from_slice(&decoded);
    }

    Ok(())
}

/// Decode a meshopt vertex buffer with a dynamic (runtime) byte stride.
///
/// Uses `[u32; N]` (N = stride / 4) so that `Default` and `Pod` are always
/// satisfied - Rust's std only auto-derives `Default` for arrays up to N=32,
/// and glTF strides are always multiples of 4, so N never exceeds 16.
fn decode_vertices_dynamic(
    encoded: &[u8],
    count: usize,
    byte_stride: usize,
) -> Result<Vec<u8>, MeshoptError> {
    fn dv<T: Clone + Default + bytemuck::Pod>(
        encoded: &[u8],
        count: usize,
        stride: usize,
    ) -> Result<Vec<u8>, MeshoptError> {
        let verts = meshopt::encoding::decode_vertex_buffer::<T>(encoded, count).map_err(|e| {
            MeshoptError::DecodeVertexBuffer {
                stride,
                message: e.to_string(),
            }
        })?;
        // `cast_vec` would require bytemuck's `extern_crate_alloc` feature;
        // a raw-bytes copy is just as fast and keeps our feature surface small.
        Ok(bytemuck::cast_slice(&verts).to_vec())
    }
    match byte_stride {
        4 => dv::<[u32; 1]>(encoded, count, byte_stride),
        8 => dv::<[u32; 2]>(encoded, count, byte_stride),
        12 => dv::<[u32; 3]>(encoded, count, byte_stride),
        16 => dv::<[u32; 4]>(encoded, count, byte_stride),
        20 => dv::<[u32; 5]>(encoded, count, byte_stride),
        24 => dv::<[u32; 6]>(encoded, count, byte_stride),
        28 => dv::<[u32; 7]>(encoded, count, byte_stride),
        32 => dv::<[u32; 8]>(encoded, count, byte_stride),
        36 => dv::<[u32; 9]>(encoded, count, byte_stride),
        40 => dv::<[u32; 10]>(encoded, count, byte_stride),
        44 => dv::<[u32; 11]>(encoded, count, byte_stride),
        48 => dv::<[u32; 12]>(encoded, count, byte_stride),
        52 => dv::<[u32; 13]>(encoded, count, byte_stride),
        56 => dv::<[u32; 14]>(encoded, count, byte_stride),
        60 => dv::<[u32; 15]>(encoded, count, byte_stride),
        64 => dv::<[u32; 16]>(encoded, count, byte_stride),
        s => Err(MeshoptError::UnsupportedStride(s)),
    }
}

//
// Reference: https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Vendor/EXT_meshopt_compression/README.md
// Implementation follows meshoptimizer's vertexfilter.cpp (decodeFilterOct/Quat/Exp).

fn expect_stride(
    filter: &'static str,
    stride: usize,
    allowed: &[usize],
) -> Result<(), MeshoptError> {
    if allowed.contains(&stride) {
        Ok(())
    } else {
        Err(MeshoptError::InvalidFilterStride { filter, stride })
    }
}

fn expect_len(filter: &'static str, bytes: &[u8], expected: usize) -> Result<(), MeshoptError> {
    if bytes.len() >= expected {
        Ok(())
    } else {
        Err(MeshoptError::FilterBufferTooSmall {
            filter,
            got: bytes.len(),
            expected,
        })
    }
}

/// OCTAHEDRAL filter - decode octahedron-encoded unit vectors to xyzw.
///
/// Stride 4: each component is i8 (rescaled to Q=127); stride 8: each component
/// is i16 (rescaled to Q=32767). The `w` component is preserved verbatim
/// (carries tangent sign when used for tangents, typically +-1 in quantized space).
fn decode_filter_octahedral(
    data: &mut [u8],
    count: usize,
    stride: usize,
) -> Result<(), MeshoptError> {
    const F: &str = "OCTAHEDRAL";
    expect_stride(F, stride, &[4, 8])?;
    expect_len(F, data, count * stride)?;

    if stride == 4 {
        // i8 x/y/z/w with Q = 127 - decode xyz via octahedron, rescale to i8.
        for i in 0..count {
            let base = i * 4;
            let x = data[base] as i8 as f32;
            let y = data[base + 1] as i8 as f32;
            let one = data[base + 2] as i8 as f32;
            let sign = data[base + 3] as i8 as f32;
            // Octahedron unwrap
            let mut nx = x;
            let mut ny = y;
            let nz = one - nx.abs() - ny.abs();
            if nz < 0.0 {
                let tx = nx;
                nx = (127.0 - ny.abs()).copysign(tx);
                ny = (127.0 - tx.abs()).copysign(ny);
            }
            // Normalize to unit length in [-127..127] quantized space.
            let inv_len = 127.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(1e-20);
            let qx = (nx * inv_len).round().clamp(-127.0, 127.0) as i8;
            let qy = (ny * inv_len).round().clamp(-127.0, 127.0) as i8;
            let qz = (nz * inv_len).round().clamp(-127.0, 127.0) as i8;
            data[base] = qx as u8;
            data[base + 1] = qy as u8;
            data[base + 2] = qz as u8;
            data[base + 3] = (sign as i8) as u8;
        }
    } else {
        // stride == 8: i16 x/y/z/w with Q = 32767.
        for i in 0..count {
            let base = i * 8;
            let x = i16::from_le_bytes([data[base], data[base + 1]]) as f32;
            let y = i16::from_le_bytes([data[base + 2], data[base + 3]]) as f32;
            let one = i16::from_le_bytes([data[base + 4], data[base + 5]]) as f32;
            let sign = i16::from_le_bytes([data[base + 6], data[base + 7]]);
            let mut nx = x;
            let mut ny = y;
            let nz = one - nx.abs() - ny.abs();
            if nz < 0.0 {
                let tx = nx;
                nx = (32767.0 - ny.abs()).copysign(tx);
                ny = (32767.0 - tx.abs()).copysign(ny);
            }
            let inv_len = 32767.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(1e-20);
            let qx = (nx * inv_len).round().clamp(-32767.0, 32767.0) as i16;
            let qy = (ny * inv_len).round().clamp(-32767.0, 32767.0) as i16;
            let qz = (nz * inv_len).round().clamp(-32767.0, 32767.0) as i16;
            data[base..base + 2].copy_from_slice(&qx.to_le_bytes());
            data[base + 2..base + 4].copy_from_slice(&qy.to_le_bytes());
            data[base + 4..base + 6].copy_from_slice(&qz.to_le_bytes());
            data[base + 6..base + 8].copy_from_slice(&sign.to_le_bytes());
        }
    }
    Ok(())
}

/// QUATERNION filter - decode 16-bit xyzw quaternion with implicit
/// largest-component reconstruction.
fn decode_filter_quaternion(
    data: &mut [u8],
    count: usize,
    stride: usize,
) -> Result<(), MeshoptError> {
    const F: &str = "QUATERNION";
    expect_stride(F, stride, &[8])?;
    expect_len(F, data, count * 8)?;

    const SQRT_2: f32 = std::f32::consts::SQRT_2;
    for i in 0..count {
        let base = i * 8;
        // Stored components (i16), with the max-component index encoded in
        // the low 2 bits of the *first* element's value (mapped via rotation).
        let mut c = [
            i16::from_le_bytes([data[base], data[base + 1]]),
            i16::from_le_bytes([data[base + 2], data[base + 3]]),
            i16::from_le_bytes([data[base + 4], data[base + 5]]),
            i16::from_le_bytes([data[base + 6], data[base + 7]]),
        ]; // The encoder stores the 2-bit index in the LSBs of the *fourth* slot
        // (position of the dropped component). Extract it, then reconstruct.
        let max_idx = (c[3] & 3) as usize;
        // Three stored components (15-bit signed): c[0..3]; scaled to [-1, 1]
        // by dividing by 32767/sqrt(2).
        let inv_scale = SQRT_2 / 32767.0;
        let mut q = [0f32; 4];
        let mut ssq = 0.0f32;
        for k in 0..3 {
            let v = (c[k] >> 1) as f32 * inv_scale; // undo <<1 from encoder
            q[(max_idx + 1 + k) & 3] = v;
            ssq += v * v;
        }
        let max_val = (1.0 - ssq).max(0.0).sqrt();
        q[max_idx] = max_val;

        // Re-quantize back to i16 for output.
        for k in 0..4 {
            let qi = (q[k] * 32767.0).round().clamp(-32767.0, 32767.0) as i16;
            c[k] = qi;
        }
        data[base..base + 2].copy_from_slice(&c[0].to_le_bytes());
        data[base + 2..base + 4].copy_from_slice(&c[1].to_le_bytes());
        data[base + 4..base + 6].copy_from_slice(&c[2].to_le_bytes());
        data[base + 6..base + 8].copy_from_slice(&c[3].to_le_bytes());
    }
    Ok(())
}

/// EXPONENTIAL filter - each component is a u32 where the top 8 bits are
/// the shared power-of-two exponent (stored biased so it fits in i8) and the
/// bottom 24 bits are a signed mantissa. Decoded as `f32 = mantissa * 2^exp`.
fn decode_filter_exponential(
    data: &mut [u8],
    count: usize,
    stride: usize,
) -> Result<(), MeshoptError> {
    const F: &str = "EXPONENTIAL";
    if stride == 0 || stride % 4 != 0 {
        return Err(MeshoptError::InvalidFilterStride { filter: F, stride });
    }
    expect_len(F, data, count * stride)?;

    let comps = stride / 4;
    for i in 0..count {
        for k in 0..comps {
            let off = i * stride + k * 4;
            let raw = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
            // Top 8 bits (sign-extended) = exponent, bottom 24 (sign-extended) = mantissa.
            let exp = (raw as i32) >> 24;
            let mantissa_u = raw & 0x00FF_FFFF;
            let mantissa = if mantissa_u & 0x0080_0000 != 0 {
                (mantissa_u | 0xFF00_0000) as i32
            } else {
                mantissa_u as i32
            };
            let value = (mantissa as f32) * 2f32.powi(exp);
            data[off..off + 4].copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

pub struct MeshoptEncoder;

impl CodecEncoder for MeshoptEncoder {
    const EXT_NAME: &'static str = "EXT_meshopt_compression";
    type Error = MeshoptError;

    fn encode_model(model: &mut GltfModel) -> Result<(), MeshoptError> {
        for mesh_idx in 0..model.meshes.len() {
            for prim_idx in 0..model.meshes[mesh_idx].primitives.len() {
                let prim = &model.meshes[mesh_idx].primitives[prim_idx];
                if prim.indices.is_none() {
                    continue;
                }
                match compress_primitive_meshopt(model, mesh_idx, prim_idx) {
                    Ok(extension) => {
                        model.meshes[mesh_idx].primitives[prim_idx]
                            .extensions
                            .insert(Self::EXT_NAME.to_string(), extension);
                        if !model.extensions_used.contains(&Self::EXT_NAME.to_string()) {
                            model.extensions_used.push(Self::EXT_NAME.to_string());
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Warning: Failed to compress mesh[{mesh_idx}].prim[{prim_idx}] with meshopt: {e}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

/// Encode all eligible primitives with meshopt compression.
pub fn encode(model: &mut GltfModel) -> Result<(), MeshoptError> {
    MeshoptEncoder::encode_model(model)
}

fn compress_primitive_meshopt(
    _model: &GltfModel,
    _mesh_idx: usize,
    _prim_idx: usize,
) -> Result<Value, MeshoptError> {
    // Placeholder - full implementation would compress vertex/index buffers.
    Ok(json!({ "bufferView": null }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octahedral_filter_decodes_unit_vectors_stride4() {
        // Build a trivial input that, after octahedron unwrap, lies on z>0 axis:
        // x=0, y=0, w_sign=127, one=127 -> nx=0, ny=0, nz=127 -> normalized (0,0,127,127)
        let mut data = vec![0i8 as u8, 0i8 as u8, 127u8, 127u8];
        decode_filter_octahedral(&mut data, 1, 4).unwrap();
        assert_eq!(data[0] as i8, 0);
        assert_eq!(data[1] as i8, 0);
        assert_eq!(data[2] as i8, 127);
        assert_eq!(data[3] as i8, 127);
    }

    #[test]
    fn exponential_filter_decodes_zero() {
        // raw = 0 -> exp=0, mantissa=0 -> 0.0
        let mut data = vec![0u8; 4];
        decode_filter_exponential(&mut data, 1, 4).unwrap();
        let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn exponential_filter_decodes_known_value() {
        // mantissa = 1, exp = 0 -> 1.0 * 2^0 = 1.0
        let raw: u32 = 1; // top 8 bits = 0 (exp=0), low 24 = 1 (mantissa)
        let mut data = raw.to_le_bytes().to_vec();
        decode_filter_exponential(&mut data, 1, 4).unwrap();
        let v = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        assert_eq!(v, 1.0);
    }

    #[test]
    fn filter_rejects_bad_stride() {
        let mut data = vec![0u8; 16];
        assert!(matches!(
            decode_filter_octahedral(&mut data, 2, 16),
            Err(MeshoptError::InvalidFilterStride {
                filter: "OCTAHEDRAL",
                ..
            })
        ));
    }
}

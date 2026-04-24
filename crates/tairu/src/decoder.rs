//! Binary tile format decoders for all 3D Tiles 1.x container formats.
//!
//! Each `decode_*` function accepts raw file bytes and returns a [`GltfModel`]
//! that the GPU-upload phase can consume directly, implementing the same
//! conversion logic as cesium-native's `*ToGltfConverter` family.
//!
//! # Supported formats
//!
//! | Format | Magic | Notes |
//! |--------|-------|-------|
//! | GLB    | `glTF`| Passed through as-is. |
//! | b3dm   | `b3dm`| All three header variants (modern, legacy 1, legacy 2). RTC_CENTER injected as CESIUM_RTC. |
//! | i3dm   | `i3dm`| Embedded-GLB mode only (`gltfFormat == 1`). Instance transforms applied via `EXT_mesh_gpu_instancing`. |
//! | cmpt   | `cmpt`| Recursively decodes inner tiles and merges them. |
//! | pnts   | `pnts`| All position/colour/normal encodings; result is a POINTS primitive. |

use crate::tile::{TileFormat, TileFormat::*};
use moderu::{
    Buffer, BufferView, Class, ClassProperty, EXT_STRUCTURAL_METADATA, ExtStructuralMetadata,
    GltfExtension, GltfModel, GltfModelBuilder, GltfReader, HasExtensions, Material, Node,
    PrimitiveMode, PropertyComponentType, PropertyTable, PropertyTableProperty, PropertyType,
    Scene, Schema, UpAxis,
};
use serde_json::Value as Json;
use std::collections::HashMap;

use crate::ext_mesh_features::{ExtMeshFeatures, FeatureId as MeshFeatureId};

const MAX_SANE_TILE_COUNT: usize = 65_536;

/// Tag a model's `extras["gltfUpAxis"]` with the tileset-declared up-axis.
///
/// `TilesetContentManager::postProcessGltfInWorkerThread`, which store the
/// value as an integer (0=X, 1=Y, 2=Z).  Downstream consumers use
/// `CesiumGltfContent::GltfUtilities::applyGltfUpAxisTransform` (or our
/// equivalent `zukei::apply_up_axis_correction`) when baking tile transforms.
fn tag_up_axis(model: &mut GltfModel, up_axis: UpAxis) {
    let extras = model.extras.get_or_insert_with(|| serde_json::json!({}));
    extras["gltfUpAxis"] = serde_json::json!(up_axis as u8);
}

/// Decode any supported tile format from raw bytes.
///
/// `up_axis` is the tileset's `asset.gltfUpAxis` value - it is recorded in
/// the resulting model's `extras["gltfUpAxis"]` so downstream code can bake
/// the correct rotation into tile transforms.  Pass [`UpAxis::Y`] for modern
/// tilesets that do not set the property.
///
/// Returns `None` for unsupported formats (pnts metadata-only, unrecognised),
/// or when the tile header is malformed.
pub fn decode_tile(
    data: &[u8],
    format: &TileFormat,
    up_axis: UpAxis,
    external_glb: Option<&[u8]>,
) -> Option<GltfModel> {
    let mut model = match format {
        Glb => decode_glb(data),
        B3dm => decode_b3dm(data),
        I3dm => decode_i3dm(data, external_glb),
        Cmpt => decode_cmpt(data, up_axis),
        Pnts => decode_pnts(data),
        Json | Unknown => None,
    }?;
    tag_up_axis(&mut model, up_axis);
    Some(model)
}

fn decode_glb(data: &[u8]) -> Option<GltfModel> {
    let model = GltfReader::default().read_bytes(data).ok()?;
    Some(model)
}

/// Threshold for detecting legacy b3dm header variants.
///
/// When the field that would be `batchTableJsonByteLength` in the modern header
/// (bytes `[20..24]`) reads ≥ this value, the bytes there are actually the
/// start of JSON or GLB data, indicating a Legacy-1 header.  The same check
/// on `[24..28]` identifies Legacy-2.  (Values &gt;= 0x22000000 are impossible
/// as byte-length fields for any real payload.)
const B3DM_LEGACY_THRESHOLD: usize = 0x2200_0000;

/// All parsings of a b3dm (modern, legacy-1, legacy-2) reduce to these offsets.
struct B3dmOffsets {
    /// Number of bytes before the feature-table JSON.
    header_len: usize,
    ft_json_len: usize,
    ft_bin_len: usize,
    bt_json_len: usize,
    bt_bin_len: usize,
}

impl B3dmOffsets {
    /// Parse from raw bytes, detecting all three header variants.
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 28 {
            return None;
        }
        let ft_json = le32(data, 12) as usize;
        let ft_bin = le32(data, 16) as usize;
        let bt_json = le32(data, 20) as usize;
        let bt_bin = le32(data, 24) as usize;

        if bt_json >= B3DM_LEGACY_THRESHOLD {
            // Legacy-1: 20-byte header - `ft_bin` holds batchTableByteLength.
            Some(Self {
                header_len: 20,
                ft_json_len: 0,
                ft_bin_len: 0,
                bt_json_len: ft_bin,
                bt_bin_len: 0,
            })
        } else if bt_bin >= B3DM_LEGACY_THRESHOLD {
            // Legacy-2: 24-byte header - `ft_json`/`ft_bin` are btJSON/btBin.
            Some(Self {
                header_len: 24,
                ft_json_len: 0,
                ft_bin_len: 0,
                bt_json_len: ft_json,
                bt_bin_len: ft_bin,
            })
        } else {
            // Modern 28-byte header.
            Some(Self {
                header_len: 28,
                ft_json_len: ft_json,
                ft_bin_len: ft_bin,
                bt_json_len: bt_json,
                bt_bin_len: bt_bin,
            })
        }
    }

    fn glb_start(&self) -> usize {
        self.header_len + self.ft_json_len + self.ft_bin_len + self.bt_json_len + self.bt_bin_len
    }

    fn ft_json_range(&self) -> std::ops::Range<usize> {
        self.header_len..self.header_len + self.ft_json_len
    }
}

/// Convert a 3D Tiles batch table (JSON + optional binary) to an
/// `EXT_structural_metadata` property table attached to `model`.
///
/// Returns `true` if at least one property was successfully encoded.
fn batch_table_to_structural_metadata(
    model: &mut GltfModel,
    bt_json: &[u8],
    bt_bin: &[u8],
    count: usize,
) -> bool {
    let map: serde_json::Map<String, Json> = match serde_json::from_slice(bt_json) {
        Ok(Json::Object(m)) if !m.is_empty() => m,
        _ => return false,
    };

    let buf_idx = model.buffers.len();
    let mut bv_start = model.buffer_views.len();
    let mut prop_buffer: Vec<u8> = Vec::new();
    let mut class_props: HashMap<String, ClassProperty> = HashMap::new();
    let mut table_props: HashMap<String, PropertyTableProperty> = HashMap::new();

    for (key, value) in &map {
        // JSON array of values
        if let Some(arr) = value.as_array() {
            if arr.len() != count {
                continue;
            }
            let all_numbers = arr.iter().all(|v| v.is_number());
            let all_bools = arr.iter().all(|v| v.is_boolean());

            if all_numbers {
                // Encode as FLOAT64
                let byte_offset = prop_buffer.len();
                for v in arr {
                    let f = v.as_f64().unwrap_or(0.0);
                    prop_buffer.extend_from_slice(&f.to_le_bytes());
                }
                let byte_length = prop_buffer.len() - byte_offset;
                model.buffer_views.push(BufferView {
                    buffer: buf_idx,
                    byte_offset,
                    byte_length,
                    ..Default::default()
                });
                class_props.insert(
                    key.clone(),
                    ClassProperty {
                        r#type: PropertyType::Scalar,
                        component_type: Some(PropertyComponentType::Float64),
                        ..Default::default()
                    },
                );
                table_props.insert(
                    key.clone(),
                    PropertyTableProperty {
                        values: Some(bv_start),
                        ..Default::default()
                    },
                );
                bv_start += 1;
            } else if all_bools {
                // Encode as UINT8 (0 or 1)
                let byte_offset = prop_buffer.len();
                for v in arr {
                    prop_buffer.push(if v.as_bool().unwrap_or(false) { 1 } else { 0 });
                }
                let byte_length = prop_buffer.len() - byte_offset;
                model.buffer_views.push(BufferView {
                    buffer: buf_idx,
                    byte_offset,
                    byte_length,
                    ..Default::default()
                });
                class_props.insert(
                    key.clone(),
                    ClassProperty {
                        r#type: PropertyType::Scalar,
                        component_type: Some(PropertyComponentType::Uint8),
                        ..Default::default()
                    },
                );
                table_props.insert(
                    key.clone(),
                    PropertyTableProperty {
                        values: Some(bv_start),
                        ..Default::default()
                    },
                );
                bv_start += 1;
            } else {
                // String (or mixed) — encode as EXT_structural_metadata STRING
                // string_offsets: (count+1) x u32 cumulative byte offsets
                let offsets_bv_idx = bv_start;
                let values_bv_idx = bv_start + 1;
                bv_start += 2;

                let offsets_byte_offset = prop_buffer.len();
                // Collect strings first to build offsets
                let mut strings: Vec<Vec<u8>> = Vec::with_capacity(count);
                for v in arr {
                    let s = if let Some(s) = v.as_str() {
                        s.as_bytes().to_vec()
                    } else {
                        v.to_string().into_bytes()
                    };
                    strings.push(s);
                }
                // Write offsets buffer view (count+1 u32s)
                let mut cumulative: u32 = 0;
                prop_buffer.extend_from_slice(&cumulative.to_le_bytes());
                for s in &strings {
                    cumulative = cumulative.saturating_add(s.len() as u32);
                    prop_buffer.extend_from_slice(&cumulative.to_le_bytes());
                }
                let offsets_byte_length = prop_buffer.len() - offsets_byte_offset;
                model.buffer_views.push(BufferView {
                    buffer: buf_idx,
                    byte_offset: offsets_byte_offset,
                    byte_length: offsets_byte_length,
                    ..Default::default()
                });

                // Write values buffer view (concatenated UTF-8 bytes)
                let values_byte_offset = prop_buffer.len();
                for s in &strings {
                    prop_buffer.extend_from_slice(s);
                }
                let values_byte_length = prop_buffer.len() - values_byte_offset;
                model.buffer_views.push(BufferView {
                    buffer: buf_idx,
                    byte_offset: values_byte_offset,
                    byte_length: values_byte_length,
                    ..Default::default()
                });

                class_props.insert(
                    key.clone(),
                    ClassProperty {
                        r#type: PropertyType::String,
                        component_type: None,
                        ..Default::default()
                    },
                );
                table_props.insert(
                    key.clone(),
                    PropertyTableProperty {
                        values: Some(values_bv_idx),
                        string_offsets: Some(offsets_bv_idx),
                        ..Default::default()
                    },
                );
            }
            continue;
        }

        // Binary reference { "byteOffset": N, "componentType": "...", "type": "..." }
        if let Some(obj) = value.as_object() {
            let byte_offset = match obj.get("byteOffset").and_then(Json::as_u64) {
                Some(n) => n as usize,
                None => continue,
            };
            let component_type_str = obj
                .get("componentType")
                .and_then(Json::as_str)
                .unwrap_or("FLOAT");
            let type_str = obj.get("type").and_then(Json::as_str).unwrap_or("SCALAR");

            let (comp_type, elem_size): (PropertyComponentType, usize) = match component_type_str {
                "BYTE" => (PropertyComponentType::Int8, 1),
                "UNSIGNED_BYTE" => (PropertyComponentType::Uint8, 1),
                "SHORT" => (PropertyComponentType::Int16, 2),
                "UNSIGNED_SHORT" => (PropertyComponentType::Uint16, 2),
                "INT" => (PropertyComponentType::Int32, 4),
                "UNSIGNED_INT" => (PropertyComponentType::Uint32, 4),
                "FLOAT" => (PropertyComponentType::Float32, 4),
                "DOUBLE" => (PropertyComponentType::Float64, 8),
                _ => continue,
            };
            let (prop_type, vec_comps): (PropertyType, usize) = match type_str {
                "SCALAR" => (PropertyType::Scalar, 1),
                "VEC2" => (PropertyType::Vec2, 2),
                "VEC3" => (PropertyType::Vec3, 3),
                "VEC4" => (PropertyType::Vec4, 4),
                _ => continue,
            };

            let total_bytes = count * elem_size * vec_comps;
            let src_end = byte_offset + total_bytes;
            if src_end > bt_bin.len() {
                continue;
            }

            let dst_offset = prop_buffer.len();
            prop_buffer.extend_from_slice(&bt_bin[byte_offset..src_end]);
            let byte_length = prop_buffer.len() - dst_offset;

            model.buffer_views.push(BufferView {
                buffer: buf_idx,
                byte_offset: dst_offset,
                byte_length,
                ..Default::default()
            });
            class_props.insert(
                key.clone(),
                ClassProperty {
                    r#type: prop_type,
                    component_type: Some(comp_type),
                    ..Default::default()
                },
            );
            table_props.insert(
                key.clone(),
                PropertyTableProperty {
                    values: Some(bv_start),
                    ..Default::default()
                },
            );
            bv_start += 1;
        }
    }

    if prop_buffer.is_empty() {
        return false;
    }

    // Push the backing buffer
    let byte_length = prop_buffer.len();
    model.buffers.push(Buffer {
        data: prop_buffer,
        byte_length,
        ..Default::default()
    });

    // Build EXT_structural_metadata
    let mut classes = HashMap::new();
    classes.insert(
        "batchTable".to_owned(),
        Class {
            properties: class_props,
            ..Default::default()
        },
    );
    let ext = ExtStructuralMetadata {
        schema: Some(Schema {
            id: "batch_table_schema".to_owned(),
            classes,
            ..Default::default()
        }),
        property_tables: vec![PropertyTable {
            class: "batchTable".to_owned(),
            count: count as i64,
            properties: table_props,
            ..Default::default()
        }],
        ..Default::default()
    };

    if let Ok(val) = serde_json::to_value(&ext) {
        model
            .extensions
            .insert(EXT_STRUCTURAL_METADATA.to_owned(), val);
        model
            .extensions_used
            .push(EXT_STRUCTURAL_METADATA.to_owned());
        model.extensions_used.sort();
        model.extensions_used.dedup();
    }

    true
}

fn decode_b3dm(data: &[u8]) -> Option<GltfModel> {
    let offsets = B3dmOffsets::parse(data)?;
    let glb_start = offsets.glb_start();
    let glb = data.get(glb_start..)?;

    let mut model = GltfReader::default().read_bytes(glb).ok()?;

    // Inject RTC_CENTER from the feature-table JSON if present.
    if offsets.ft_json_len > 0 {
        let ft_json_bytes = data.get(offsets.ft_json_range())?;
        if let Ok(ft) = serde_json::from_slice::<Json>(ft_json_bytes) {
            if let Some(center) = parse_vec3(&ft, "RTC_CENTER") {
                model.extensions.insert(
                    "CESIUM_RTC".to_owned(),
                    serde_json::json!({ "center": center }),
                );
                model.extensions_used.push("CESIUM_RTC".to_owned());
            }
            // Add BATCH_ID as _FEATURE_ID_0 if present
            let ft_bin_start = offsets.header_len + offsets.ft_json_len;
            let ft_bin = data
                .get(ft_bin_start..ft_bin_start + offsets.ft_bin_len)
                .unwrap_or(&[]);
            let batch_length = ft.get("BATCH_LENGTH").and_then(Json::as_u64).unwrap_or(0) as usize;
            if batch_length > 0 {
                b3dm_add_batch_ids(&mut model, &ft, ft_bin, batch_length);
            }
        }
    }

    // Parse batch table and convert to EXT_structural_metadata
    let bt_json_start = offsets.header_len + offsets.ft_json_len + offsets.ft_bin_len;
    let bt_bin_start = bt_json_start + offsets.bt_json_len;
    if offsets.bt_json_len > 0 {
        let bt_json_bytes = data
            .get(bt_json_start..bt_json_start + offsets.bt_json_len)
            .unwrap_or(&[]);
        let bt_bin_bytes = data
            .get(bt_bin_start..bt_bin_start + offsets.bt_bin_len)
            .unwrap_or(&[]);
        let batch_length = if offsets.ft_json_len > 0 {
            data.get(offsets.ft_json_range())
                .and_then(|b| serde_json::from_slice::<Json>(b).ok())
                .and_then(|ft| ft.get("BATCH_LENGTH").and_then(Json::as_u64))
                .map(|n| n as usize)
                .unwrap_or(0)
        } else {
            0
        };
        if batch_length > 0 {
            batch_table_to_structural_metadata(
                &mut model,
                bt_json_bytes,
                bt_bin_bytes,
                batch_length,
            );
        }
    }

    Some(model)
}

fn b3dm_add_batch_ids(model: &mut GltfModel, ft: &Json, ft_bin: &[u8], batch_length: usize) {
    let Some(batch_id) = ft.get("BATCH_ID") else {
        return;
    };
    let Some(byte_offset) = batch_id.get("byteOffset").and_then(Json::as_u64) else {
        return;
    };
    let component_type = batch_id
        .get("componentType")
        .and_then(Json::as_str)
        .unwrap_or("UNSIGNED_SHORT");
    // Use accessor count from first primitive to get vertex count
    let vertex_count = model
        .meshes
        .first()
        .and_then(|m| m.primitives.first())
        .and_then(|p| p.attributes.values().next().copied())
        .and_then(|acc_idx| model.accessors.get(acc_idx))
        .map(|a| a.count)
        .unwrap_or(0);
    if vertex_count == 0 {
        return;
    }
    let count = vertex_count;
    let mut builder = GltfModelBuilder::new();
    let acc_idx = match component_type {
        "UNSIGNED_BYTE" => {
            let raw = read_u8_vec(ft_bin, byte_offset as usize, count).to_vec();
            builder.add_accessor(&raw)
        }
        "UNSIGNED_INT" => {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = byte_offset as usize + i * 4;
                let bytes = ft_bin.get(off..off + 4).unwrap_or(&[0, 0, 0, 0]);
                out.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            builder.add_accessor(&out)
        }
        _ => {
            // UNSIGNED_SHORT (default)
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = byte_offset as usize + i * 2;
                let bytes = ft_bin.get(off..off + 2).unwrap_or(&[0, 0]);
                out.push(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
            builder.add_accessor(&out)
        }
    };
    let feature_id_model = builder.finish();
    let acc_base = model.accessors.len();
    let merged = std::mem::take(model).merge(feature_id_model);
    *model = merged;
    let feature_acc = acc_base + acc_idx.0;
    let Some(prim) = model
        .meshes
        .first_mut()
        .and_then(|m| m.primitives.first_mut())
    else {
        return;
    };
    prim.attributes
        .insert("_FEATURE_ID_0".to_owned(), feature_acc);
    let mesh_features = ExtMeshFeatures {
        feature_ids: vec![MeshFeatureId {
            feature_count: batch_length as u32,
            attribute: Some(0),
            property_table: Some(0),
            label: Some("_FEATURE_ID_0".to_owned()),
        }],
    };
    prim.set_extension(mesh_features).unwrap();
    model.extensions_used.push(ExtMeshFeatures::NAME.to_owned());
    model.extensions_used.sort_unstable();
    model.extensions_used.dedup();
}

fn i3dm_read_positions(ft: &Json, ft_bin: &[u8], count: usize) -> Option<Vec<[f32; 3]>> {
    if let Some(off) = parse_byte_offset(ft, "POSITION") {
        return Some(read_vec3f32(ft_bin, off, count));
    }
    if let Some(off) = parse_byte_offset(ft, "POSITION_QUANTIZED") {
        let vo = parse_vec3(ft, "QUANTIZED_VOLUME_OFFSET").unwrap_or([0.0; 3]);
        let vs = parse_vec3(ft, "QUANTIZED_VOLUME_SCALE").unwrap_or([1.0; 3]);
        return Some(read_quantized_vec3(ft_bin, off, count, vo, vs));
    }
    None
}

fn i3dm_read_rotations(
    ft: &Json,
    ft_bin: &[u8],
    positions: &[[f32; 3]],
    count: usize,
) -> Vec<[f32; 4]> {
    let east_north_up = ft
        .get("EAST_NORTH_UP")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    if east_north_up {
        return positions
            .iter()
            .map(|&[x, y, z]| {
                let q = terra::enu_quaternion(glam::DVec3::new(x as f64, y as f64, z as f64));
                [q.x as f32, q.y as f32, q.z as f32, q.w as f32]
            })
            .collect();
    }
    if let (Some(up_off), Some(right_off)) = (
        parse_byte_offset(ft, "NORMAL_UP"),
        parse_byte_offset(ft, "NORMAL_RIGHT"),
    ) {
        let ups = read_vec3f32(ft_bin, up_off, count);
        let rights = read_vec3f32(ft_bin, right_off, count);
        return ups
            .iter()
            .zip(rights.iter())
            .map(|(u, r)| zukei::rotation_from_up_right(*u, *r))
            .collect();
    }
    if let (Some(up_off), Some(right_off)) = (
        parse_byte_offset(ft, "NORMAL_UP_OCT32P"),
        parse_byte_offset(ft, "NORMAL_RIGHT_OCT32P"),
    ) {
        let ups = read_oct_normals(ft_bin, up_off, count);
        let rights = read_oct_normals(ft_bin, right_off, count);
        return ups
            .iter()
            .zip(rights.iter())
            .map(|(u, r)| zukei::rotation_from_up_right(*u, *r))
            .collect();
    }
    vec![[0.0, 0.0, 0.0, 1.0]; count]
}

fn i3dm_read_scales(ft: &Json, ft_bin: &[u8], count: usize) -> Vec<[f32; 3]> {
    if let Some(off) = parse_byte_offset(ft, "SCALE_NON_UNIFORM") {
        return read_vec3f32(ft_bin, off, count);
    }
    if let Some(off) = parse_byte_offset(ft, "SCALE") {
        return (0..count)
            .map(|i| {
                let s = read_f32(ft_bin, off + i * 4).unwrap_or(1.0);
                [s, s, s]
            })
            .collect();
    }
    vec![[1.0, 1.0, 1.0]; count]
}

/// Recentre instance positions around their mean to avoid f32 precision loss.
///
/// I3DM positions are ECEF coordinates (~6.3 × 10⁶ m). Stored as f32 this
/// exhausts the mantissa entirely, leaving zero sub-metre precision. By
/// subtracting the mean and storing it as `CESIUM_RTC`, we reduce the stored
/// values to small offsets while preserving full accuracy.
///
/// Mirrors `repositionInstances` in cesium-native's `I3dmToGltfConverter.cpp`.
fn i3dm_recentre_positions(
    positions: &mut Vec<[f32; 3]>,
    rtc_center: Option<[f64; 3]>,
) -> [f64; 3] {
    let count = positions.len();
    if count == 0 {
        return rtc_center.unwrap_or([0.0; 3]);
    }
    // Accumulate as f64 to get an accurate mean.
    let mut sum = [0.0f64; 3];
    for p in positions.iter() {
        sum[0] += p[0] as f64;
        sum[1] += p[1] as f64;
        sum[2] += p[2] as f64;
    }
    let mean = [
        sum[0] / count as f64,
        sum[1] / count as f64,
        sum[2] / count as f64,
    ];
    // Subtract the mean from each position (via f64 to keep precision).
    for p in positions.iter_mut() {
        *p = [
            (p[0] as f64 - mean[0]) as f32,
            (p[1] as f64 - mean[1]) as f32,
            (p[2] as f64 - mean[2]) as f32,
        ];
    }
    // The new RTC centre is mean + any pre-existing RTC centre.
    match rtc_center {
        Some(c) => [mean[0] + c[0], mean[1] + c[1], mean[2] + c[2]],
        None => mean,
    }
}

fn i3dm_apply_instancing(
    mut model: GltfModel,
    positions: Vec<[f32; 3]>,
    rotations: Vec<[f32; 4]>,
    scales: Vec<[f32; 3]>,
    rtc_center: [f64; 3],
) -> GltfModel {
    let mut inst_builder = GltfModelBuilder::new();
    let trans_acc = inst_builder.add_accessor(&positions);
    let rot_acc = inst_builder.add_accessor(&rotations);
    let scale_acc = inst_builder.add_accessor(&scales);
    let inst_model = inst_builder.finish();
    let acc_base = model.accessors.len();
    model = model.merge(inst_model);
    let instancing_ext = serde_json::json!({
        "attributes": {
            "TRANSLATION": acc_base + trans_acc.0,
            "ROTATION":    acc_base + rot_acc.0,
            "SCALE":       acc_base + scale_acc.0,
        }
    });
    let target_node = model
        .nodes
        .iter()
        .position(|n| n.mesh.is_some())
        .or_else(|| {
            if model.nodes.is_empty() {
                None
            } else {
                Some(0)
            }
        });
    if let Some(idx) = target_node {
        model.nodes[idx]
            .extensions
            .insert("EXT_mesh_gpu_instancing".to_owned(), instancing_ext);
    }
    model
        .extensions_used
        .push("EXT_mesh_gpu_instancing".to_owned());
    model.extensions_used.sort();
    model.extensions_used.dedup();
    // Inject the pre-computed RTC centre (mean of instance positions + any
    // original feature-table RTC_CENTER). Positions were already recentred in
    // `i3dm_recentre_positions`, so this is always valid.
    model.extensions.insert(
        "CESIUM_RTC".to_owned(),
        serde_json::json!({ "center": rtc_center }),
    );
    model.extensions_used.push("CESIUM_RTC".to_owned());
    model.extensions_used.sort();
    model.extensions_used.dedup();
    model
}

fn decode_i3dm(data: &[u8], external_glb: Option<&[u8]>) -> Option<GltfModel> {
    // 32-byte header (no legacy variants).
    if data.len() < 32 {
        return None;
    }
    let gltf_format = le32(data, 28);
    let ft_json_len = le32(data, 12) as usize;
    let ft_bin_len = le32(data, 16) as usize;
    let bt_json_len = le32(data, 20) as usize;
    let bt_bin_len = le32(data, 24) as usize;
    let ft_json_start = 32;
    let ft_bin_start = ft_json_start + ft_json_len;
    let glb_start = ft_bin_start + ft_bin_len + bt_json_len + bt_bin_len;

    let glb: &[u8] = if gltf_format == 0 {
        match external_glb {
            Some(bytes) => bytes,
            None => return None, // URI mode not supported without external GLB
        }
    } else {
        data.get(glb_start..)?
    };

    let mut model = GltfReader::default().read_bytes(glb).ok()?;
    let ft: Json = if ft_json_len > 0 {
        let bytes = data.get(ft_json_start..ft_json_start + ft_json_len)?;
        serde_json::from_slice(bytes).ok()?
    } else {
        Json::Object(Default::default())
    };
    let count = ft
        .get("INSTANCES_LENGTH")
        .and_then(Json::as_u64)
        .unwrap_or(0) as usize;
    if count == 0 {
        return Some(model);
    }
    let ft_bin = data
        .get(ft_bin_start..ft_bin_start + ft_bin_len)
        .unwrap_or(&[]);
    let Some(mut positions) = i3dm_read_positions(&ft, ft_bin, count) else {
        return Some(model);
    };
    let existing_rtc = parse_vec3(&ft, "RTC_CENTER");
    let rtc_center = i3dm_recentre_positions(&mut positions, existing_rtc);
    let rotations = i3dm_read_rotations(&ft, ft_bin, &positions, count);
    let scales = i3dm_read_scales(&ft, ft_bin, count);
    model = i3dm_apply_instancing(model, positions, rotations, scales, rtc_center);

    // Parse batch table and convert to EXT_structural_metadata
    let bt_json_start = ft_bin_start + ft_bin_len;
    let bt_bin_start_off = bt_json_start + bt_json_len;
    let batch_table_ok = if bt_json_len > 0 {
        let bt_json_bytes = data
            .get(bt_json_start..bt_json_start + bt_json_len)
            .unwrap_or(&[]);
        let bt_bin_bytes = data
            .get(bt_bin_start_off..bt_bin_start_off + bt_bin_len)
            .unwrap_or(&[]);
        batch_table_to_structural_metadata(&mut model, bt_json_bytes, bt_bin_bytes, count)
    } else {
        false
    };

    // Add _FEATURE_ID_0 to GPU instancing attributes if we have a batch table
    if batch_table_ok {
        if model
            .nodes
            .iter()
            .any(|n| n.extensions.contains_key("EXT_mesh_gpu_instancing"))
        {
            let ids: Vec<u16> = (0..count as u16).collect();
            let mut id_builder = GltfModelBuilder::new();
            let acc_idx = id_builder.add_accessor(&ids);
            let id_model = id_builder.finish();
            let acc_base = model.accessors.len();
            model = model.merge(id_model);
            let feature_acc = acc_base + acc_idx.0;
            if let Some(node) = model
                .nodes
                .iter_mut()
                .find(|n| n.extensions.contains_key("EXT_mesh_gpu_instancing"))
            {
                if let Some(ext_val) = node.extensions.get_mut("EXT_mesh_gpu_instancing") {
                    if let Some(attrs) = ext_val
                        .get_mut("attributes")
                        .and_then(|v| v.as_object_mut())
                    {
                        attrs.insert("_FEATURE_ID_0".to_owned(), serde_json::json!(feature_acc));
                    }
                }
            }
            model.extensions_used.push("EXT_mesh_features".to_owned());
            model.extensions_used.sort();
            model.extensions_used.dedup();
        }
    }

    Some(model)
}

/// Read the external glTF URI from an I3DM with `gltfFormat == 0`.
///
/// Returns `None` if `gltfFormat != 0` or the header is malformed.
pub fn i3dm_external_uri(data: &[u8]) -> Option<String> {
    if data.len() < 32 {
        return None;
    }
    if le32(data, 28) != 0 {
        return None; // gltfFormat != 0
    }
    let ft_json_len = le32(data, 12) as usize;
    let ft_bin_len = le32(data, 16) as usize;
    let bt_json_len = le32(data, 20) as usize;
    let bt_bin_len = le32(data, 24) as usize;
    let glb_start = 32 + ft_json_len + ft_bin_len + bt_json_len + bt_bin_len;
    let uri_bytes = data.get(glb_start..)?;
    let uri_end = uri_bytes
        .iter()
        .position(|&b| b == 0)
        .unwrap_or(uri_bytes.len());
    std::str::from_utf8(&uri_bytes[..uri_end])
        .ok()
        .map(str::to_owned)
}

fn decode_cmpt(data: &[u8], up_axis: UpAxis) -> Option<GltfModel> {
    // 16-byte outer header: magic(4) version(4) byteLength(4) tilesLength(4).
    if data.len() < 16 {
        return None;
    }
    let version = le32(data, 4);
    let byte_len = le32(data, 8) as usize;
    let tile_count = le32(data, 12) as usize;

    if version != 1 || byte_len > data.len() {
        return None;
    }
    debug_assert!(
        tile_count < MAX_SANE_TILE_COUNT,
        "cmpt tile_count={tile_count} exceeds sane maximum - likely corrupt header"
    );
    let tile_count = tile_count.min(MAX_SANE_TILE_COUNT);
    let mut merged: Option<GltfModel> = None;
    let mut pos = 16usize;

    for _ in 0..tile_count {
        // Each inner tile starts with: magic(4) version(4) byteLength(4).
        if pos + 12 > byte_len {
            break;
        }
        let inner_len = le32(data, pos + 8) as usize;
        if inner_len < 12 || pos + inner_len > byte_len {
            break;
        }

        let inner_data = &data[pos..pos + inner_len];
        let inner_format = TileFormat::detect("", inner_data);
        if let Some(inner_model) = decode_tile(inner_data, &inner_format, up_axis, None) {
            merged = Some(match merged.take() {
                None => inner_model,
                Some(m) => m.merge(inner_model),
            });
        }
        pos += inner_len;
    }

    merged
}

/// Per-point or constant colour from a pnts feature table.
enum PntsColorData {
    /// Per-point RGBA [0,255] -> linear f32 [0,1].
    Rgba(Vec<[f32; 4]>),
    /// Per-point RGB [0,255] -> linear f32 [0,1] (alpha = 1).
    Rgb(Vec<[f32; 3]>),
    /// Constant RGBA for all points, pre-converted to linear f32.
    Constant([f32; 4]),
}

fn pnts_read_positions(ft: &Json, ft_bin: &[u8], count: usize) -> Option<Vec<[f32; 3]>> {
    if let Some(off) = parse_byte_offset(ft, "POSITION") {
        return Some(read_vec3f32(ft_bin, off, count));
    }
    if let Some(off) = parse_byte_offset(ft, "POSITION_QUANTIZED") {
        let vo = parse_vec3(ft, "QUANTIZED_VOLUME_OFFSET").unwrap_or([0.0; 3]);
        let vs = parse_vec3(ft, "QUANTIZED_VOLUME_SCALE").unwrap_or([1.0; 3]);
        return Some(read_quantized_vec3(ft_bin, off, count, vo, vs));
    }
    None
}

fn pnts_read_color(ft: &Json, ft_bin: &[u8], count: usize) -> Option<PntsColorData> {
    if let Some(off) = parse_byte_offset(ft, "RGBA") {
        let raw = read_u8_vec(ft_bin, off, count * 4);
        let rgba = raw
            .chunks_exact(4)
            .map(|c| {
                [
                    srgb_u8_to_linear(c[0]),
                    srgb_u8_to_linear(c[1]),
                    srgb_u8_to_linear(c[2]),
                    c[3] as f32 / 255.0,
                ]
            })
            .collect();
        return Some(PntsColorData::Rgba(rgba));
    }
    if let Some(off) = parse_byte_offset(ft, "RGB") {
        let raw = read_u8_vec(ft_bin, off, count * 3);
        let rgb = raw
            .chunks_exact(3)
            .map(|c| {
                [
                    srgb_u8_to_linear(c[0]),
                    srgb_u8_to_linear(c[1]),
                    srgb_u8_to_linear(c[2]),
                ]
            })
            .collect();
        return Some(PntsColorData::Rgb(rgb));
    }
    if let Some(off) = parse_byte_offset(ft, "RGB565") {
        return Some(PntsColorData::Rgb(read_rgb565(ft_bin, off, count)));
    }
    if let Some(arr) = ft.get("CONSTANT_RGBA").and_then(Json::as_array) {
        if arr.len() >= 4 {
            let c: Vec<f32> = arr
                .iter()
                .take(4)
                .map(|v| {
                    debug_assert!(
                        v.as_u64().is_some(),
                        "CONSTANT_RGBA element is not a valid u8 integer"
                    );
                    v.as_u64().unwrap_or(255) as f32 / 255.0
                })
                .collect();
            let rgba = [
                srgb_linear_to_linear(c[0]),
                srgb_linear_to_linear(c[1]),
                srgb_linear_to_linear(c[2]),
                c[3],
            ];
            return Some(PntsColorData::Constant(rgba));
        }
    }
    None
}

fn pnts_build_model(
    positions: Vec<[f32; 3]>,
    color: Option<PntsColorData>,
    normals: Option<Vec<[f32; 3]>>,
) -> GltfModel {
    let mut builder = GltfModelBuilder::new();
    let pos_acc = builder.add_accessor(&positions);
    let mut prim = builder
        .primitive()
        .mode(PrimitiveMode::Points)
        .attribute("POSITION", pos_acc);
    if let Some(norm_data) = normals {
        let norm_acc = builder.add_accessor(&norm_data);
        prim = prim.attribute("NORMAL", norm_acc);
    }
    match &color {
        Some(PntsColorData::Rgba(vals)) => {
            let acc = builder.add_accessor(vals.as_slice());
            prim = prim.attribute("COLOR_0", acc);
        }
        Some(PntsColorData::Rgb(vals)) => {
            let acc = builder.add_accessor(vals.as_slice());
            prim = prim.attribute("COLOR_0", acc);
        }
        Some(PntsColorData::Constant(_)) | None => {}
    }
    builder.add_mesh(prim.build());
    let mut model = builder.finish();
    if let Some(PntsColorData::Constant([r, g, b, a])) = color {
        let mat = Material {
            pbr_metallic_roughness: Some(moderu::MaterialPbrMetallicRoughness {
                base_color_factor: vec![r as f64, g as f64, b as f64, a as f64],
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                ..Default::default()
            }),
            ..Default::default()
        };
        model.materials.push(mat);
        if let Some(mesh) = model.meshes.first_mut() {
            if let Some(prim) = mesh.primitives.first_mut() {
                prim.material = Some(0);
            }
        }
    }
    model.nodes.push(Node {
        mesh: Some(0),
        ..Default::default()
    });
    model.scenes.push(Scene {
        nodes: Some(vec![0]),
        ..Default::default()
    });
    model.scene = Some(0);
    model
}

/// Honour the PNTS `BATCH_ID` semantic by emitting a `_FEATURE_ID_0` vertex
/// attribute and a primitive-level `EXT_mesh_features` extension.
///
/// * `BATCH_ID.componentType` (UNSIGNED_BYTE / UNSIGNED_SHORT / UNSIGNED_INT;
///   default UNSIGNED_SHORT) determines the accessor component type.
/// * `BATCH_LENGTH`, if present, is `featureCount`.  Otherwise the BATCH_ID
///   is silently ignored \u2014 cesium-native warns and skips in that case
///   because there is no associated batch table.
fn pnts_add_batch_ids(model: &mut GltfModel, ft: &Json, ft_bin: &[u8], count: usize) {
    let Some(batch_id) = ft.get("BATCH_ID") else {
        return;
    };
    let Some(byte_offset) = batch_id.get("byteOffset").and_then(Json::as_u64) else {
        return;
    };
    let Some(batch_length) = ft.get("BATCH_LENGTH").and_then(Json::as_u64) else {
        return;
    };
    let component_type = batch_id
        .get("componentType")
        .and_then(Json::as_str)
        .unwrap_or("UNSIGNED_SHORT");

    let mut builder = GltfModelBuilder::new();
    let acc_idx = match component_type {
        "UNSIGNED_BYTE" => {
            let raw = read_u8_vec(ft_bin, byte_offset as usize, count);
            builder.add_accessor(&raw)
        }
        "UNSIGNED_SHORT" => {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = byte_offset as usize + i * 2;
                let bytes = ft_bin.get(off..off + 2).unwrap_or(&[0, 0]);
                out.push(u16::from_le_bytes([bytes[0], bytes[1]]));
            }
            builder.add_accessor(&out)
        }
        "UNSIGNED_INT" => {
            let mut out = Vec::with_capacity(count);
            for i in 0..count {
                let off = byte_offset as usize + i * 4;
                let bytes = ft_bin.get(off..off + 4).unwrap_or(&[0, 0, 0, 0]);
                out.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            }
            builder.add_accessor(&out)
        }
        _ => return,
    };
    let feature_id_model = builder.finish();
    let acc_base = model.accessors.len();
    let merged = std::mem::take(model).merge(feature_id_model);
    *model = merged;
    let feature_acc = acc_base + acc_idx.0;

    let Some(prim) = model
        .meshes
        .first_mut()
        .and_then(|m| m.primitives.first_mut())
    else {
        return;
    };
    prim.attributes
        .insert("_FEATURE_ID_0".to_owned(), feature_acc);
    let mesh_features = ExtMeshFeatures {
        feature_ids: vec![MeshFeatureId {
            feature_count: batch_length as u32,
            attribute: Some(0),
            property_table: Some(0),
            label: Some("_FEATURE_ID_0".to_owned()),
        }],
    };
    prim.set_extension(mesh_features).unwrap();
    model.extensions_used.push(ExtMeshFeatures::NAME.to_owned());
    model.extensions_used.sort_unstable();
    model.extensions_used.dedup();
}

/// Decode a Draco-compressed PNTS feature table into (positions, color, normals).
///
/// Only compiled when the `draco` feature is enabled.
#[cfg(feature = "draco")]
fn decode_pnts_draco(
    draco_ext: &Json,
    ft_bin: &[u8],
    count: usize,
) -> Option<(Vec<[f32; 3]>, Option<PntsColorData>, Option<Vec<[f32; 3]>>)> {
    use draco_core::decoder_buffer::DecoderBuffer;
    use draco_core::draco_types::DataType;
    use draco_core::point_cloud::PointCloud;
    use draco_core::point_cloud_decoder::PointCloudDecoder;

    let byte_offset = draco_ext.get("byteOffset").and_then(|v| v.as_u64())? as usize;
    let byte_length = draco_ext.get("byteLength").and_then(|v| v.as_u64())? as usize;
    let props = draco_ext.get("properties").and_then(|v| v.as_object())?;

    let compressed = ft_bin.get(byte_offset..byte_offset + byte_length)?;

    let mut pc = PointCloud::new();
    let mut decoder = PointCloudDecoder::new();
    let mut buf = DecoderBuffer::new(compressed);
    decoder.decode(&mut buf, &mut pc).ok()?;

    let num_points = pc.num_points();
    if num_points != count {
        return None;
    }

    // Extract raw f32 values from a PointCloud attribute by its unique_id.
    // After PointCloudDecoder::decode(), inverse transforms (dequantization,
    // oct-decoding) have already been applied — positions and normals are Float32.
    let extract_attr = |unique_id: u32, expect_components: usize| -> Option<Vec<f32>> {
        // PointCloud::add_attribute overwrites unique_id with the sequential index,
        // so unique_id == attribute index in practice.
        let attr_idx = (0..pc.num_attributes() as usize)
            .find(|&i| pc.attribute(i as i32).unique_id() == unique_id)?;
        let attr = pc.attribute(attr_idx as i32);
        let raw = attr.buffer().data();
        let stride = attr.byte_stride() as usize;
        let num_comp = attr.num_components() as usize;
        let dt = attr.data_type();
        if num_comp != expect_components || stride == 0 {
            return None;
        }
        let comp_size = dt.byte_length();
        let mut out = Vec::with_capacity(num_points * num_comp);
        for pt in 0..num_points {
            let base = pt * stride;
            for c in 0..num_comp {
                let off = base + c * comp_size;
                let v: f32 = match dt {
                    DataType::Float32 => raw
                        .get(off..off + 4)
                        .and_then(|b| b.try_into().ok())
                        .map(f32::from_le_bytes)
                        .unwrap_or(0.0),
                    DataType::Uint8 => raw.get(off).copied().unwrap_or(0) as f32,
                    DataType::Int8 => raw.get(off).copied().unwrap_or(0) as i8 as f32,
                    DataType::Uint16 => raw
                        .get(off..off + 2)
                        .and_then(|b| b.try_into().ok())
                        .map(u16::from_le_bytes)
                        .unwrap_or(0) as f32,
                    DataType::Int16 => raw
                        .get(off..off + 2)
                        .and_then(|b| b.try_into().ok())
                        .map(i16::from_le_bytes)
                        .unwrap_or(0) as f32,
                    DataType::Uint32 => raw
                        .get(off..off + 4)
                        .and_then(|b| b.try_into().ok())
                        .map(u32::from_le_bytes)
                        .unwrap_or(0) as f32,
                    DataType::Int32 => raw
                        .get(off..off + 4)
                        .and_then(|b| b.try_into().ok())
                        .map(i32::from_le_bytes)
                        .unwrap_or(0) as f32,
                    _ => 0.0,
                };
                out.push(v);
            }
        }
        Some(out)
    };

    // POSITION — required; Float32 x3 after dequantization
    let pos_uid = props.get("POSITION").and_then(|v| v.as_u64())? as u32;
    let pos_flat = extract_attr(pos_uid, 3)?;
    let positions: Vec<[f32; 3]> = pos_flat
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    // Color — optional; Draco stores Uint8 per-component
    let color = if let Some(uid) = props.get("RGBA").and_then(|v| v.as_u64()) {
        extract_attr(uid as u32, 4).map(|flat| {
            PntsColorData::Rgba(
                flat.chunks_exact(4)
                    .map(|c| {
                        [
                            srgb_u8_to_linear(c[0] as u8),
                            srgb_u8_to_linear(c[1] as u8),
                            srgb_u8_to_linear(c[2] as u8),
                            c[3] / 255.0,
                        ]
                    })
                    .collect(),
            )
        })
    } else if let Some(uid) = props.get("RGB").and_then(|v| v.as_u64()) {
        extract_attr(uid as u32, 3).map(|flat| {
            PntsColorData::Rgb(
                flat.chunks_exact(3)
                    .map(|c| {
                        [
                            srgb_u8_to_linear(c[0] as u8),
                            srgb_u8_to_linear(c[1] as u8),
                            srgb_u8_to_linear(c[2] as u8),
                        ]
                    })
                    .collect(),
            )
        })
    } else {
        None
    };

    // Normals — optional; Float32 x3 after oct-decoding
    let normals = props
        .get("NORMAL")
        .and_then(|v| v.as_u64())
        .and_then(|uid| {
            extract_attr(uid as u32, 3)
                .map(|flat| flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
        });

    Some((positions, color, normals))
}

fn decode_pnts(data: &[u8]) -> Option<GltfModel> {
    if data.len() < 28 {
        return None;
    }
    if le32(data, 4) != 1 {
        return None;
    }
    let ft_json_len = le32(data, 12) as usize;
    let ft_bin_len = le32(data, 16) as usize;
    let bt_json_len = le32(data, 20) as usize;
    let bt_bin_len = le32(data, 24) as usize;
    let ft_json_start = 28;
    let ft_bin_start = ft_json_start + ft_json_len;
    let bt_json_start = ft_bin_start + ft_bin_len;
    let bt_bin_start = bt_json_start + bt_json_len;
    let ft_json_bytes = data
        .get(ft_json_start..ft_json_start + ft_json_len)
        .unwrap_or(&[]);
    let ft_bin = data
        .get(ft_bin_start..ft_bin_start + ft_bin_len)
        .unwrap_or(&[]);
    let ft: Json = if ft_json_bytes.is_empty() {
        Json::Object(Default::default())
    } else {
        serde_json::from_slice(ft_json_bytes).ok()?
    };

    let count = ft.get("POINTS_LENGTH").and_then(Json::as_u64).unwrap_or(0) as usize;
    if count == 0 {
        return None;
    }

    // Draco-compressed path: decode attributes from the Draco bitstream.
    if let Some(draco_ext) = ft
        .get("extensions")
        .and_then(|e| e.get("3DTILES_draco_point_compression"))
    {
        #[cfg(feature = "draco")]
        {
            let (positions, color, normals) = decode_pnts_draco(draco_ext, ft_bin, count)?;
            let mut model = pnts_build_model(positions, color, normals);
            if let Some(center) = parse_vec3(&ft, "RTC_CENTER") {
                model.extensions.insert(
                    "CESIUM_RTC".to_owned(),
                    serde_json::json!({ "center": center }),
                );
                model.extensions_used.push("CESIUM_RTC".to_owned());
            }
            if bt_json_len > 0 {
                let bt_json_bytes = data
                    .get(bt_json_start..bt_json_start + bt_json_len)
                    .unwrap_or(&[]);
                let bt_bin_bytes = data
                    .get(bt_bin_start..bt_bin_start + bt_bin_len)
                    .unwrap_or(&[]);
                batch_table_to_structural_metadata(&mut model, bt_json_bytes, bt_bin_bytes, count);
            }
            return Some(model);
        }
        #[cfg(not(feature = "draco"))]
        {
            let _ = draco_ext;
            eprintln!(
                "tairu: 3DTILES_draco_point_compression is not supported (build with --features draco); tile skipped"
            );
            return None;
        }
    }

    let positions = pnts_read_positions(&ft, ft_bin, count)?;
    let color = pnts_read_color(&ft, ft_bin, count);
    let normals = if let Some(off) = parse_byte_offset(&ft, "NORMAL") {
        Some(read_vec3f32(ft_bin, off, count))
    } else if let Some(off) = parse_byte_offset(&ft, "NORMAL_OCT16P") {
        Some(read_oct_normals(ft_bin, off, count))
    } else {
        None
    };
    let mut model = pnts_build_model(positions, color, normals);
    pnts_add_batch_ids(&mut model, &ft, ft_bin, count);
    if let Some(center) = parse_vec3(&ft, "RTC_CENTER") {
        model.extensions.insert(
            "CESIUM_RTC".to_owned(),
            serde_json::json!({ "center": center }),
        );
        model.extensions_used.push("CESIUM_RTC".to_owned());
    }
    // Parse batch table and convert to EXT_structural_metadata
    if bt_json_len > 0 {
        let bt_json_bytes = data
            .get(bt_json_start..bt_json_start + bt_json_len)
            .unwrap_or(&[]);
        let bt_bin_bytes = data
            .get(bt_bin_start..bt_bin_start + bt_bin_len)
            .unwrap_or(&[]);
        batch_table_to_structural_metadata(&mut model, bt_json_bytes, bt_bin_bytes, count);
    }
    Some(model)
}

// (GltfModel::merge lives in moderu - see moderu::merge)

// Binary-read helpers

/// Read a little-endian u32 from `data` at byte offset `off`.
#[inline]
pub(crate) fn le32(data: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(data[off..off + 4].try_into().unwrap_or([0; 4]))
}

#[inline]
fn read_f32(data: &[u8], off: usize) -> Option<f32> {
    data.get(off..off + 4)
        .and_then(|b| b.try_into().ok())
        .map(f32::from_le_bytes)
}

/// Read `count` Vec3-f32 values (12 bytes each) starting at binary offset `off`.
fn read_vec3f32(data: &[u8], off: usize, count: usize) -> Vec<[f32; 3]> {
    (0..count)
        .filter_map(|i| {
            let base = off + i * 12;
            let x = read_f32(data, base)?;
            let y = read_f32(data, base + 4)?;
            let z = read_f32(data, base + 8)?;
            Some([x, y, z])
        })
        .collect()
}

/// Read `count` Vec3 u16-quantized values and dequantize them.
fn read_quantized_vec3(
    data: &[u8],
    off: usize,
    count: usize,
    volume_offset: [f64; 3],
    volume_scale: [f64; 3],
) -> Vec<[f32; 3]> {
    (0..count)
        .filter_map(|i| {
            let base = off + i * 6;
            if base + 6 > data.len() {
                return None;
            }
            let xq = u16::from_le_bytes(data[base..base + 2].try_into().ok()?) as f64;
            let yq = u16::from_le_bytes(data[base + 2..base + 4].try_into().ok()?) as f64;
            let zq = u16::from_le_bytes(data[base + 4..base + 6].try_into().ok()?) as f64;
            let x = (xq / 65535.0 * volume_scale[0] + volume_offset[0]) as f32;
            let y = (yq / 65535.0 * volume_scale[1] + volume_offset[1]) as f32;
            let z = (zq / 65535.0 * volume_scale[2] + volume_offset[2]) as f32;
            Some([x, y, z])
        })
        .collect()
}

/// Read `count` oct-encoded normals (2xu16 each, 4 bytes/normal).
fn read_oct_normals(data: &[u8], off: usize, count: usize) -> Vec<[f32; 3]> {
    (0..count)
        .filter_map(|i| {
            let base = off + i * 4;
            if base + 4 > data.len() {
                return None;
            }
            let ox = u16::from_le_bytes(data[base..base + 2].try_into().ok()?);
            let oy = u16::from_le_bytes(data[base + 2..base + 4].try_into().ok()?);
            let v = outil::codec::oct_decode_16p(ox, oy);
            Some([v[0] as f32, v[1] as f32, v[2] as f32])
        })
        .collect()
}

/// Read a contiguous slice of raw bytes from `data` starting at `off`.
fn read_u8_vec(data: &[u8], off: usize, len: usize) -> &[u8] {
    data.get(off..off + len).unwrap_or(&[])
}

/// Read `count` RGB565-encoded colours and expand to linear f32 vec3.
fn read_rgb565(data: &[u8], off: usize, count: usize) -> Vec<[f32; 3]> {
    (0..count)
        .filter_map(|i| {
            let base = off + i * 2;
            if base + 2 > data.len() {
                return None;
            }
            let raw = u16::from_le_bytes(data[base..base + 2].try_into().ok()?);
            let r5 = ((raw >> 11) & 0x1F) as f32 / 31.0;
            let g6 = ((raw >> 5) & 0x3F) as f32 / 63.0;
            let b5 = (raw & 0x1F) as f32 / 31.0;
            // RGB565 is sRGB; convert to linear.
            Some([
                srgb_linear_to_linear(r5),
                srgb_linear_to_linear(g6),
                srgb_linear_to_linear(b5),
            ])
        })
        .collect()
}

/// sRGB u8 -> linear f32.
#[inline]
fn srgb_u8_to_linear(u: u8) -> f32 {
    srgb_linear_to_linear(u as f32 / 255.0)
}

/// sRGB normalized [0,1] -> linear [0,1].  (IEC 61966-2-1 exact piecewise.)
fn srgb_linear_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

// Feature-table JSON helpers

/// Read a `{ "byteOffset": N }` field from the feature-table JSON.
fn parse_byte_offset(ft: &Json, key: &str) -> Option<usize> {
    ft.get(key)?.get("byteOffset")?.as_u64().map(|n| n as usize)
}

/// Read a `[x, y, z]` array (f64) from the feature-table JSON.
fn parse_vec3(ft: &Json, key: &str) -> Option<[f64; 3]> {
    let arr = ft.get(key)?.as_array()?;
    if arr.len() < 3 {
        return None;
    }
    Some([arr[0].as_f64()?, arr[1].as_f64()?, arr[2].as_f64()?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use moderu::{AccessorType, Asset};

    fn le32_bytes(n: u32) -> [u8; 4] {
        n.to_le_bytes()
    }

    #[test]
    fn oct_decode_x_axis() {
        // (ox=65535, oy=32767) should decode to approximately [1, 0, 0].
        let n = outil::codec::oct_decode_16p(65535, 32767);
        assert!((n[0] - 1.0).abs() < 0.01, "x~1 got {:?}", n);
        assert!(n[1].abs() < 0.01, "y~0 got {:?}", n);
    }

    #[test]
    fn oct_decode_z_axis() {
        // (32767, 32767) should decode to approximately [0, 0, 1] (top of sphere).
        let n = outil::codec::oct_decode_16p(32767, 32767);
        assert!(n[2] > 0.9, "z should be positive, got {:?}", n);
    }

    #[test]
    fn oct_decode_is_unit() {
        let test_cases = [(0u16, 0u16), (65535, 65535), (32767, 0), (0, 32767)];
        for (ox, oy) in test_cases {
            let n = outil::codec::oct_decode_16p(ox, oy);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "({ox},{oy}) len={len:.6} n={n:?}");
        }
    }

    #[test]
    fn identity_matrix_gives_identity_quat() {
        let q = zukei::mat3_to_quat([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        // Identity quaternion: [0,0,0,1]
        assert!((q[3] - 1.0).abs() < 1e-5, "w should be 1, got {:?}", q);
        assert!(q[0].abs() < 1e-5);
        assert!(q[1].abs() < 1e-5);
        assert!(q[2].abs() < 1e-5);
    }

    #[test]
    fn quat_is_unit_length() {
        let q = zukei::mat3_to_quat([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]);
        let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "len={len:.6}");
    }

    #[test]
    fn srgb_black_and_white() {
        assert!((srgb_u8_to_linear(0) - 0.0).abs() < 1e-6);
        assert!((srgb_u8_to_linear(255) - 1.0).abs() < 1e-4);
    }

    #[test]
    fn srgb_midpoint_is_darker_in_linear() {
        let mid = srgb_u8_to_linear(128);
        assert!(mid < 0.5, "linear(128) should be < 0.5, got {mid}");
        assert!(mid > 0.2, "linear(128) should be > 0.2, got {mid}");
    }

    fn minimal_model() -> GltfModel {
        let mut m = GltfModel {
            asset: Asset {
                version: "2.0".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        m.buffers.push(moderu::Buffer {
            data: vec![0u8, 1, 2, 3],
            byte_length: 4,
            ..Default::default()
        });
        m.buffer_views.push(moderu::BufferView {
            buffer: 0,
            byte_offset: 0,
            byte_length: 4,
            ..Default::default()
        });
        m.accessors.push(moderu::Accessor {
            buffer_view: Some(0),
            count: 1,
            component_type: moderu::AccessorComponentType::Float,
            r#type: AccessorType::Scalar,
            ..Default::default()
        });
        m
    }

    #[test]
    fn merge_two_minimal_models_remaps_indices() {
        let a = minimal_model();
        let b = minimal_model();
        let merged = a.merge(b);
        assert_eq!(merged.buffers.len(), 2);
        assert_eq!(merged.buffer_views.len(), 2);
        assert_eq!(merged.accessors.len(), 2);
        // Second buffer_view must point to buffer 1.
        assert_eq!(merged.buffer_views[1].buffer, 1);
        // Second accessor must point to buffer_view 1.
        assert_eq!(merged.accessors[1].buffer_view, Some(1));
    }

    fn build_b3dm(
        header_extra: &[u8],
        ft_json: &[u8],
        ft_bin: &[u8],
        bt_json: &[u8],
        bt_bin: &[u8],
        glb: &[u8],
    ) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(b"b3dm");
        data.extend_from_slice(&le32_bytes(1)); // version
        let total = 28 + ft_json.len() + ft_bin.len() + bt_json.len() + bt_bin.len() + glb.len();
        data.extend_from_slice(&le32_bytes(total as u32));
        data.extend_from_slice(&le32_bytes(ft_json.len() as u32));
        data.extend_from_slice(&le32_bytes(ft_bin.len() as u32));
        data.extend_from_slice(&le32_bytes(bt_json.len() as u32));
        data.extend_from_slice(&le32_bytes(bt_bin.len() as u32));
        let _ = header_extra;
        data.extend_from_slice(ft_json);
        data.extend_from_slice(ft_bin);
        data.extend_from_slice(bt_json);
        data.extend_from_slice(bt_bin);
        data.extend_from_slice(glb);
        data
    }

    /// Minimal valid GLB (12-byte header only, no JSON chunk - not useful for
    /// GltfReader but sufficient to test offset arithmetic).
    fn tiny_glb() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(b"glTF");
        v.extend_from_slice(&le32_bytes(2));
        v.extend_from_slice(&le32_bytes(12));
        v
    }

    #[test]
    fn b3dm_offsets_modern_no_tables() {
        let offsets =
            B3dmOffsets::parse(&build_b3dm(&[], &[], &[], &[], &[], &tiny_glb())).unwrap();
        assert_eq!(offsets.glb_start(), 28);
    }

    #[test]
    fn b3dm_offsets_modern_with_tables() {
        let ft = b"{}";
        let bt = b"{\"a\":1}";
        let data = build_b3dm(&[], ft, &[], bt, &[], &tiny_glb());
        let offsets = B3dmOffsets::parse(&data).unwrap();
        assert_eq!(offsets.glb_start(), 28 + ft.len() + bt.len());
    }

    #[test]
    fn b3dm_legacy1_offsets() {
        // manually construct legacy-1 layout
        let glb = tiny_glb();
        let bt_data = b"[]"; // 2 bytes - will appear at [20..24] as '[', ']', then GLB start
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(b"b3dm");
        data.extend_from_slice(&le32_bytes(1));
        data.extend_from_slice(&le32_bytes((20 + bt_data.len() + glb.len()) as u32));
        data.extend_from_slice(&le32_bytes(0x1000)); // batchLength - read as ft_json
        data.extend_from_slice(&le32_bytes(bt_data.len() as u32)); // batchTableByteLength - read as ft_bin
        data.extend_from_slice(bt_data);
        data.extend_from_slice(&glb);
        // [20..24] should be first 4 bytes of bt_data ('['=0x5B) + first 3 bytes of glb
        let bt_json_field = u32::from_le_bytes(data[20..24].try_into().unwrap());
        assert!(
            bt_json_field >= 0x2200_0000,
            "legacy-1 detection should fire"
        );
        let offsets = B3dmOffsets::parse(&data).unwrap();
        assert_eq!(offsets.glb_start(), 20 + bt_data.len());
    }

    #[test]
    fn cmpt_too_short_returns_none() {
        assert!(decode_cmpt(b"cmpt\x01\x00\x00\x00", UpAxis::Y).is_none());
    }

    #[test]
    fn quantized_vec3_dequantizes_correctly() {
        // A u16 of 65535 at scale 10 and offset 0 should give 10.0.
        let mut data = [0u8; 6];
        data[0..2].copy_from_slice(&65535u16.to_le_bytes());
        data[2..4].copy_from_slice(&0u16.to_le_bytes());
        data[4..6].copy_from_slice(&32767u16.to_le_bytes());
        let pts = read_quantized_vec3(&data, 0, 1, [0.0; 3], [10.0, 10.0, 10.0]);
        assert_eq!(pts.len(), 1);
        assert!((pts[0][0] - 10.0).abs() < 1e-3, "x={}", pts[0][0]);
        assert!((pts[0][1]).abs() < 1e-3, "y={}", pts[0][1]);
        assert!((pts[0][2] - 4.999).abs() < 0.01, "z~5, got {}", pts[0][2]);
    }

    #[test]
    fn pnts_batch_id_emits_feature_ids_extension() {
        // Feature-table JSON: 4 points, BATCH_ID at offset 48 (after 4 * vec3
        // positions = 48 bytes), BATCH_LENGTH = 3.
        let ft_json = br#"{"POINTS_LENGTH":4,"POSITION":{"byteOffset":0},"BATCH_ID":{"byteOffset":48,"componentType":"UNSIGNED_BYTE"},"BATCH_LENGTH":3}"#;
        // Pad JSON to 8-byte alignment.
        let mut ft_json_padded = ft_json.to_vec();
        while ft_json_padded.len() % 8 != 0 {
            ft_json_padded.push(b' ');
        }
        // Binary: 4 zero positions (48 bytes) + 4 batch ids [0,1,2,1]
        let mut ft_bin = vec![0u8; 48];
        ft_bin.extend_from_slice(&[0u8, 1, 2, 1]);
        // Pad binary to 8-byte alignment.
        while ft_bin.len() % 8 != 0 {
            ft_bin.push(0);
        }

        let mut data = Vec::new();
        data.extend_from_slice(b"pnts");
        data.extend_from_slice(&le32_bytes(1)); // version
        let total = 28 + ft_json_padded.len() + ft_bin.len();
        data.extend_from_slice(&le32_bytes(total as u32));
        data.extend_from_slice(&le32_bytes(ft_json_padded.len() as u32));
        data.extend_from_slice(&le32_bytes(ft_bin.len() as u32));
        data.extend_from_slice(&le32_bytes(0)); // bt_json
        data.extend_from_slice(&le32_bytes(0)); // bt_bin
        data.extend_from_slice(&ft_json_padded);
        data.extend_from_slice(&ft_bin);

        let model = decode_pnts(&data).expect("decoded");
        assert!(
            model
                .extensions_used
                .iter()
                .any(|e| e == "EXT_mesh_features"),
            "EXT_mesh_features should be in extensions_used; got {:?}",
            model.extensions_used
        );
        let prim = &model.meshes[0].primitives[0];
        assert!(prim.attributes.contains_key("_FEATURE_ID_0"));
        let ext = prim
            .extensions
            .get("EXT_mesh_features")
            .expect("EXT_mesh_features on primitive");
        assert_eq!(ext["featureIds"][0]["featureCount"], 3);
        assert_eq!(ext["featureIds"][0]["attribute"], 0);
    }

    #[test]
    fn pnts_draco_compression_returns_none() {
        // Build a minimal PNTS with 3DTILES_draco_point_compression in the feature table.
        let ft_json = br#"{"POINTS_LENGTH":4,"extensions":{"3DTILES_draco_point_compression":{"properties":{"POSITION":0}}}}"#;
        let mut ft_json_padded = ft_json.to_vec();
        while ft_json_padded.len() % 8 != 0 {
            ft_json_padded.push(b' ');
        }
        let ft_bin: Vec<u8> = vec![0u8; 8]; // dummy compressed bytes

        let mut data = Vec::new();
        data.extend_from_slice(b"pnts");
        data.extend_from_slice(&le32_bytes(1)); // version
        let total = 28 + ft_json_padded.len() + ft_bin.len();
        data.extend_from_slice(&le32_bytes(total as u32));
        data.extend_from_slice(&le32_bytes(ft_json_padded.len() as u32));
        data.extend_from_slice(&le32_bytes(ft_bin.len() as u32));
        data.extend_from_slice(&le32_bytes(0)); // bt_json
        data.extend_from_slice(&le32_bytes(0)); // bt_bin
        data.extend_from_slice(&ft_json_padded);
        data.extend_from_slice(&ft_bin);

        // Must return None, not corrupt geometry or panic.
        assert!(
            decode_pnts(&data).is_none(),
            "Draco-compressed PNTS must be skipped (return None)"
        );
    }
}

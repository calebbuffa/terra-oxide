//! Quantized-mesh-1.0 terrain data decoder.
//!
//! Decodes Cesium-style quantized-mesh binary tiles into a [`moderu::GltfModel`]
//! (ECEF positions + smooth normals + skirts + optional water mask texture).
//!
//! # Format reference
//!
//! Buffer layout (offsets from byte 0):
//! ```text
//! ┌──────────────────────────────────────────────────── [92 bytes header] ───┐
//! │ CenterX/Y/Z           : f64 x 3  - ECEF bounding sphere centre          │
//! │ MinimumHeight         : f32                                              │
//! │ MaximumHeight         : f32                                              │
//! │ BoundingSphereCenter  : f64 x 3                                          │
//! │ BoundingSphereRadius  : f64                                              │
//! │ HorizonOcclusionPoint : f64 x 3                                          │
//! │ vertexCount           : u32                                              │
//! └──────────────────────────────────────────────────────────────────────────┘
//! uBuffer   : [u16; vertexCount]   zigzag-delta coded, 0-32767 -> west->east
//! vBuffer   : [u16; vertexCount]   zigzag-delta coded, 0-32767 -> south->north
//! heightBuf : [u16; vertexCount]   zigzag-delta coded, 0-32767 -> min->max
//! [4-byte align pad if vertexCount > 65536]
//! triangleCount : u32
//! indices       : [u16 or u32; triangleCount x 3]  high-watermark encoded
//! west/south/east/northEdgeIndicesCount : u32 each + matching index arrays
//! [extension loop while bytes remain]
//!   extensionID     : u8
//!   extensionLength : u32
//!   data            : [u8; extensionLength]
//!   ID 1 = oct-encoded normals (2 bytes / vertex)
//!   ID 2 = water mask (1 byte = land/water flag, or 65536 bytes = 256×256 map)
//!   ID 4 = metadata JSON (u32 length prefix + UTF-8 JSON)
//! ```

use glam::DVec3;
use moderu::{GltfModelBuilder, Image, ImageData, Sampler, SkirtMeshMetadata, Texture, UpAxis};
use outil::io::{BufferReader, UnexpectedEndOfData};
use terra::{Cartographic, Ellipsoid, calc_quadtree_max_geometric_error};
use thiserror::Error;
pub use zukei::QuadtreeTileRectangularRange;

#[derive(Debug, Error)]
pub enum QuantizedMeshError {
    #[error("data is too short for a quantized-mesh header: {0} bytes (need 92)")]
    HeaderTooShort(usize),
    #[error(transparent)]
    Read(#[from] UnexpectedEndOfData),
}

/// Parsed 92-byte quantized-mesh file header.
#[derive(Debug, Default, Clone)]
pub struct QuantizedMeshHeader {
    /// ECEF coordinates of the tile's bounding-sphere centre.
    pub center: [f64; 3],
    pub min_height: f64,
    pub max_height: f64,
    pub bounding_sphere_center: [f64; 3],
    pub bounding_sphere_radius: f64,
    pub horizon_occlusion_point: [f64; 3],
    pub vertex_count: usize,
}

impl QuantizedMeshHeader {
    pub const SIZE: usize = 92;

    pub fn parse(data: &[u8]) -> Result<Self, QuantizedMeshError> {
        if data.len() < Self::SIZE {
            return Err(QuantizedMeshError::HeaderTooShort(data.len()));
        }
        let mut r = BufferReader::new(data);
        Ok(Self {
            center: [
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
            ],
            min_height: r.read_le::<f32>()? as f64,
            max_height: r.read_le::<f32>()? as f64,
            bounding_sphere_center: [
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
            ],
            bounding_sphere_radius: r.read_le::<f64>()?,
            horizon_occlusion_point: [
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
                r.read_le::<f64>()?,
            ],
            vertex_count: r.read_le::<u32>()? as usize,
        })
    }

    pub(crate) fn center_vec(&self) -> DVec3 {
        DVec3::from(self.center)
    }
}

/// Result of decoding a quantized-mesh-1.0 tile.
#[derive(Debug)]
pub struct QuantizedMeshResult {
    /// Decoded glTF model: ECEF positions, smooth normals, skirt geometry,
    /// optional water mask texture, and `SkirtMeshMetadata` in primitive extras.
    pub model: moderu::GltfModel,
    /// Tile availability rectangles parsed from the metadata extension (ext ID 4).
    ///
    /// Each entry covers `[start_x..=end_x, start_y..=end_y]` at `level`.
    pub available_tiles: Vec<QuadtreeTileRectangularRange>,
    /// The entire tile surface is water.
    pub only_water: bool,
    /// The entire tile surface is land (no water).
    pub only_land: bool,
}

/// Decode a quantized-mesh-1.0 binary blob into a [`QuantizedMeshResult`].
///
/// `tile_level` is the quadtree zoom level of this tile; used as the base for
/// availability metadata (ext ID 4 lists ranges starting at `tile_level + 1`).
///
/// `west`, `south`, `east`, `north` are the tile's geodetic bounds in radians.
pub fn decode_quantized_mesh(
    data: &[u8],
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    tile_level: u32,
    ellipsoid: &Ellipsoid,
) -> Result<QuantizedMeshResult, QuantizedMeshError> {
    let header = QuantizedMeshHeader::parse(data)?;
    let mut r = BufferReader::new(data);
    r.seek(QuantizedMeshHeader::SIZE);

    let vc = header.vertex_count;

    // u/v/height vertex buffers (zigzag-delta encoded u16)
    let buf_bytes = vc * 2;
    let u_buf = le_slice::<u16>(r.read_bytes(buf_bytes)?);
    let v_buf = le_slice::<u16>(r.read_bytes(buf_bytes)?);
    let h_buf = le_slice::<u16>(r.read_bytes(buf_bytes)?);

    // Decode to ECEF positions
    let center = header.center_vec();
    let (mut positions, uvh, mut pos_min, mut pos_max) = decode_positions(
        &header, &u_buf, &v_buf, &h_buf, west, south, east, north, center, ellipsoid,
    );

    // Indices (u32 for > 65536 vertices, u16 otherwise)
    let use_32bit = vc > 65536;
    // Alignment before IndexData (spec):
    //   16-bit: 2-byte alignment (always met after header + vc×6 bytes, no skip needed)
    //   32-bit: 4-byte alignment (may need 0 or 2 bytes of padding)
    if use_32bit {
        r.align_to_4();
    }
    let triangle_count = r.read_le::<u32>()? as usize;
    let mut indices = read_index_buffer(&mut r, triangle_count * 3, use_32bit)?;

    // Edge vertex lists (west, south, east, north)
    let west_edge = read_edge_buffer(&mut r, use_32bit)?;
    let south_edge = read_edge_buffer(&mut r, use_32bit)?;
    let east_edge = read_edge_buffer(&mut r, use_32bit)?;
    let north_edge = read_edge_buffer(&mut r, use_32bit)?;

    // Extensions
    let ext = parse_extensions(&mut r, vc, tile_level)?;

    // Normals: use oct-decoded if present, otherwise compute from geometry
    let mut normals = decode_normals(&positions, &indices, ext.oct_normals.as_deref(), vc);

    // Snapshot pre-skirt counts for SkirtMeshMetadata
    let no_skirt_vertices_count = positions.len() as u32;
    let no_skirt_indices_count = indices.len() as u32;

    // Append skirt geometry
    let skirt_ht = append_skirts(
        &mut positions,
        &mut normals,
        &mut indices,
        &uvh,
        [&west_edge, &south_edge, &east_edge, &north_edge],
        &mut pos_min,
        &mut pos_max,
        west,
        south,
        east,
        north,
        header.min_height,
        header.max_height,
        center,
        ellipsoid,
    );

    let model = build_gltf(
        positions,
        normals,
        indices,
        pos_min,
        pos_max,
        center,
        no_skirt_vertices_count,
        no_skirt_indices_count,
        skirt_ht,
        ext.only_water,
        ext.only_land,
        ext.water_mask_pixels,
    );

    Ok(QuantizedMeshResult {
        model,
        available_tiles: ext.available_tiles,
        only_water: ext.only_water,
        only_land: ext.only_land,
    })
}

struct ParsedExtensions {
    oct_normals: Option<Vec<u8>>,
    only_water: bool,
    only_land: bool,
    water_mask_pixels: Option<Vec<u8>>,
    available_tiles: Vec<QuadtreeTileRectangularRange>,
}

fn parse_extensions(
    r: &mut BufferReader<'_>,
    vertex_count: usize,
    tile_level: u32,
) -> Result<ParsedExtensions, QuantizedMeshError> {
    const EXT_HEADER_SIZE: usize = 5; // 1-byte id + 4-byte length

    let mut ext = ParsedExtensions {
        oct_normals: None,
        only_water: false,
        only_land: true,
        water_mask_pixels: None,
        available_tiles: Vec::new(),
    };

    while r.remaining() >= EXT_HEADER_SIZE {
        let ext_id = r.read_le::<u8>()?;
        let ext_len = r.read_le::<u32>()? as usize;

        if r.remaining() < ext_len {
            break;
        }
        let body_start = r.position();

        match ext_id {
            1 if ext_len >= vertex_count * 2 => {
                ext.oct_normals = Some(r.read_bytes(vertex_count * 2)?.to_vec());
            }
            2 if ext_len == 1 => {
                let flag = r.read_le::<u8>()?;
                ext.only_water = flag != 0;
                ext.only_land = !ext.only_water;
            }
            2 if ext_len == 65536 => {
                ext.only_water = false;
                ext.only_land = false;
                ext.water_mask_pixels = Some(r.read_bytes(65536)?.to_vec());
            }
            4 if ext_len >= 4 => {
                let json_len = r.read_le::<u32>()? as usize;
                if json_len > 0 && r.remaining() >= json_len {
                    if let Ok(s) = std::str::from_utf8(r.read_bytes(json_len)?) {
                        parse_availability(s, tile_level + 1, &mut ext.available_tiles);
                    }
                }
            }
            _ => {}
        }

        // Always jump to the next extension boundary.
        r.seek(body_start + ext_len);
    }

    Ok(ext)
}

fn decode_positions(
    header: &QuantizedMeshHeader,
    u_buf: &[u16],
    v_buf: &[u16],
    h_buf: &[u16],
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    center: DVec3,
    ellipsoid: &Ellipsoid,
) -> (Vec<[f32; 3]>, Vec<(f64, f64, f64)>, [f64; 3], [f64; 3]) {
    let vc = header.vertex_count;
    let mut positions = Vec::with_capacity(vc);
    let mut uvh = Vec::with_capacity(vc);
    let mut pos_min = [f64::MAX; 3];
    let mut pos_max = [f64::MIN; 3];

    let mut u_acc = 0i32;
    let mut v_acc = 0i32;
    let mut h_acc = 0i32;

    for i in 0..vc {
        u_acc += outil::codec::zigzag_decode(u_buf[i] as i32);
        v_acc += outil::codec::zigzag_decode(v_buf[i] as i32);
        h_acc += outil::codec::zigzag_decode(h_buf[i] as i32);

        let ur = u_acc as f64 / 32767.0;
        let vr = v_acc as f64 / 32767.0;
        let hr = h_acc as f64 / 32767.0;

        let ecef = geodetic_to_ecef(
            header, ur, vr, hr, west, south, east, north, 0.0, 0.0, 0.0, center, ellipsoid,
        );
        expand_bounds(&mut pos_min, &mut pos_max, ecef);
        positions.push([ecef.x as f32, ecef.y as f32, ecef.z as f32]);
        uvh.push((ur, vr, hr));
    }

    (positions, uvh, pos_min, pos_max)
}

/// Convert tile-local (ur, vr, hr) ratios to an ECEF offset from `center`.
fn geodetic_to_ecef(
    header: &QuantizedMeshHeader,
    ur: f64,
    vr: f64,
    hr: f64,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    dlon: f64,
    dlat: f64,
    dheight: f64,
    center: DVec3,
    ellipsoid: &Ellipsoid,
) -> DVec3 {
    let lon = west + (east - west) * ur + dlon;
    let lat = south + (north - south) * vr + dlat;
    let height = header.min_height + (header.max_height - header.min_height) * hr + dheight;
    ellipsoid.cartographic_to_ecef(Cartographic {
        longitude: lon,
        latitude: lat,
        height,
    }) - center
}

fn decode_normals(
    positions: &[[f32; 3]],
    indices: &[u32],
    oct_normals: Option<&[u8]>,
    vertex_count: usize,
) -> Vec<[f32; 3]> {
    let mut normals: Vec<[f32; 3]> = match oct_normals {
        Some(enc) => enc[..vertex_count * 2]
            .chunks_exact(2)
            .map(|c| {
                let n = outil::codec::oct_decode(c[0], c[1]);
                [n[0] as f32, n[1] as f32, n[2] as f32]
            })
            .collect(),
        None => smooth_normals(positions, indices),
    };
    // Skirt vertices will be appended later; pre-fill to vertex_count.
    normals.resize(vertex_count, [0.0, 1.0, 0.0]);
    normals
}

fn read_index_buffer(
    r: &mut BufferReader<'_>,
    count: usize,
    use_32bit: bool,
) -> Result<Vec<u32>, QuantizedMeshError> {
    if use_32bit {
        Ok(high_watermark_decode(le_slice::<u32>(
            r.read_bytes(count * 4)?,
        )))
    } else {
        Ok(high_watermark_decode(le_slice::<u16>(
            r.read_bytes(count * 2)?,
        )))
    }
}

fn read_edge_buffer(
    r: &mut BufferReader<'_>,
    use_32bit: bool,
) -> Result<Vec<u32>, QuantizedMeshError> {
    let count = r.read_le::<u32>()? as usize;
    read_index_buffer(r, count, use_32bit)
}

/// Append skirt triangles to `positions`, `normals`, and `indices`.
///
/// `edges` is `[west, south, east, north]`. Returns the skirt drop height.
fn append_skirts(
    positions: &mut Vec<[f32; 3]>,
    normals: &mut Vec<[f32; 3]>,
    indices: &mut Vec<u32>,
    uvh: &[(f64, f64, f64)],
    edges: [&[u32]; 4],
    pos_min: &mut [f64; 3],
    pos_max: &mut [f64; 3],
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    min_height: f64,
    max_height: f64,
    center: DVec3,
    ellipsoid: &Ellipsoid,
) -> f64 {
    let width = east - west;
    let skirt_ht = calc_quadtree_max_geometric_error(ellipsoid) * width * 5.0;
    let lon_off = width * 0.0001;
    let lat_off = (north - south) * 0.0001;

    // Sort each edge so skirt quads are wound consistently:
    //   west:  ascending  V (south -> north)
    //   south: descending U (east  -> west)
    //   east:  descending V (north -> south)
    //   north: ascending  U (west  -> east)
    let sorted: [Vec<u32>; 4] = {
        let sort = |edge: &[u32], key: fn((f64, f64, f64)) -> f64, asc: bool| {
            let mut v = edge.to_vec();
            v.sort_by(|&a, &b| {
                let ka = key(uvh[a as usize]);
                let kb = key(uvh[b as usize]);
                if asc {
                    ka.partial_cmp(&kb)
                } else {
                    kb.partial_cmp(&ka)
                }
                .unwrap_or(std::cmp::Ordering::Equal)
            });
            v
        };
        [
            sort(edges[0], |(_, v, _)| v, true),  // west:  asc V
            sort(edges[1], |(u, _, _)| u, false), // south: desc U
            sort(edges[2], |(_, v, _)| v, false), // east:  desc V
            sort(edges[3], |(u, _, _)| u, true),  // north: asc U
        ]
    };

    // lon/lat offsets per edge to push skirts slightly outward
    let offsets: [(f64, f64); 4] = [
        (-lon_off, 0.0),
        (0.0, -lat_off),
        (lon_off, 0.0),
        (0.0, lat_off),
    ];

    // Reuse QuantizedMeshHeader fields via a minimal local stub
    let stub_header = QuantizedMeshHeader {
        min_height,
        max_height,
        ..Default::default()
    };

    for (edge, (dlon, dlat)) in sorted.iter().zip(offsets.iter()) {
        let base = positions.len() as u32;
        for (i, &ei) in edge.iter().enumerate() {
            let (ur, vr, hr) = uvh[ei as usize];
            let ecef = geodetic_to_ecef(
                &stub_header,
                ur,
                vr,
                hr,
                west,
                south,
                east,
                north,
                *dlon,
                *dlat,
                -skirt_ht,
                center,
                ellipsoid,
            );
            expand_bounds(pos_min, pos_max, ecef);
            positions.push([ecef.x as f32, ecef.y as f32, ecef.z as f32]);
            normals.push(normals.get(ei as usize).copied().unwrap_or([0.0, 1.0, 0.0]));

            if i + 1 < edge.len() {
                let (a, b) = (ei, edge[i + 1]);
                let (c, d) = (base + i as u32, base + i as u32 + 1);
                indices.extend_from_slice(&[a, b, c, b, d, c]);
            }
        }
    }

    skirt_ht
}

fn build_gltf(
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
    pos_min: [f64; 3],
    pos_max: [f64; 3],
    center: DVec3,
    no_skirt_vertices_count: u32,
    no_skirt_indices_count: u32,
    skirt_ht: f64,
    only_water: bool,
    only_land: bool,
    water_mask_pixels: Option<Vec<u8>>,
) -> moderu::GltfModel {
    let skirt_meta = SkirtMeshMetadata {
        no_skirt_indices_begin: 0,
        no_skirt_indices_count,
        no_skirt_vertices_begin: 0,
        no_skirt_vertices_count,
        mesh_center: [center.x, center.y, center.z],
        skirt_west_height: skirt_ht,
        skirt_south_height: skirt_ht,
        skirt_east_height: skirt_ht,
        skirt_north_height: skirt_ht,
    };

    let mut prim_extras = skirt_meta.to_extras_value();
    prim_extras["OnlyWater"] = serde_json::json!(only_water);
    prim_extras["OnlyLand"] = serde_json::json!(only_land);
    prim_extras["WaterMaskTranslationX"] = serde_json::json!(0.0);
    prim_extras["WaterMaskTranslationY"] = serde_json::json!(0.0);
    prim_extras["WaterMaskScale"] = serde_json::json!(1.0);

    let mut b = GltfModelBuilder::new();
    b.up_axis(UpAxis::Y);

    let pos_acc = b.add_accessor(&positions);
    let norm_acc = b.add_accessor(&normals);
    let idx_acc = b.add_indices_compact(&indices);
    let mat = b.add_default_material([0.8, 0.8, 0.8, 1.0]);

    let water_mask_tex: i32 = water_mask_pixels
        .map(|px| add_water_mask_texture(b.model_mut(), px) as i32)
        .unwrap_or(-1);
    prim_extras["WaterMaskTex"] = serde_json::json!(water_mask_tex);

    let prim = b
        .primitive()
        .indices(idx_acc)
        .attribute("POSITION", pos_acc)
        .attribute("NORMAL", norm_acc)
        .material(mat)
        .extras(prim_extras)
        .build();
    b.add_mesh(prim);

    // Set position accessor min/max bounds.
    let model = b.model_mut();
    if let Some(acc) = model.accessors.get_mut(pos_acc.0) {
        acc.min = pos_min.to_vec();
        acc.max = pos_max.to_vec();
    }

    b.finish()
}

/// Add a 256×256 single-channel water mask image/sampler/texture to `model`.
///
/// Returns the texture index.
fn add_water_mask_texture(model: &mut moderu::GltfModel, pixels: Vec<u8>) -> usize {
    let image_idx = model.images.len();
    model.images.push(Image {
        pixels: ImageData {
            data: pixels,
            width: 256,
            height: 256,
            channels: 1,
            bytes_per_channel: 1,
            ..Default::default()
        },
        ..Default::default()
    });
    let sampler_idx = model.samplers.len();
    model.samplers.push(Sampler {
        mag_filter: Some(9729), // LINEAR
        min_filter: Some(9985), // LINEAR_MIPMAP_NEAREST
        wrap_s: 33071,          // CLAMP_TO_EDGE
        wrap_t: 33071,
        ..Default::default()
    });
    let tex_idx = model.textures.len();
    model.textures.push(Texture {
        sampler: Some(sampler_idx),
        source: Some(image_idx),
        ..Default::default()
    });
    tex_idx
}

/// Decode a high-watermark encoded index buffer into plain `u32` indices.
///
/// The quantized-mesh spec encodes indices as deltas from a running
/// "high-water mark" (the highest index seen so far). `code == 0` means
/// "next new vertex" and advances the watermark; any other value means
/// `watermark - code` (a reference to a previously seen vertex).
pub fn high_watermark_decode<T: HwmIndex>(encoded: Vec<T>) -> Vec<u32> {
    let mut out = Vec::with_capacity(encoded.len());
    let mut highest = T::ZERO;
    for code in encoded {
        out.push(HwmIndex::wrapping_sub(highest, code).into());
        if code.is_zero() {
            highest = highest.wrapping_add_one();
        }
    }
    out
}

/// Sealed trait implemented by `u16` and `u32` for high-watermark decoding.
pub trait HwmIndex: Copy + Into<u32> {
    const ZERO: Self;
    fn wrapping_sub(self, other: Self) -> Self;
    fn wrapping_add_one(self) -> Self;
    fn is_zero(self) -> bool;
}

impl HwmIndex for u16 {
    const ZERO: Self = 0;
    fn wrapping_sub(self, other: Self) -> Self {
        u16::wrapping_sub(self, other)
    }
    fn wrapping_add_one(self) -> Self {
        self.wrapping_add(1)
    }
    fn is_zero(self) -> bool {
        self == 0
    }
}

impl HwmIndex for u32 {
    const ZERO: Self = 0;
    fn wrapping_sub(self, other: Self) -> Self {
        u32::wrapping_sub(self, other)
    }
    fn wrapping_add_one(self) -> Self {
        self.wrapping_add(1)
    }
    fn is_zero(self) -> bool {
        self == 0
    }
}

/// Generate area-weighted smooth normals.
///
/// Accumulates the cross product of each triangle's edges (weighted by area)
/// across all triangles sharing a vertex, then normalises. Matches
/// `generateNormals` in `QuantizedMeshLoader.cpp`.
fn smooth_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = positions[i0];
        let p1 = positions[i1];
        let p2 = positions[i2];
        let e1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
        let n = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for &i in &[i0, i1, i2] {
            normals[i][0] += n[0];
            normals[i][1] += n[1];
            normals[i][2] += n[2];
        }
    }
    for n in &mut normals {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > f32::EPSILON {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        }
    }
    normals
}

/// Parse the `"available"` array from a quantized-mesh metadata JSON string.
///
/// The array is a list-of-lists of `{startX, startY, endX, endY}` objects.
/// Each outer index `i` corresponds to `starting_level + i`.
fn parse_availability(
    json: &str,
    starting_level: u32,
    out: &mut Vec<QuadtreeTileRectangularRange>,
) {
    let Ok(doc) = serde_json::from_str::<serde_json::Value>(json) else {
        return;
    };
    let Some(available) = doc.get("available").and_then(|v| v.as_array()) else {
        return;
    };
    for (i, ranges_at_level) in available.iter().enumerate() {
        let Some(arr) = ranges_at_level.as_array() else {
            continue;
        };
        let level = starting_level + i as u32;
        for range in arr {
            let get_u32 = |key: &str| range.get(key).and_then(|v| v.as_u64()).map(|n| n as u32);
            if let (Some(sx), Some(sy), Some(ex), Some(ey)) = (
                get_u32("startX"),
                get_u32("startY"),
                get_u32("endX"),
                get_u32("endY"),
            ) {
                out.push(QuadtreeTileRectangularRange {
                    level,
                    start_x: sx,
                    start_y: sy,
                    end_x: ex,
                    end_y: ey,
                });
            }
        }
    }
}

/// Parse a byte slice as a sequence of little-endian `T` values.
fn le_slice<T: LeReadable>(bytes: &[u8]) -> Vec<T> {
    bytes.chunks_exact(T::STRIDE).map(T::from_le).collect()
}

trait LeReadable: Sized + Copy {
    const STRIDE: usize;
    fn from_le(bytes: &[u8]) -> Self;
}

impl LeReadable for u16 {
    const STRIDE: usize = 2;
    fn from_le(bytes: &[u8]) -> Self {
        u16::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl LeReadable for u32 {
    const STRIDE: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        u32::from_le_bytes(bytes.try_into().unwrap())
    }
}

fn expand_bounds(min: &mut [f64; 3], max: &mut [f64; 3], v: DVec3) {
    min[0] = min[0].min(v.x);
    min[1] = min[1].min(v.y);
    min[2] = min[2].min(v.z);
    max[0] = max[0].max(v.x);
    max[1] = max[1].max(v.y);
    max[2] = max[2].max(v.z);
}

/// Input for [`encode_quantized_mesh`].
///
/// Heights are elevation in **metres** relative to the WGS84 ellipsoid, stored
/// in a `grid_size × grid_size` row-major grid ordered **south-to-north** (row
/// 0 = southernmost row) and **west-to-east** (col 0 = westernmost column).
///
/// `west`, `south`, `east`, `north` are tile bounds in **radians**.
pub struct QuantizedMeshInput<'a> {
    /// Elevation samples in metres, length must be `grid_size * grid_size`.
    pub heights: &'a [f64],
    /// Side length of the height grid (minimum 2).
    pub grid_size: usize,
    /// Tile bounds in radians (EPSG:4326).
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
    pub ellipsoid: &'a terra::Ellipsoid,
}

/// Encode a regular-grid elevation array into a `quantized-mesh-1.0` binary blob.
///
/// The output is raw (uncompressed); callers may gzip it if needed.
/// Tile indices are `u16` (max 65 535 vertices per tile, suitable for all
/// reasonable grid sizes).
///
/// The encoding follows the [quantized-mesh-1.0
/// spec](https://github.com/CesiumGS/quantized-mesh).
pub fn encode_quantized_mesh(input: &QuantizedMeshInput<'_>) -> Vec<u8> {
    let QuantizedMeshInput {
        heights,
        grid_size,
        west,
        south,
        east,
        north,
        ellipsoid,
    } = input;
    let n = *grid_size;
    assert!(n >= 2, "grid_size must be at least 2");
    assert_eq!(heights.len(), n * n, "heights length must equal grid_size²");

    // 1. Derive min/max heights
    let min_h = heights.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_h = heights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let h_range = (max_h - min_h).max(1.0); // avoid division by zero on flat tiles

    // 2. Quantise u/v/height to u16 [0, 32767]
    //
    // u = column / (n-1), v = row / (n-1)
    let vertex_count = n * n;
    let mut u_raw = Vec::with_capacity(vertex_count);
    let mut v_raw = Vec::with_capacity(vertex_count);
    let mut h_raw = Vec::with_capacity(vertex_count);
    let mut ecef_positions: Vec<DVec3> = Vec::with_capacity(vertex_count);

    for row in 0..n {
        for col in 0..n {
            let ur = col as f64 / (n - 1) as f64;
            let vr = row as f64 / (n - 1) as f64;
            let h = heights[row * n + col];
            let hr = (h - min_h) / h_range;

            u_raw.push((ur * 32767.0).round() as u16);
            v_raw.push((vr * 32767.0).round() as u16);
            h_raw.push((hr * 32767.0).round() as u16);

            let lon = west + (east - west) * ur;
            let lat = south + (north - south) * vr;
            ecef_positions.push(ellipsoid.cartographic_to_ecef(terra::Cartographic {
                longitude: lon,
                latitude: lat,
                height: h,
            }));
        }
    }

    // 6. Build regular-grid triangulation(row-major, CCW winding)
    //
    // Each quad (col, row) -> (col+1, row+1) is split into 2 CCW triangles:
    //   lower-left:  BL=(col,row),   BR=(col+1,row), TL=(col,row+1)
    //   upper-right: BR=(col+1,row), TR=(col+1,row+1), TL=(col,row+1)
    let idx = |row: usize, col: usize| (row * n + col) as u16;
    let mut indices: Vec<u16> = Vec::with_capacity((n - 1) * (n - 1) * 6);
    for row in 0..n - 1 {
        for col in 0..n - 1 {
            let bl = idx(row, col);
            let br = idx(row, col + 1);
            let tl = idx(row + 1, col);
            let tr = idx(row + 1, col + 1);
            indices.extend_from_slice(&[bl, br, tl, br, tr, tl]);
        }
    }

    // 7. Edge index arrays (before reindexing — same original indices)
    let west_edge: Vec<u16> = (0..n).map(|r| idx(r, 0)).collect();
    let south_edge: Vec<u16> = (0..n).map(|c| idx(0, c)).collect();
    let east_edge: Vec<u16> = (0..n).map(|r| idx(r, n - 1)).collect();
    let north_edge: Vec<u16> = (0..n).map(|c| idx(n - 1, c)).collect();

    // 8. Reindex vertices in first-appearance order within the index buffer.
    //
    // The quantized-mesh high-watermark encoding requires that vertex `k` is
    // introduced exactly when the running watermark equals `k`.  With a
    // row-major grid, vertex `n` (start of row 1) first appears long after
    // vertices 0..n-1, so a naïve encode produces watermark jumps that the
    // decoder cannot reproduce.  Fix: remap vertex indices so that they are
    // numbered 0, 1, 2, … in the order they first appear in the triangle
    // stream, then reorder the vertex arrays to match.
    let mut remap: Vec<Option<u16>> = vec![None; vertex_count];
    let mut new_order: Vec<usize> = Vec::with_capacity(vertex_count); // new_idx → old_idx
    let mut remapped_indices: Vec<u16> = Vec::with_capacity(indices.len());
    for &old in &indices {
        let entry = &mut remap[old as usize];
        if entry.is_none() {
            *entry = Some(new_order.len() as u16);
            new_order.push(old as usize);
        }
        remapped_indices.push(entry.unwrap());
    }
    // Any vertex not yet seen (can only happen on isolated boundary vertices
    // not referenced by any triangle) gets appended last.
    for v in 0..vertex_count {
        if remap[v].is_none() {
            remap[v] = Some(new_order.len() as u16);
            new_order.push(v);
        }
    }

    // Reorder vertex arrays to the new order.
    let u_raw: Vec<u16> = new_order.iter().map(|&i| u_raw[i]).collect();
    let v_raw: Vec<u16> = new_order.iter().map(|&i| v_raw[i]).collect();
    let h_raw: Vec<u16> = new_order.iter().map(|&i| h_raw[i]).collect();
    let ecef_positions: Vec<DVec3> = new_order.iter().map(|&i| ecef_positions[i]).collect();

    // Remap edge arrays.
    let remap_edge =
        |edge: Vec<u16>| -> Vec<u16> { edge.iter().map(|&v| remap[v as usize].unwrap()).collect() };
    let west_edge = remap_edge(west_edge);
    let south_edge = remap_edge(south_edge);
    let east_edge = remap_edge(east_edge);
    let north_edge = remap_edge(north_edge);

    // Recompute bounding sphere and HOP with reordered positions.
    // bs_center is the geometric center used for both the tile center header
    // field and the HOP direction.
    let (bs_center, bs_radius) = bounding_sphere(&ecef_positions);
    let hop = horizon_occlusion_point(bs_center, &ecef_positions, ellipsoid);

    // 9. Zigzag-delta encode vertex buffers
    let zz_buf = |raw: &[u16]| -> Vec<u16> {
        let mut out = Vec::with_capacity(raw.len());
        let mut prev = 0i32;
        for &v in raw {
            let delta = (v as i32 - prev) as i16;
            out.push(outil::codec::zigzag_encode(delta));
            prev = v as i32;
        }
        out
    };
    let u_enc = zz_buf(&u_raw);
    let v_enc = zz_buf(&v_raw);
    let h_enc = zz_buf(&h_raw);

    // 10. High-watermark encode triangle indices.
    //     After reindexing, every vertex is introduced exactly when the
    //     watermark equals its new index, so code=0 is always correct for
    //     first appearances.
    let hwm_indices = high_watermark_encode_u16(&remapped_indices);
    let triangle_count = remapped_indices.len() / 3;

    // 11. Write binary
    //
    // Alignment (spec §IndexData):
    //   IndexData16 (vc ≤ 65536): requires 2-byte alignment before triangleCount.
    //   IndexData32 (vc > 65536): requires 4-byte alignment before triangleCount.
    //
    // After the 92-byte header, vertex buffers occupy vc × 6 bytes.
    // 92 is even and vc×6 is always even, so 2-byte alignment is always
    // satisfied — no padding is needed for the 16-bit case.
    // For the 32-bit case we may need up to 2 bytes of padding.
    let use_32bit = vertex_count > 65536;
    let index_size = if use_32bit { 4usize } else { 2usize };
    let pre_index_offset = QuantizedMeshHeader::SIZE + vertex_count * 6;
    let align = if use_32bit { 4usize } else { 2usize };
    let padding = (align - pre_index_offset % align) % align;
    let capacity = pre_index_offset
        + padding
        + 4                   // triangleCount
        + hwm_indices.len() * index_size
        + 4 * 4               // edge count fields
        + (west_edge.len() + south_edge.len() + east_edge.len() + north_edge.len()) * index_size;

    let mut w = outil::io::BufferWriter::with_capacity(capacity);

    // Header
    w.write_le(bs_center.x);
    w.write_le(bs_center.y);
    w.write_le(bs_center.z);
    w.write_le(min_h as f32);
    w.write_le(max_h as f32);
    w.write_le(bs_center.x);
    w.write_le(bs_center.y);
    w.write_le(bs_center.z);
    w.write_le(bs_radius);
    w.write_le(hop.x);
    w.write_le(hop.y);
    w.write_le(hop.z);
    w.write_le(vertex_count as u32);

    // Vertex buffers (zigzag-delta encoded)
    for &v in &u_enc {
        w.write_le(v);
    }
    for &v in &v_enc {
        w.write_le(v);
    }
    for &v in &h_enc {
        w.write_le(v);
    }

    // Alignment padding before IndexData (see spec):
    //   16-bit: 2-byte alignment (always met; padding=0)
    //   32-bit: 4-byte alignment (pad if needed)
    if padding > 0 {
        w.align_to(align, 0);
    }

    // Triangle indices (high-watermark encoded)
    w.write_le(triangle_count as u32);
    if use_32bit {
        for &i in &hwm_indices {
            w.write_le(i as u32);
        }
    } else {
        for &i in &hwm_indices {
            w.write_le(i);
        }
    }

    // Edge index arrays
    let write_edge = |w: &mut outil::io::BufferWriter, edge: &[u16]| {
        w.write_le(edge.len() as u32);
        if use_32bit {
            for &i in edge {
                w.write_le(i as u32);
            }
        } else {
            for &i in edge {
                w.write_le(i);
            }
        }
    };
    write_edge(&mut w, &west_edge);
    write_edge(&mut w, &south_edge);
    write_edge(&mut w, &east_edge);
    write_edge(&mut w, &north_edge);

    w.finish()
}

/// High-watermark encode a `u16` index buffer.
///
/// **Precondition**: every vertex index must be introduced (first appear) in
/// strictly sequential order — i.e. the first occurrence of vertex `k` must
/// happen exactly when the running watermark equals `k`.  This is guaranteed
/// when the caller has reindexed the mesh in first-appearance order.
///
/// Encoding rule (matches the Cesium decoder):
/// - First appearance of vertex `k` (== watermark): emit `0`, advance watermark.
/// - Back-reference to previous vertex `k` (< watermark): emit `watermark - k`.
fn high_watermark_encode_u16(indices: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(indices.len());
    let mut highest = 0u16;
    for &idx in indices {
        if idx == highest {
            // First appearance of the next sequential vertex.
            out.push(0);
            highest = highest.wrapping_add(1);
        } else {
            // Back-reference to a previously introduced vertex.
            debug_assert!(
                idx < highest,
                "index {idx} skips watermark {highest} — reindex before encoding"
            );
            out.push(highest.wrapping_sub(idx));
        }
    }
    out
}

fn ecef_center(positions: &[DVec3]) -> DVec3 {
    if positions.is_empty() {
        return DVec3::ZERO;
    }
    let mut min = DVec3::splat(f64::INFINITY);
    let mut max = DVec3::splat(f64::NEG_INFINITY);
    for &p in positions {
        min = min.min(p);
        max = max.max(p);
    }
    (min + max) * 0.5
}

fn bounding_sphere(positions: &[DVec3]) -> (DVec3, f64) {
    let center = ecef_center(positions);
    let radius = positions
        .iter()
        .map(|&p| (p - center).length())
        .fold(0.0f64, f64::max);
    (center, radius)
}

fn horizon_occlusion_point(
    center: DVec3,
    positions: &[DVec3],
    ellipsoid: &terra::Ellipsoid,
) -> DVec3 {
    // The HOP must be expressed in the ellipsoid-scaled ECEF frame (where the
    // ellipsoid is a unit sphere).  We find the direction from the Earth centre
    // to the tile centre in scaled space, then compute the maximum projection
    // of all tile vertices onto that direction.  Any view-point along that
    // direction with a smaller magnitude is guaranteed to be below the horizon.
    let scaled_center = ellipsoid.transform_position_to_scaled_space(center);
    let scaled_dir = if scaled_center.length_squared() > 0.0 {
        scaled_center.normalize()
    } else {
        DVec3::Z
    };

    let mut max_mag: f64 = 1.0; // at minimum keep the ellipsoid surface (mag=1)
    for &p in positions {
        let scaled = ellipsoid.transform_position_to_scaled_space(p);
        let mag = scaled.dot(scaled_dir);
        if mag > max_mag {
            max_mag = mag;
        }
    }

    // Return in the scaled frame (ellipsoid = unit sphere).
    scaled_dir * max_mag
}

#[cfg(test)]
mod tests {
    use super::*;

    // BufferReader

    #[test]
    fn reader_reads_primitives() {
        let data: &[u8] = &[0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00];
        let mut r = BufferReader::new(data);
        assert_eq!(r.read_le::<u8>().unwrap(), 0x01);
        assert_eq!(r.read_le::<u8>().unwrap(), 0x02);
        assert_eq!(r.read_le::<u32>().unwrap(), 0x0000_0300);
        // 1 byte remains — a u8 read succeeds but a u32 read does not
        assert!(r.read_le::<u8>().is_ok());
        assert!(r.read_le::<u32>().is_err());
    }

    #[test]
    fn reader_seek_and_position() {
        let data = [0u8; 16];
        let mut r = BufferReader::new(&data);
        r.seek(8);
        assert_eq!(r.position(), 8);
        assert_eq!(r.remaining(), 8);
    }

    #[test]
    fn reader_align_to_4() {
        let data = [0u8; 16];
        let mut r = BufferReader::new(&data);
        r.seek(3);
        r.align_to_4();
        assert_eq!(r.position(), 4);
        r.seek(4);
        r.align_to_4();
        assert_eq!(r.position(), 4); // already aligned
    }

    #[test]
    fn reader_read_bytes_borrows() {
        let data: Vec<u8> = (0u8..8).collect();
        let mut r = BufferReader::new(&data);
        let slice = r.read_bytes(4).unwrap();
        assert_eq!(slice, &[0, 1, 2, 3]);
        assert_eq!(r.position(), 4);
    }

    // ── QuantizedMeshHeader ───────────────────────────────────────────────────

    #[test]
    fn header_parse_too_short() {
        let data = vec![0u8; 64];
        assert!(matches!(
            QuantizedMeshHeader::parse(&data),
            Err(QuantizedMeshError::HeaderTooShort(64))
        ));
    }

    #[test]
    fn header_parse_all_zeros() {
        let data = vec![0u8; 96];
        let h = QuantizedMeshHeader::parse(&data).unwrap();
        assert_eq!(h.center, [0.0; 3]);
        assert_eq!(h.vertex_count, 0);
    }

    #[test]
    fn zigzag_decode_roundtrip() {
        assert_eq!(outil::codec::zigzag_decode(0), 0);
        assert_eq!(outil::codec::zigzag_decode(1), -1);
        assert_eq!(outil::codec::zigzag_decode(2), 1);
        assert_eq!(outil::codec::zigzag_decode(3), -2);
        assert_eq!(outil::codec::zigzag_decode(4), 2);
    }

    #[test]
    fn high_watermark_simple() {
        let decoded = high_watermark_decode(vec![0u16, 0u16, 0u16]);
        assert_eq!(decoded, vec![0, 1, 2]);
    }

    #[test]
    fn oct_decode_normal_up() {
        let d = outil::codec::oct_decode(127, 127);
        let n = [d[0] as f32, d[1] as f32, d[2] as f32];
        assert!(n[2] > 0.9, "expected upward normal, got {n:?}");
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 0.01, "expected unit length, got {len}");
    }

    // ── Availability parsing ──────────────────────────────────────────────────

    #[test]
    fn parse_availability_basic() {
        let json = r#"{"available":[[{"startX":0,"startY":0,"endX":1,"endY":1}]]}"#;
        let mut out = Vec::new();
        parse_availability(json, 5, &mut out);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].level, 5);
        assert_eq!(out[0].start_x, 0);
        assert_eq!(out[0].end_y, 1);
    }

    #[test]
    fn parse_availability_invalid_json_is_silent() {
        let mut out = Vec::new();
        parse_availability("not json", 0, &mut out);
        assert!(out.is_empty());
    }

    #[test]
    fn parse_availability_multi_level() {
        let json = r#"{"available":[
            [{"startX":0,"startY":0,"endX":1,"endY":1}],
            [{"startX":2,"startY":2,"endX":3,"endY":3}]
        ]}"#;
        let mut out = Vec::new();
        parse_availability(json, 10, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].level, 10);
        assert_eq!(out[1].level, 11);
        assert_eq!(out[1].start_x, 2);
    }

    // ── Encoder ───────────────────────────────────────────────────────────────

    /// Build a minimal flat 2×2 tile and check the header parses correctly.
    #[test]
    fn encode_header_round_trip() {
        use std::f64::consts::PI;
        let ellipsoid = terra::Ellipsoid::wgs84();

        // Flat tile at sea level: 4 corners
        let heights = vec![0.0f64; 4];
        let encoded = encode_quantized_mesh(&QuantizedMeshInput {
            heights: &heights,
            grid_size: 2,
            west: 0.0,
            south: 0.0,
            east: PI / 180.0, // 1 degree
            north: PI / 180.0,
            ellipsoid: &ellipsoid,
        });

        let header = QuantizedMeshHeader::parse(&encoded).expect("header parse failed");
        assert_eq!(header.vertex_count, 4);
        assert!(
            (header.min_height - 0.0).abs() < 1.0,
            "min_height={}",
            header.min_height
        );
        assert!(
            (header.max_height - 0.0).abs() < 1.0,
            "max_height={}",
            header.max_height
        );
    }

    #[test]
    fn encode_size_grows_with_grid() {
        let ellipsoid = terra::Ellipsoid::wgs84();
        let make = |n: usize| {
            let h = vec![100.0f64; n * n];
            encode_quantized_mesh(&QuantizedMeshInput {
                heights: &h,
                grid_size: n,
                west: 0.0,
                south: 0.0,
                east: 0.01,
                north: 0.01,
                ellipsoid: &ellipsoid,
            })
            .len()
        };
        assert!(make(4) > make(2), "4×4 tile should be larger than 2×2");
        assert!(make(8) > make(4), "8×8 tile should be larger than 4×4");
    }

    #[test]
    fn encode_non_flat_heights() {
        let ellipsoid = terra::Ellipsoid::wgs84();
        // 3×3 tile with varying heights
        let heights = vec![0.0, 100.0, 200.0, 500.0, 1000.0, 500.0, 200.0, 100.0, 0.0];
        let encoded = encode_quantized_mesh(&QuantizedMeshInput {
            heights: &heights,
            grid_size: 3,
            west: 0.0,
            south: 0.0,
            east: 0.01,
            north: 0.01,
            ellipsoid: &ellipsoid,
        });

        let header = QuantizedMeshHeader::parse(&encoded).expect("header parse failed");
        assert_eq!(header.vertex_count, 9);
        assert!(
            header.min_height < header.max_height,
            "min should be < max for non-flat tile"
        );
        assert!(header.bounding_sphere_radius > 0.0);
    }
}

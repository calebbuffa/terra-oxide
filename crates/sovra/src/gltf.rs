//! Raster overlay UV generation and application for glTF models.
//!
//! reads ECEF `POSITION` per vertex, converts to geodetic via `Ellipsoid::ecef_to_cartographic`,
//! then projects to overlay UV space (Geographic or Web Mercator).
//!
//! This works correctly for any geometry - terrain, buildings, non-axis-aligned tiles -
//! with no assumptions about how TEXCOORD_0 is parameterised.
//!
//! # Architecture
//!
//! The two streaming primitives - [`extract_ecef_positions`] and
//! [`compute_overlay_uvs_from_positions`] - are pure functions that leave geometry
//! immutable.  Renderers receive GLB bytes once, then receive separate UV + pixel
//! payloads for each overlay attach/detach event, exactly as CesiumJS does with its
//! per-tile `TileImagery` array.
//!
//! [`apply_raster_overlay`] composes both primitives in-place - useful for offline
//! / batch pipelines that pre-bake tilesets to disk.

use terra::WebMercatorProjection;

use crate::overlay::{OverlayProjection, RasterOverlayTile};

/// Extract the raw ECEF `POSITION` attribute (`[f32; 3]`) from every primitive
/// in `model` that has one.
///
/// Returns one `Vec<[f32; 3]>` per qualifying primitive, in mesh-then-primitive
/// order.  Cache this alongside the GLB bytes so that
/// [`compute_overlay_uvs_from_positions`] can generate overlay UVs without
/// re-parsing the GLB on every overlay attach event.
pub fn extract_ecef_positions(model: &moderu::GltfModel) -> Vec<Vec<[f32; 3]>> {
    let mut result = Vec::new();
    for mesh in &model.meshes {
        for prim in &mesh.primitives {
            let Some(&pos_acc) = prim.attributes.get("POSITION") else {
                continue;
            };
            let positions: Vec<[f32; 3]> =
                match moderu::resolve_accessor::<[f32; 3]>(model, pos_acc) {
                    Ok(view) => view.iter().collect(),
                    Err(_) => continue,
                };
            result.push(positions);
        }
    }
    result
}

/// Compute projection-correct overlay UV coordinates from pre-extracted ECEF
/// position arrays.
///
/// - Each vertex position is transformed from local space to ECEF via `transform`.
/// - `ellipsoid.ecef_to_cartographic()` yields geodetic (lon, lat).
/// - For **Geographic** overlays: UV is a linear remap of (lon, lat) into the
///   overlay tile rectangle. V=0 = north, V=1 = south (glTF top-down convention).
/// - For **WebMercator** overlays: U is the same linear longitude remap;
///   V is computed in Mercator-Y space via
///   `WebMercatorProjection::geodetic_latitude_to_mercator_angle`.
/// - All output UVs are clamped to `[0, 1]`.
///
/// `positions` must be the output of [`extract_ecef_positions`] for the same tile.
pub fn compute_overlay_uvs_from_positions(
    positions: &[Vec<[f32; 3]>],
    tile: &RasterOverlayTile,
    ellipsoid: &terra::Ellipsoid,
    transform: glam::DMat4,
) -> Vec<Vec<[f32; 2]>> {
    let tile_rect = tile.rectangle;
    // Use `width()` so antimeridian-crossing rectangles (east < west) get a
    // positive span. Falls back to plain `east - west` for normal rectangles.
    let tile_lon_span = tile_rect.width().max(f64::EPSILON);
    let crosses_antimeridian = tile_rect.east < tile_rect.west;

    // Precompute north Y and span in the tile's native projection space.
    let (y_north, y_span) = match tile.projection {
        OverlayProjection::Geographic => {
            let n = tile_rect.north;
            let s = tile_rect.south;
            (n, (n - s).max(f64::EPSILON))
        }
        OverlayProjection::WebMercator => {
            let n = WebMercatorProjection::geodetic_latitude_to_mercator_angle(tile_rect.north);
            let s = WebMercatorProjection::geodetic_latitude_to_mercator_angle(tile_rect.south);
            (n, (n - s).max(f64::EPSILON))
        }
    };

    let project_lat = |lat: f64| -> f64 {
        match tile.projection {
            OverlayProjection::Geographic => lat,
            OverlayProjection::WebMercator => {
                WebMercatorProjection::geodetic_latitude_to_mercator_angle(lat)
            }
        }
    };

    positions
        .iter()
        .map(|prim_positions| {
            prim_positions
                .iter()
                .map(|&pos| {
                    let local = glam::DVec3::new(pos[0] as f64, pos[1] as f64, pos[2] as f64);
                    let ecef = transform.transform_point3(local);

                    let Some(carto) = ellipsoid.ecef_to_cartographic(ecef) else {
                        return [0.0f32, 0.0f32];
                    };

                    // Compute longitude relative to the tile's west edge, retries
                    // with `longitude +- 2\pi` when a vertex projects outside
                    // the tile rectangle near the antimeridian.
                    let mut delta = carto.longitude - tile_rect.west;
                    if crosses_antimeridian {
                        // West>0, east<0. Vertices in the eastern half have
                        // longitudes in [-\pi, east] and need +2\pi to land in
                        // [west, west + width].
                        if delta < 0.0 {
                            delta += std::f64::consts::TAU;
                        }
                    } else {
                        // Non-crossing rectangle: still try wrap if the vertex
                        // came out the wrong side of the antimeridian (e.g.
                        // longitude ~ -\pi for a tile at longitude ~ +\pi).
                        if delta < 0.0 {
                            let wrapped = delta + std::f64::consts::TAU;
                            if wrapped <= tile_lon_span {
                                delta = wrapped;
                            }
                        } else if delta > tile_lon_span {
                            let wrapped = delta - std::f64::consts::TAU;
                            if wrapped >= 0.0 {
                                delta = wrapped;
                            }
                        }
                    }

                    let u = (delta / tile_lon_span).clamp(0.0, 1.0) as f32;
                    let v =
                        ((y_north - project_lat(carto.latitude)) / y_span).clamp(0.0, 1.0) as f32;
                    [u, v]
                })
                .collect()
        })
        .collect()
}

/// Convenience wrapper: extracts ECEF positions from `model` and immediately
/// computes overlay UVs. Prefer caching the output of [`extract_ecef_positions`]
/// and calling [`compute_overlay_uvs_from_positions`] for streaming use.
pub fn compute_overlay_uvs(
    model: &moderu::GltfModel,
    tile: &RasterOverlayTile,
    ellipsoid: &terra::Ellipsoid,
    transform: glam::DMat4,
) -> Vec<Vec<[f32; 2]>> {
    compute_overlay_uvs_from_positions(&extract_ecef_positions(model), tile, ellipsoid, transform)
}

/// Encode the pixel data of `tile` to a PNG byte buffer.
///
/// Returns `None` if the pixel data is empty or encoding fails.
pub fn encode_overlay_png(tile: &RasterOverlayTile) -> Option<Vec<u8>> {
    use image::ImageEncoder;
    if tile.pixels.is_empty() {
        return None;
    }
    let mut buf = std::io::Cursor::new(Vec::with_capacity(tile.pixels.len() + 1024));
    let encoder = image::codecs::png::PngEncoder::new_with_quality(
        &mut buf,
        image::codecs::png::CompressionType::Fast,
        image::codecs::png::FilterType::NoFilter,
    );
    encoder
        .write_image(
            &tile.pixels,
            tile.width,
            tile.height,
            image::ColorType::Rgba8.into(),
        )
        .ok()?;
    Some(buf.into_inner())
}

/// Apply a raster overlay tile to a [`GltfModel`](moderu::GltfModel) **in place**.
///
/// This is the right tool for offline / batch pipelines that pre-bake overlay
/// textures into GLB files on disk.  For streaming renderers use
/// [`compute_overlay_uvs_from_positions`] instead - it leaves geometry immutable.
///
/// Generates a new `TEXCOORD_{uv_index+1}` attribute on every primitive using
/// ECEF-based UV projection, adds the overlay image as a texture, and assigns
/// it as the base-color texture on every material.
pub fn apply_raster_overlay(
    model: &mut moderu::GltfModel,
    tile: &RasterOverlayTile,
    uv_index: u32,
    ellipsoid: &terra::Ellipsoid,
    transform: glam::DMat4,
) -> bool {
    let overlay_texcoord_set = (uv_index + 1) as usize;
    let positions = extract_ecef_positions(model);

    let mut prim_uv_sets: Vec<(usize, usize, Vec<[f32; 2]>)> = Vec::new();
    let mut prim_idx_global = 0usize;
    for mesh_idx in 0..model.meshes.len() {
        for prim_idx in 0..model.meshes[mesh_idx].primitives.len() {
            if model.meshes[mesh_idx].primitives[prim_idx]
                .attributes
                .contains_key("POSITION")
            {
                if let Some(uvs) = positions.get(prim_idx_global) {
                    let overlay_uvs = compute_overlay_uvs_from_positions(
                        std::slice::from_ref(uvs),
                        tile,
                        ellipsoid,
                        transform,
                    )
                    .into_iter()
                    .next()
                    .unwrap_or_default();
                    prim_uv_sets.push((mesh_idx, prim_idx, overlay_uvs));
                }
                prim_idx_global += 1;
            }
        }
    }

    let attr_name = format!("TEXCOORD_{overlay_texcoord_set}");
    for (mesh_idx, prim_idx, overlay_uvs) in prim_uv_sets {
        let acc_idx = model.append_accessor(&overlay_uvs);
        model.meshes[mesh_idx].primitives[prim_idx]
            .attributes
            .insert(attr_name.clone(), acc_idx);
    }

    let png_bytes = match encode_overlay_png(tile) {
        Some(b) => b,
        None => return false,
    };

    if model.buffers.is_empty() {
        model.buffers.push(moderu::Buffer::default());
    }
    let buf = &mut model.buffers[0];
    while buf.data.len() % 4 != 0 {
        buf.data.push(0);
    }
    let byte_offset = buf.data.len();
    buf.data.extend_from_slice(&png_bytes);
    buf.byte_length = buf.data.len();

    let bv_index = model.buffer_views.len();
    model.buffer_views.push(moderu::BufferView {
        buffer: 0,
        byte_offset,
        byte_length: png_bytes.len(),
        ..Default::default()
    });

    let img_index = model.images.len();
    model.images.push(moderu::Image {
        buffer_view: Some(bv_index),
        mime_type: Some("image/png".into()),
        name: Some("overlay".into()),
        ..Default::default()
    });

    let sampler_index = model.samplers.len();
    model.samplers.push(moderu::Sampler {
        mag_filter: Some(9729),
        min_filter: Some(9729),
        wrap_s: 33071,
        wrap_t: 33071,
        ..Default::default()
    });

    let tex_index = model.textures.len();
    model.textures.push(moderu::Texture {
        source: Some(img_index),
        sampler: Some(sampler_index),
        ..Default::default()
    });

    let tex_info = moderu::TextureInfo {
        index: tex_index,
        tex_coord: overlay_texcoord_set,
        ..Default::default()
    };

    for mat in &mut model.materials {
        let pbr = mat
            .pbr_metallic_roughness
            .get_or_insert_with(Default::default);
        pbr.base_color_texture = Some(tex_info.clone());
        pbr.base_color_factor = vec![1.0, 1.0, 1.0, 1.0];
    }

    if model.materials.is_empty() {
        model.materials.push(moderu::Material {
            pbr_metallic_roughness: Some(moderu::MaterialPbrMetallicRoughness {
                base_color_texture: Some(tex_info),
                ..Default::default()
            }),
            ..Default::default()
        });
        for mesh in &mut model.meshes {
            for prim in &mut mesh.primitives {
                if prim.material.is_none() {
                    prim.material = Some(0);
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlay::{OverlayProjection, RasterOverlayTile};
    use std::f64::consts::PI;
    use std::sync::Arc;

    fn make_tile(west: f64, south: f64, east: f64, north: f64) -> RasterOverlayTile {
        RasterOverlayTile {
            pixels: Arc::from(Vec::<u8>::new().into_boxed_slice()),
            width: 0,
            height: 0,
            rectangle: terra::GlobeRectangle {
                west,
                south,
                east,
                north,
            },
            projection: OverlayProjection::Geographic,
        }
    }

    /// Vertex at lon = -179 degree should map to U ~ 1/3 inside a tile spanning
    /// lon ∈ [+170 degree, -175 degree] (crossing the antimeridian; 15 degree total width,
    /// -179 degree is 9 degree past +170 degree = east 6 degree before end).
    #[test]
    fn antimeridian_crossing_rectangle_wraps_u() {
        let deg = PI / 180.0;
        let tile = make_tile(170.0 * deg, -10.0 * deg, -175.0 * deg, 10.0 * deg);
        let ellipsoid = terra::Ellipsoid::wgs84();

        // Vertex on the ellipsoid at (-179 degree, 0 degree). Use a position shim: build
        // an ECEF cartographic-to-cartesian and feed directly.
        let carto = terra::Cartographic::new(-179.0 * deg, 0.0, 0.0);
        let ecef = ellipsoid.cartographic_to_ecef(carto);
        let positions = vec![vec![[ecef.x as f32, ecef.y as f32, ecef.z as f32]]];

        let uvs = compute_overlay_uvs_from_positions(
            &positions,
            &tile,
            &ellipsoid,
            glam::DMat4::IDENTITY,
        );
        let u = uvs[0][0][0];
        // width = 15 degree; lon −179 degree wraps to (−179 + 360) − 170 = 11 degree past west.
        // U = 11/15 ~ 0.7333.
        assert!(
            (u - 11.0 / 15.0).abs() < 0.01,
            "expected U ~ 0.733 for antimeridian wrap, got {u}",
        );
    }

    /// Sanity: a normal (non-crossing) rectangle still produces linear U.
    #[test]
    fn non_crossing_rectangle_linear_u() {
        let deg = PI / 180.0;
        let tile = make_tile(0.0, 0.0, 10.0 * deg, 10.0 * deg);
        let ellipsoid = terra::Ellipsoid::wgs84();
        let carto = terra::Cartographic::new(5.0 * deg, 5.0 * deg, 0.0);
        let ecef = ellipsoid.cartographic_to_ecef(carto);
        let positions = vec![vec![[ecef.x as f32, ecef.y as f32, ecef.z as f32]]];

        let uvs = compute_overlay_uvs_from_positions(
            &positions,
            &tile,
            &ellipsoid,
            glam::DMat4::IDENTITY,
        );
        let [u, v] = uvs[0][0];
        assert!((u - 0.5).abs() < 0.01, "U midpoint expected 0.5, got {u}");
        assert!((v - 0.5).abs() < 0.01, "V midpoint expected 0.5, got {v}");
    }
}

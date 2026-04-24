//! Geometry helpers: node transforms.

use crate::Node;
use glam::{DMat4, DQuat, DVec3};

/// Error type for tangent generation.
#[derive(Debug, thiserror::Error)]
pub enum TangentError {
    #[error("insufficient vertex data")]
    InsufficientData,
    #[error("degenerate geometry: no valid triangles")]
    DegenerateGeometry,
}

/// Compute per-vertex tangents for a mesh primitive.
///
/// Uses the standard UV-based tangent calculation (length-weighted average of
/// triangle contributions). If the inputs are too short for a single triangle,
/// returns [`TangentError::InsufficientData`]. If no valid triangles are found
/// (all degenerate), returns [`TangentError::DegenerateGeometry`].
///
/// Result is a `Vec` of `[f32; 4]` where `xyz` is the tangent direction and
/// `w` is the handedness (`+1` or `-1`).
pub fn generate_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tex_coords: &[[f32; 2]],
    indices: Option<&[u32]>,
) -> Result<Vec<[f32; 4]>, TangentError> {
    let n = positions.len();
    if n < 3 || normals.len() < n || tex_coords.len() < n {
        return Err(TangentError::InsufficientData);
    }

    let mut tan1 = vec![[0.0f64; 3]; n];
    let mut tan2 = vec![[0.0f64; 3]; n];
    let mut valid = false;

    let tri_indices: Box<dyn Iterator<Item = (usize, usize, usize)>> = if let Some(idx) = indices {
        Box::new(
            idx.chunks_exact(3)
                .map(|t| (t[0] as usize, t[1] as usize, t[2] as usize)),
        )
    } else {
        Box::new((0..n / 3).map(|i| (i * 3, i * 3 + 1, i * 3 + 2)))
    };

    for (ai, bi, ci) in tri_indices {
        if ai >= n || bi >= n || ci >= n {
            continue;
        }
        let p1 = positions[ai];
        let p2 = positions[bi];
        let p3 = positions[ci];
        let w1 = tex_coords[ai];
        let w2 = tex_coords[bi];
        let w3 = tex_coords[ci];

        let x1 = (p2[0] - p1[0]) as f64;
        let x2 = (p3[0] - p1[0]) as f64;
        let y1 = (p2[1] - p1[1]) as f64;
        let y2 = (p3[1] - p1[1]) as f64;
        let z1 = (p2[2] - p1[2]) as f64;
        let z2 = (p3[2] - p1[2]) as f64;

        let s1 = (w2[0] - w1[0]) as f64;
        let s2 = (w3[0] - w1[0]) as f64;
        let t1 = (w2[1] - w1[1]) as f64;
        let t2 = (w3[1] - w1[1]) as f64;

        let denom = s1 * t2 - s2 * t1;
        if denom.abs() < 1e-15 {
            continue;
        }
        valid = true;
        let r = 1.0 / denom;

        let sdir = [
            (t2 * x1 - t1 * x2) * r,
            (t2 * y1 - t1 * y2) * r,
            (t2 * z1 - t1 * z2) * r,
        ];
        let tdir = [
            (s1 * x2 - s2 * x1) * r,
            (s1 * y2 - s2 * y1) * r,
            (s1 * z2 - s2 * z1) * r,
        ];

        for &vi in &[ai, bi, ci] {
            tan1[vi][0] += sdir[0];
            tan1[vi][1] += sdir[1];
            tan1[vi][2] += sdir[2];
            tan2[vi][0] += tdir[0];
            tan2[vi][1] += tdir[1];
            tan2[vi][2] += tdir[2];
        }
    }

    if !valid {
        return Err(TangentError::DegenerateGeometry);
    }

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let n3 = normals[i];
        let nx = n3[0] as f64;
        let ny = n3[1] as f64;
        let nz = n3[2] as f64;

        let tx = tan1[i][0];
        let ty = tan1[i][1];
        let tz = tan1[i][2];

        // Gram-Schmidt orthogonalize
        let dot = nx * tx + ny * ty + nz * tz;
        let ox = tx - dot * nx;
        let oy = ty - dot * ny;
        let oz = tz - dot * nz;
        let len = (ox * ox + oy * oy + oz * oz).sqrt();
        let (ox, oy, oz) = if len > 1e-15 {
            (ox / len, oy / len, oz / len)
        } else {
            (1.0, 0.0, 0.0)
        };

        // Handedness: sign(cross(n, t) · tan2)
        let cx = ny * oz - nz * oy;
        let cy = nz * ox - nx * oz;
        let cz = nx * oy - ny * ox;
        let dot2 = cx * tan2[i][0] + cy * tan2[i][1] + cz * tan2[i][2];
        let w = if dot2 < 0.0 { -1.0f32 } else { 1.0f32 };

        out.push([ox as f32, oy as f32, oz as f32, w]);
    }
    Ok(out)
}

pub(crate) fn get_node_transform(node: &Node) -> Option<DMat4> {
    if node.matrix.len() == 16 {
        let m: [f64; 16] = node.matrix.as_slice().try_into().ok()?;
        return Some(DMat4::from_cols_array(&m));
    }
    let t = if node.translation.len() == 3 {
        DVec3::new(
            node.translation[0],
            node.translation[1],
            node.translation[2],
        )
    } else {
        DVec3::ZERO
    };
    let r = if node.rotation.len() == 4 {
        DQuat::from_xyzw(
            node.rotation[0],
            node.rotation[1],
            node.rotation[2],
            node.rotation[3],
        )
        .normalize()
    } else {
        DQuat::IDENTITY
    };
    let s = if node.scale.len() == 3 {
        DVec3::new(node.scale[0], node.scale[1], node.scale[2])
    } else {
        DVec3::ONE
    };
    Some(DMat4::from_scale_rotation_translation(s, r, t))
}

/// Set the local-space transformation matrix for a node (overwrites TRS components).
pub(crate) fn set_node_transform(node: &mut Node, mat: DMat4) {
    node.matrix = mat.to_cols_array().to_vec();
    node.translation.clear();
    node.rotation.clear();
    node.scale.clear();
}

impl crate::Node {
    /// Get the local-space transformation matrix for this node.
    pub fn transform(&self) -> Option<DMat4> {
        get_node_transform(self)
    }

    /// Overwrite this node's transform with a matrix (clears TRS components).
    pub fn set_transform(&mut self, mat: DMat4) {
        set_node_transform(self, mat);
    }

    /// Zero-copy view over the `TRANSLATION` accessor from `EXT_mesh_gpu_instancing`.
    pub fn instancing_translation<'a>(
        &self,
        model: &'a crate::GltfModel,
    ) -> Result<crate::AccessorView<'a, glam::Vec3>, crate::AccessorViewError> {
        let ext = self
            .extensions
            .get("EXT_mesh_gpu_instancing")
            .ok_or_else(|| {
                crate::AccessorViewError::MissingAttribute(
                    "EXT_mesh_gpu_instancing not present".into(),
                )
            })?;
        let acc_idx = ext
            .get("attributes")
            .and_then(|a| a.get("TRANSLATION"))
            .and_then(|v| v.as_u64())
            .ok_or_else(|| {
                crate::AccessorViewError::MissingAttribute(
                    "no TRANSLATION in EXT_mesh_gpu_instancing".into(),
                )
            })? as usize;
        crate::resolve_accessor(model, acc_idx)
    }
}

/// Compute area-weighted, per-vertex normals from an indexed triangle soup.
///
/// For each triangle the cross product of two edges gives a face normal
/// proportional to the triangle's area. That vector is accumulated into each
/// of the three vertex slots and then normalised. Vertices that appear in no
/// triangle are assigned the Y-up fallback `[0, 1, 0]`.
///
/// Input positions are `[x, y, z]` tuples of `f32`; the computation is done
/// in `f64` to reduce accumulation error.
pub fn flat_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut out = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (ai, bi, ci) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if ai >= positions.len() || bi >= positions.len() || ci >= positions.len() {
            continue;
        }
        let pa = DVec3::new(
            positions[ai][0] as f64,
            positions[ai][1] as f64,
            positions[ai][2] as f64,
        );
        let pb = DVec3::new(
            positions[bi][0] as f64,
            positions[bi][1] as f64,
            positions[bi][2] as f64,
        );
        let pc = DVec3::new(
            positions[ci][0] as f64,
            positions[ci][1] as f64,
            positions[ci][2] as f64,
        );
        let n = (pb - pa).cross(pc - pa).normalize_or_zero();
        for &idx in tri {
            let o = &mut out[idx as usize];
            o[0] += n.x as f32;
            o[1] += n.y as f32;
            o[2] += n.z as f32;
        }
    }
    for n in &mut out {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if len > f32::EPSILON {
            n[0] /= len;
            n[1] /= len;
            n[2] /= len;
        } else {
            *n = [0.0, 1.0, 0.0];
        }
    }
    out
}

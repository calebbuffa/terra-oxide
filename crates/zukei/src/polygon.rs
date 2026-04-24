//! 2D polygon geometry utilities.

use glam::DVec2;

/// Returns `true` if `p` is inside the 2D polygon using the winding-number rule.
///
/// Returns `false` for degenerate polygons with fewer than 3 vertices.
pub fn point_in_polygon_2d(p: DVec2, verts: &[DVec2]) -> bool {
    if verts.len() < 3 {
        return false;
    }
    let mut winding = 0i32;
    let n = verts.len();
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        if a.y <= p.y {
            if b.y > p.y && cross2(a, b, p) > 0.0 {
                winding += 1;
            }
        } else if b.y <= p.y && cross2(a, b, p) < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

/// Minimum 2D distance from `p` to the nearest edge of a polygon.
///
/// Does not test whether `p` is inside; call [`point_in_polygon_2d`] first and
/// return `0.0` if it returns `true`.
pub fn polygon_boundary_distance_2d(p: DVec2, verts: &[DVec2]) -> f64 {
    let n = verts.len();
    if n == 0 {
        return f64::MAX;
    }
    let mut min_dist = f64::MAX;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        min_dist = min_dist.min(point_to_segment_dist_2d(p, a, b));
    }
    min_dist
}

/// Shortest distance from point `p` to segment `[a, b]`.
#[inline]
pub fn point_to_segment_dist_2d(p: DVec2, a: DVec2, b: DVec2) -> f64 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    if len_sq < f64::EPSILON {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    p.distance(a + t * ab)
}

/// 2D cross product of vectors (a->b) and (a->p).
#[inline]
pub fn cross2(a: DVec2, b: DVec2, p: DVec2) -> f64 {
    (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
}

/// Triangulates a simple (non-self-intersecting) polygon using the O(n^2)
/// ear-clipping algorithm.
///
/// Returns a flat list of triangle vertex indices into the original `polygon`
/// slice (groups of three).  Degenerate polygons with fewer than 3 vertices
/// return an empty list.  Self-intersecting polygons terminate early with a
/// partial triangulation.
pub fn ear_clip(polygon: &[DVec2]) -> Vec<u32> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    if n == 3 {
        return vec![0, 1, 2];
    }

    // Signed area to determine winding; we need CCW for the ear test.
    let signed_area: f64 = {
        let mut area = 0.0f64;
        for i in 0..n {
            let a = polygon[i];
            let b = polygon[(i + 1) % n];
            area += a.x * b.y - b.x * a.y;
        }
        area / 2.0
    };

    let mut ring: Vec<usize> = (0..n).collect();
    if signed_area < 0.0 {
        ring.reverse();
    }

    let mut indices = Vec::with_capacity((n - 2) * 3);
    let mut attempts = 0usize;
    let mut i = 0usize;

    while ring.len() > 3 {
        let len = ring.len();
        let prev = ring[(i + len - 1) % len];
        let curr = ring[i % len];
        let next = ring[(i + 1) % len];

        if is_ear(polygon, &ring, prev, curr, next) {
            indices.push(prev as u32);
            indices.push(curr as u32);
            indices.push(next as u32);
            ring.remove(i % len);
            attempts = 0;
        } else {
            i += 1;
            attempts += 1;
            if attempts > len {
                // Polygon is degenerate (self-intersecting); bail out.
                break;
            }
        }
    }

    if ring.len() == 3 {
        indices.push(ring[0] as u32);
        indices.push(ring[1] as u32);
        indices.push(ring[2] as u32);
    }

    indices
}

/// Returns `true` when vertex `curr` forms an ear of the polygon reduced to `ring`.
fn is_ear(polygon: &[DVec2], ring: &[usize], prev: usize, curr: usize, next: usize) -> bool {
    let a = polygon[prev];
    let b = polygon[curr];
    let c = polygon[next];

    // The ear triangle must be CCW.
    if cross2(a, b, c) <= 0.0 {
        return false;
    }

    // No other ring vertex may lie strictly inside this triangle.
    for &idx in ring {
        if idx == prev || idx == curr || idx == next {
            continue;
        }
        if point_in_triangle_strict(polygon[idx], a, b, c) {
            return false;
        }
    }
    true
}

/// Returns `true` if `p` is strictly inside triangle `(a, b, c)` (CCW).
fn point_in_triangle_strict(p: DVec2, a: DVec2, b: DVec2, c: DVec2) -> bool {
    let d1 = cross2(a, b, p);
    let d2 = cross2(b, c, p);
    let d3 = cross2(c, a, p);
    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_neg && has_pos)
}

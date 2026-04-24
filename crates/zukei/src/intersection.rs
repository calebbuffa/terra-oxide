use glam::DVec3;

use crate::aabb::AxisAlignedBoundingBox;
use crate::obb::OrientedBoundingBox;
use crate::plane::Plane;
use crate::ray::Ray;
use crate::sphere::BoundingSphere;
use outil::{EPSILON8, EPSILON12, EPSILON15};

/// Intersect a ray with a plane (Hessian normal form). Returns the intersection
/// point, or `None` if the ray is parallel to the plane or points away from it.
///
/// Mirrors `CesiumGeometry::IntersectionTests::rayPlane`.
#[inline]
pub fn ray_plane(ray: &Ray, plane: &Plane) -> Option<DVec3> {
    let denom = plane.normal.dot(ray.direction);
    if denom.abs() < EPSILON15 {
        return None; // Ray is parallel to the plane.
    }
    let t = (-plane.distance - plane.normal.dot(ray.origin)) / denom;
    if t < 0.0 {
        return None; // Intersection is behind the ray origin.
    }
    Some(ray.origin + ray.direction * t)
}

/// Intersect a ray with an ellipsoid defined by its radii vector.
///
/// Returns `Some((t_near, t_far))` — the two parametric distances along the
/// ray — or `None` when there is no real intersection.
///
/// Mirrors `CesiumGeometry::IntersectionTests::rayEllipsoid`.
pub fn ray_ellipsoid(ray: &Ray, radii: DVec3) -> Option<(f64, f64)> {
    if radii.x == 0.0 || radii.y == 0.0 || radii.z == 0.0 {
        return None;
    }
    let inv_r = DVec3::ONE / radii;
    let q = inv_r * ray.origin;
    let w = inv_r * ray.direction;

    let q2 = q.length_squared();
    let qw = q.dot(w);

    if q2 > 1.0 {
        // Outside ellipsoid.
        if qw >= 0.0 {
            return None; // Looking outward or tangent.
        }
        let qw2 = qw * qw;
        let difference = q2 - 1.0; // > 0
        let w2 = w.length_squared();
        let product = w2 * difference;
        if qw2 < product {
            return None; // Imaginary roots.
        }
        if qw2 > product {
            // Two distinct intersections.
            let discriminant = qw2 - product;
            let temp = -qw + discriminant.sqrt(); // avoid cancellation
            let root0 = temp / w2;
            let root1 = difference / temp;
            return Some(if root0 < root1 {
                (root0, root1)
            } else {
                (root1, root0)
            });
        }
        // Repeated root (tangent).
        let root = (difference / w2).sqrt();
        Some((root, root))
    } else if q2 < 1.0 {
        // Inside ellipsoid — one forward intersection.
        let difference = q2 - 1.0; // < 0
        let w2 = w.length_squared();
        let product = w2 * difference; // < 0
        let discriminant = qw * qw - product; // > 0
        let temp = -qw + discriminant.sqrt(); // > 0
        Some((0.0, temp / w2))
    } else {
        // q2 == 1.0: on the surface.
        if qw < 0.0 {
            let w2 = w.length_squared();
            Some((0.0, -qw / w2))
        } else {
            None // Looking outward or tangent.
        }
    }
}

/// Test whether a 2-D point lies inside or on the boundary of a 2-D triangle.
///
/// Uses the barycentric-coordinate method.
///
/// Mirrors `CesiumGeometry::IntersectionTests::pointInTriangle` (2-D overload).
#[inline]
pub fn point_in_triangle_2d(point: [f64; 2], p0: [f64; 2], p1: [f64; 2], p2: [f64; 2]) -> bool {
    let [px, py] = point;
    let [ax, ay] = [p0[0], p0[1]];
    let v0 = [p2[0] - ax, p2[1] - ay];
    let v1 = [p1[0] - ax, p1[1] - ay];
    let v2 = [px - ax, py - ay];

    let dot00 = v0[0] * v0[0] + v0[1] * v0[1];
    let dot01 = v0[0] * v1[0] + v0[1] * v1[1];
    let dot02 = v0[0] * v2[0] + v0[1] * v2[1];
    let dot11 = v1[0] * v1[0] + v1[1] * v1[1];
    let dot12 = v1[0] * v2[0] + v1[1] * v2[1];

    let inv_denom = dot00 * dot11 - dot01 * dot01;
    if inv_denom.abs() < EPSILON15 {
        return false; // Degenerate triangle.
    }
    let u = (dot11 * dot02 - dot01 * dot12) / inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) / inv_denom;
    u >= 0.0 && v >= 0.0 && u + v <= 1.0
}

/// Test whether a 3-D point lies inside or on the boundary of a 3-D triangle.
///
/// The point is first projected onto the triangle's plane; returns `false` if
/// the triangle is degenerate.
///
/// Mirrors `CesiumGeometry::IntersectionTests::pointInTriangle` (3-D overload).
#[inline]
pub fn point_in_triangle_3d(point: DVec3, p0: DVec3, p1: DVec3, p2: DVec3) -> bool {
    point_in_triangle_3d_barycentric(point, p0, p1, p2).is_some()
}

/// Test whether a 3-D point lies inside or on the boundary of a 3-D triangle,
/// returning the barycentric coordinates `(u, v, w)` on success.
///
/// Returns `None` if the triangle is degenerate or the point is outside.
///
/// Mirrors `CesiumGeometry::IntersectionTests::pointInTriangle` (barycentric overload).
pub fn point_in_triangle_3d_barycentric(
    point: DVec3,
    p0: DVec3,
    p1: DVec3,
    p2: DVec3,
) -> Option<(f64, f64, f64)> {
    let v0 = p2 - p0;
    let v1 = p1 - p0;
    let v2 = point - p0;

    let dot00 = v0.dot(v0);
    let dot01 = v0.dot(v1);
    let dot02 = v0.dot(v2);
    let dot11 = v1.dot(v1);
    let dot12 = v1.dot(v2);

    let inv_denom = dot00 * dot11 - dot01 * dot01;
    if inv_denom.abs() < EPSILON15 {
        return None;
    }
    let u = (dot11 * dot02 - dot01 * dot12) / inv_denom;
    let v = (dot00 * dot12 - dot01 * dot02) / inv_denom;
    if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
        Some((1.0 - u - v, v, u))
    } else {
        None
    }
}

/// Intersect a ray with a triangle using the Möller–Trumbore algorithm.
/// Returns the `t` parameter along the ray, or `None` if no hit.
///
/// When `cull_back_faces` is `true`, intersections where the ray hits the
/// back face (same side as the normal) are rejected.
///
/// Mirrors `CesiumGeometry::IntersectionTests::rayTriangleParametric`.
#[inline]
pub fn ray_triangle_parametric(
    ray: &Ray,
    p0: DVec3,
    p1: DVec3,
    p2: DVec3,
    cull_back_faces: bool,
) -> Option<f64> {
    let edge0 = p1 - p0;
    let edge1 = p2 - p0;
    let p = ray.direction.cross(edge1);
    let det = edge0.dot(p);

    if cull_back_faces {
        if det < EPSILON8 {
            return None;
        }
        let tvec = ray.origin - p0;
        let u = tvec.dot(p);
        if u < 0.0 || u > det {
            return None;
        }
        let q = tvec.cross(edge0);
        let v = ray.direction.dot(q);
        if v < 0.0 || u + v > det {
            return None;
        }
        Some(edge1.dot(q) / det)
    } else {
        if det.abs() < EPSILON8 {
            return None;
        }
        let inv_det = 1.0 / det;
        let tvec = ray.origin - p0;
        let u = tvec.dot(p) * inv_det;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = tvec.cross(edge0);
        let v = ray.direction.dot(q) * inv_det;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        Some(edge1.dot(q) * inv_det)
    }
}

/// Intersect a ray with a bounding sphere. Returns the `t` parameter of the
/// nearest intersection point, or `None` if no intersection.
/// Only returns hits with `t >= 0` (in front of the ray).
#[inline]
pub fn ray_sphere(ray: &Ray, sphere: &BoundingSphere) -> Option<f64> {
    let oc = ray.origin - sphere.center;
    let b = oc.dot(ray.direction);
    let c = oc.length_squared() - sphere.radius * sphere.radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();
    let t1 = -b - sqrt_d;
    if t1 >= 0.0 {
        return Some(t1);
    }
    let t2 = -b + sqrt_d;
    if t2 >= 0.0 {
        return Some(t2);
    }
    None
}

/// Intersect a ray with an axis-aligned bounding box (slab method).
/// Returns the `t` parameter of the nearest intersection, or `None`.
///
/// Unlike [`ray_aabb`], this function returns `t_enter` even when the ray
/// origin is inside the box (`t_enter < 0`).  Use it when you need the
/// signed parametric interval, e.g. to compute both entry and exit points.
///
/// Mirrors `CesiumGeometry::IntersectionTests::rayAABBParametric`.
#[inline]
pub fn ray_aabb_parametric(ray: &Ray, aabb: &AxisAlignedBoundingBox) -> Option<(f64, f64)> {
    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;

    for axis in 0..3 {
        let d = ray.direction[axis];
        let o = ray.origin[axis];
        let lo = aabb.min[axis];
        let hi = aabb.max[axis];

        if d.abs() <= EPSILON15 {
            if o < lo || o > hi {
                return None;
            }
            continue;
        }

        let inv = 1.0 / d;
        let mut t1 = (lo - o) * inv;
        let mut t2 = (hi - o) * inv;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
        }
        if t1 > t_enter {
            t_enter = t1;
        }
        if t2 < t_exit {
            t_exit = t2;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    if t_exit < 0.0 {
        return None;
    }

    Some((t_enter, t_exit))
}

/// Intersect a ray with an axis-aligned bounding box (slab method).
/// Returns the `t` parameter of the nearest intersection, or `None`.
#[inline]
pub fn ray_aabb(ray: &Ray, aabb: &AxisAlignedBoundingBox) -> Option<f64> {
    // Slab method. For axes where the ray direction is (near) zero we must
    // handle the slab separately: the origin must lie within [min, max] on
    // that axis, otherwise there can be no intersection. Computing
    // `INFINITY * signum(0)` here previously produced NaN and silently
    // corrupted the whole test.

    let mut t_enter = f64::NEG_INFINITY;
    let mut t_exit = f64::INFINITY;

    for axis in 0..3 {
        let d = ray.direction[axis];
        let o = ray.origin[axis];
        let lo = aabb.min[axis];
        let hi = aabb.max[axis];

        if d.abs() <= EPSILON15 {
            // Ray parallel to this slab - must already be inside it.
            if o < lo || o > hi {
                return None;
            }
            continue;
        }

        let inv = 1.0 / d;
        let mut t1 = (lo - o) * inv;
        let mut t2 = (hi - o) * inv;
        if t1 > t2 {
            core::mem::swap(&mut t1, &mut t2);
        }
        if t1 > t_enter {
            t_enter = t1;
        }
        if t2 < t_exit {
            t_exit = t2;
        }
        if t_enter > t_exit {
            return None;
        }
    }

    if t_exit < 0.0 {
        return None;
    }

    if t_enter >= 0.0 {
        Some(t_enter)
    } else {
        Some(t_exit)
    }
}

/// Intersect a ray with an oriented bounding box.
///
/// Transforms the ray into the OBB's normalised local space (where the box is
/// the unit cube `[-1, 1]^3`) and applies the AABB slab test.  The same
/// parameter `t` parameterises both spaces because the transform is linear.
///
/// Returns `None` if the OBB has a singular (degenerate) half-axes matrix -
/// e.g. a zero-thickness box - since the linear transform is not invertible.
#[inline]
pub fn ray_obb(ray: &Ray, obb: &OrientedBoundingBox) -> Option<f64> {
    // Guard against singular half-axes (flattened boxes) - `DMat3::inverse`
    // on a singular matrix returns NaN/INF entries that then poison the
    // downstream slab test. Bail out cleanly instead.
    if obb.half_axes.determinant().abs() < EPSILON12 {
        return None;
    }
    let inv = obb.half_axes.inverse();
    let local_ray = Ray {
        origin: inv * (ray.origin - obb.center),
        direction: inv * ray.direction,
    };
    let unit_cube = AxisAlignedBoundingBox::new(-DVec3::ONE, DVec3::ONE);
    ray_aabb(&local_ray, &unit_cube)
}

/// Intersect a ray with a triangle using the Möller–Trumbore algorithm.
/// Returns the `t` parameter along the ray, or `None` if no hit.
///
/// Vertices are specified counter-clockwise when viewed from the front face.
/// This function tests both front and back faces.
#[inline]
pub fn ray_triangle(ray: &Ray, v0: DVec3, v1: DVec3, v2: DVec3) -> Option<f64> {
    ray_triangle_parametric(ray, v0, v1, v2, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DMat3;

    #[test]
    fn ray_plane_hit() {
        // Ray along +Z, plane at Z=5 (normal=+Z, distance=-5).
        let ray = Ray::new(DVec3::ZERO, DVec3::Z);
        let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 5.0), DVec3::Z);
        let hit = ray_plane(&ray, &plane).unwrap();
        assert!((hit.z - 5.0).abs() < 1e-12, "z={}", hit.z);
    }

    #[test]
    fn ray_plane_parallel_returns_none() {
        let ray = Ray::new(DVec3::ZERO, DVec3::X);
        let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 5.0), DVec3::Z);
        assert!(ray_plane(&ray, &plane).is_none());
    }

    #[test]
    fn ray_plane_behind_origin_returns_none() {
        let ray = Ray::new(DVec3::new(0.0, 0.0, 10.0), DVec3::Z);
        // Plane at Z=5, ray going away from it.
        let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 5.0), DVec3::Z);
        assert!(ray_plane(&ray, &plane).is_none());
    }

    #[test]
    fn ray_ellipsoid_outside_hits_twice() {
        // Ray along +X, unit sphere, origin at (-5,0,0).
        let ray = Ray::new(DVec3::new(-5.0, 0.0, 0.0), DVec3::X);
        let radii = DVec3::ONE;
        let (t0, t1) = ray_ellipsoid(&ray, radii).unwrap();
        assert!((t0 - 4.0).abs() < 1e-10, "t0={t0}");
        assert!((t1 - 6.0).abs() < 1e-10, "t1={t1}");
    }

    #[test]
    fn ray_ellipsoid_inside_hits_once_forward() {
        // Ray origin inside unit sphere.
        let ray = Ray::new(DVec3::ZERO, DVec3::X);
        let radii = DVec3::ONE;
        let (t0, t1) = ray_ellipsoid(&ray, radii).unwrap();
        assert!(t0 < 1e-10, "t0={t0} should be ~0");
        assert!((t1 - 1.0).abs() < 1e-10, "t1={t1}");
    }

    #[test]
    fn ray_ellipsoid_outside_miss() {
        // Ray along +Z, offset in X past the sphere.
        let ray = Ray::new(DVec3::new(2.0, 0.0, -5.0), DVec3::Z);
        assert!(ray_ellipsoid(&ray, DVec3::ONE).is_none());
    }

    #[test]
    fn point_in_triangle_2d_inside() {
        assert!(point_in_triangle_2d(
            [0.5, 0.25],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ));
    }

    #[test]
    fn point_in_triangle_2d_outside() {
        assert!(!point_in_triangle_2d(
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ));
    }

    #[test]
    fn point_in_triangle_3d_inside() {
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(1.0, 0.0, 0.0);
        let v2 = DVec3::new(0.0, 1.0, 0.0);
        assert!(point_in_triangle_3d(
            DVec3::new(0.25, 0.25, 0.0),
            v0,
            v1,
            v2
        ));
    }

    #[test]
    fn point_in_triangle_3d_outside() {
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(1.0, 0.0, 0.0);
        let v2 = DVec3::new(0.0, 1.0, 0.0);
        assert!(!point_in_triangle_3d(DVec3::new(1.0, 1.0, 0.0), v0, v1, v2));
    }

    #[test]
    fn ray_triangle_parametric_culled_back_face() {
        // Ray pointing in +Z from z=-5. Back-face = CW winding when viewed
        // from the ray origin (-Z side), achieved by swapping v1/v2 so that
        // the Möller-Trumbore determinant is negative.
        let ray = Ray::new(DVec3::new(0.25, 0.25, -5.0), DVec3::Z);
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(1.0, 0.0, 0.0); // swapped -> back-face for +Z ray
        let v2 = DVec3::new(0.0, 1.0, 0.0);
        assert!(ray_triangle_parametric(&ray, v0, v1, v2, true).is_none());
    }

    #[test]
    fn ray_triangle_parametric_no_cull_hits() {
        let ray = Ray::new(DVec3::new(0.25, 0.25, -5.0), DVec3::Z);
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(0.0, 1.0, 0.0);
        let v2 = DVec3::new(1.0, 0.0, 0.0);
        let t = ray_triangle_parametric(&ray, v0, v1, v2, false).unwrap();
        assert!((t - 5.0).abs() < 1e-10, "t={t}");
    }

    #[test]
    fn ray_misses_sphere() {
        let ray = Ray::new(DVec3::new(0.0, 10.0, -10.0), DVec3::Z);
        let sphere = BoundingSphere::new(DVec3::ZERO, 1.0);
        assert!(ray_sphere(&ray, &sphere).is_none());
    }

    #[test]
    fn ray_inside_sphere() {
        let ray = Ray::new(DVec3::ZERO, DVec3::X);
        let sphere = BoundingSphere::new(DVec3::ZERO, 5.0);
        let t = ray_sphere(&ray, &sphere).unwrap();
        assert!((t - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ray_hits_aabb() {
        let ray = Ray::new(DVec3::new(-5.0, 0.5, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        let t = ray_aabb(&ray, &aabb).unwrap();
        assert!((t - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ray_misses_aabb() {
        let ray = Ray::new(DVec3::new(-5.0, 5.0, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        assert!(ray_aabb(&ray, &aabb).is_none());
    }

    #[test]
    fn ray_hits_triangle() {
        let ray = Ray::new(DVec3::new(0.25, 0.25, -5.0), DVec3::Z);
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(1.0, 0.0, 0.0);
        let v2 = DVec3::new(0.0, 1.0, 0.0);
        let t = ray_triangle(&ray, v0, v1, v2).unwrap();
        assert!((t - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ray_misses_triangle() {
        let ray = Ray::new(DVec3::new(2.0, 2.0, -5.0), DVec3::Z);
        let v0 = DVec3::new(0.0, 0.0, 0.0);
        let v1 = DVec3::new(1.0, 0.0, 0.0);
        let v2 = DVec3::new(0.0, 1.0, 0.0);
        assert!(ray_triangle(&ray, v0, v1, v2).is_none());
    }

    #[test]
    fn ray_aabb_parametric_returns_interval() {
        let ray = Ray::new(DVec3::new(-5.0, 0.5, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        let (t0, t1) = ray_aabb_parametric(&ray, &aabb).unwrap();
        assert!((t0 - 5.0).abs() < 1e-12, "t0={t0}");
        assert!((t1 - 6.0).abs() < 1e-12, "t1={t1}");
    }

    #[test]
    fn ray_aabb_parametric_inside_returns_negative_enter() {
        // Ray origin inside the box at (0.5, 0.5, 0.5) going +X.
        let ray = Ray::new(DVec3::splat(0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        let (t0, t1) = ray_aabb_parametric(&ray, &aabb).unwrap();
        assert!(
            t0 < 0.0,
            "t_enter should be negative (behind origin): t0={t0}"
        );
        assert!((t1 - 0.5).abs() < 1e-12, "t1={t1}");
    }

    #[test]
    fn ray_aabb_parametric_miss_returns_none() {
        let ray = Ray::new(DVec3::new(-5.0, 5.0, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        assert!(ray_aabb_parametric(&ray, &aabb).is_none());
    }

    /// Regression: a ray with a direction component of exactly zero used to
    /// produce `INFINITY * 0 = NaN` in `inv_dir`, corrupting every slab test.
    #[test]
    fn ray_axis_aligned_hits_aabb_without_nan() {
        // Pure +X ray passing through the centre of the box.
        let ray = Ray::new(DVec3::new(-5.0, 0.5, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        let t = ray_aabb(&ray, &aabb).expect("should hit");
        assert!((t - 5.0).abs() < 1e-12);

        // Same thing for +Y and +Z.
        let ry = Ray::new(DVec3::new(0.5, -5.0, 0.5), DVec3::Y);
        assert!((ray_aabb(&ry, &aabb).unwrap() - 5.0).abs() < 1e-12);
        let rz = Ray::new(DVec3::new(0.5, 0.5, -5.0), DVec3::Z);
        assert!((ray_aabb(&rz, &aabb).unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn ray_parallel_to_slab_outside_box_misses() {
        // Ray parallel to X axis but origin above the box in Y \u2014 must miss,
        // not produce NaN.
        let ray = Ray::new(DVec3::new(-5.0, 10.0, 0.5), DVec3::X);
        let aabb = AxisAlignedBoundingBox::new(DVec3::ZERO, DVec3::ONE);
        assert!(ray_aabb(&ray, &aabb).is_none());
    }

    /// Regression: a degenerate OBB (one half-axis collapsed) previously
    /// triggered `DMat3::inverse()` to return NaN entries, silently
    /// poisoning culling results.
    #[test]
    fn ray_obb_with_singular_half_axes_returns_none() {
        let obb = OrientedBoundingBox::new(
            DVec3::ZERO,
            DMat3::from_cols(DVec3::X, DVec3::Y, DVec3::ZERO),
        );
        let ray = Ray::new(DVec3::new(-5.0, 0.0, 0.0), DVec3::X);
        assert!(ray_obb(&ray, &obb).is_none());
    }
}

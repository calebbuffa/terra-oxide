//! Oriented bounding box.

use glam::{DMat3, DMat4, DQuat, DVec3};

use crate::aabb::AxisAlignedBoundingBox;
use crate::culling::CullingResult;
use crate::plane::Plane;
use crate::sphere::BoundingSphere;

/// An oriented bounding box defined by a center and a half-axes matrix.
///
/// The columns of `half_axes` are the three half-axis vectors of the box in
/// world space.  A point `p` is inside the box when, for each column `c`,
/// `|dot(p - center, c)| <= dot(c, c)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientedBoundingBox {
    pub center: DVec3,
    /// Columns are the three oriented half-axis vectors in world space.
    pub half_axes: DMat3,
}

impl OrientedBoundingBox {
    /// Create an OBB from a center point and a half-axes matrix.
    #[inline]
    pub fn new(center: DVec3, half_axes: DMat3) -> Self {
        Self { center, half_axes }
    }

    /// Create an OBB from a center, rotation quaternion, and half-sizes along
    /// each local axis.
    pub fn from_quat(center: DVec3, rotation: DQuat, half_size: DVec3) -> Self {
        let rot = DMat3::from_quat(rotation);
        Self {
            center,
            half_axes: DMat3::from_cols(
                rot.x_axis * half_size.x,
                rot.y_axis * half_size.y,
                rot.z_axis * half_size.z,
            ),
        }
    }

    /// Create an OBB from a 12-element array.
    ///
    /// Layout: `[cx, cy, cz, ax, ay, az, bx, by, bz, cx, cy, cz]` where
    /// `(cx, cy, cz)` is the center and each subsequent triple is a half-axis
    /// column vector.
    ///
    /// Returns `None` if the slice has fewer than 12 elements.
    pub fn from_array(a: &[f64]) -> Option<Self> {
        if a.len() < 12 {
            return None;
        }
        Some(Self {
            center: DVec3::new(a[0], a[1], a[2]),
            half_axes: DMat3::from_cols(
                DVec3::new(a[3], a[4], a[5]),
                DVec3::new(a[6], a[7], a[8]),
                DVec3::new(a[9], a[10], a[11]),
            ),
        })
    }

    /// Serialize this OBB to a 12-element array matching the [`from_array`] layout.
    ///
    /// [`from_array`]: OrientedBoundingBox::from_array
    pub fn to_array(&self) -> [f64; 12] {
        let h = &self.half_axes;
        [
            self.center.x,
            self.center.y,
            self.center.z,
            h.x_axis.x,
            h.x_axis.y,
            h.x_axis.z,
            h.y_axis.x,
            h.y_axis.y,
            h.y_axis.z,
            h.z_axis.x,
            h.z_axis.y,
            h.z_axis.z,
        ]
    }

    /// Create an axis-aligned OBB enclosing the given points.
    pub fn from_corners(corners: &[DVec3]) -> Self {
        assert!(!corners.is_empty());
        let mut min = corners[0];
        let mut max = corners[0];
        for &c in &corners[1..] {
            min = min.min(c);
            max = max.max(c);
        }
        let center = (min + max) * 0.5;
        Self {
            center,
            half_axes: DMat3::from_diagonal((max - min) * 0.5),
        }
    }

    /// Test whether a point is inside the OBB.
    #[inline]
    pub fn contains(&self, point: DVec3) -> bool {
        let d = point - self.center;
        for col in [
            self.half_axes.x_axis,
            self.half_axes.y_axis,
            self.half_axes.z_axis,
        ] {
            let len_sq = col.length_squared();
            if len_sq < f64::EPSILON * f64::EPSILON {
                continue;
            }
            if d.dot(col).abs() > len_sq {
                return false;
            }
        }
        true
    }

    /// Squared distance from the OBB surface to a point.
    /// Returns `0.0` when inside.
    ///
    /// Prefer this over [`Self::distance_to`] when the caller only needs a
    /// comparison (e.g., SSE ordering), since it avoids the final `sqrt`.
    pub fn distance_squared_to(&self, point: DVec3) -> f64 {
        let d = point - self.center;
        let mut dist_sq = 0.0_f64;
        for col in [
            self.half_axes.x_axis,
            self.half_axes.y_axis,
            self.half_axes.z_axis,
        ] {
            let len_sq = col.length_squared();
            if len_sq < f64::EPSILON {
                continue;
            }
            // `proj` is the projection of `d` onto `col` expressed in units
            // where `col` has unit length; comparing it to `len` (i.e. `1` in
            // the normalised frame, but `len` here because we divide the dot
            // product once by `len`) avoids two sqrts - one for `col` and
            // one inside the abs/excess comparison.
            let len = len_sq.sqrt();
            let proj = d.dot(col) / len;
            let excess = (proj.abs() - len).max(0.0);
            dist_sq += excess * excess;
        }
        dist_sq
    }

    /// Distance from the OBB surface to a point. Returns `0.0` when inside.
    #[inline]
    pub fn distance_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point).sqrt()
    }

    /// Test the OBB against a plane using the separating-axis theorem.
    #[inline]
    pub fn intersect_plane(&self, plane: &Plane) -> CullingResult {
        let h = &self.half_axes;
        let r = h.x_axis.dot(plane.normal).abs()
            + h.y_axis.dot(plane.normal).abs()
            + h.z_axis.dot(plane.normal).abs();
        let dist = plane.signed_distance(self.center);
        if dist > r {
            CullingResult::Inside
        } else if dist < -r {
            CullingResult::Outside
        } else {
            CullingResult::Intersecting
        }
    }

    /// Compute the smallest enclosing bounding sphere.
    ///
    /// The radius equals the half-diagonal: `sqrt(|a|^2 + |b|^2 + |c|^2)`.
    pub fn to_sphere(&self) -> BoundingSphere {
        let r_sq = self.half_axes.x_axis.length_squared()
            + self.half_axes.y_axis.length_squared()
            + self.half_axes.z_axis.length_squared();
        BoundingSphere::new(self.center, r_sq.sqrt())
    }

    /// Compute the axis-aligned bounding box that encloses this OBB.
    pub fn to_aabb(&self) -> AxisAlignedBoundingBox {
        let h = &self.half_axes;
        let extent = DVec3::new(
            h.x_axis.x.abs() + h.y_axis.x.abs() + h.z_axis.x.abs(),
            h.x_axis.y.abs() + h.y_axis.y.abs() + h.z_axis.y.abs(),
            h.x_axis.z.abs() + h.y_axis.z.abs() + h.z_axis.z.abs(),
        );
        AxisAlignedBoundingBox::new(self.center - extent, self.center + extent)
    }

    /// Compute the 8 corner vertices in world space.
    pub fn corners(&self) -> [DVec3; 8] {
        let (a, b, c) = (
            self.half_axes.x_axis,
            self.half_axes.y_axis,
            self.half_axes.z_axis,
        );
        [
            self.center - a - b - c,
            self.center + a - b - c,
            self.center - a + b - c,
            self.center + a + b - c,
            self.center - a - b + c,
            self.center + a - b + c,
            self.center - a + b + c,
            self.center + a + b + c,
        ]
    }

    /// Apply a 4x4 affine transform to this OBB.
    pub fn transform(&self, m: &DMat4) -> Self {
        Self {
            center: m.transform_point3(self.center),
            half_axes: DMat3::from_cols(
                m.transform_vector3(self.half_axes.x_axis),
                m.transform_vector3(self.half_axes.y_axis),
                m.transform_vector3(self.half_axes.z_axis),
            ),
        }
    }

    /// Approximate screen-space projected area in pixels.
    ///
    /// Uses the enclosing sphere for a disc-projection estimate.
    pub fn projected_area(&self, camera_position: DVec3, viewport_height: f64, fov_y: f64) -> f64 {
        let dist = self.center.distance(camera_position);
        if dist < 1e-10 {
            return f64::MAX;
        }
        let radius = self.to_sphere().radius;
        let d = (radius * viewport_height) / (dist * (fov_y * 0.5).tan());
        std::f64::consts::PI * 0.25 * d * d
    }
}

impl From<AxisAlignedBoundingBox> for OrientedBoundingBox {
    fn from(aabb: AxisAlignedBoundingBox) -> Self {
        Self {
            center: aabb.center(),
            half_axes: DMat3::from_diagonal(aabb.half_extents()),
        }
    }
}

impl From<BoundingSphere> for OrientedBoundingBox {
    fn from(sphere: BoundingSphere) -> Self {
        Self {
            center: sphere.center,
            half_axes: DMat3::from_diagonal(DVec3::splat(sphere.radius)),
        }
    }
}

impl From<OrientedBoundingBox> for BoundingSphere {
    fn from(obb: OrientedBoundingBox) -> Self {
        obb.to_sphere()
    }
}

impl From<AxisAlignedBoundingBox> for BoundingSphere {
    fn from(aabb: AxisAlignedBoundingBox) -> Self {
        aabb.to_bounding_sphere()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axis_aligned(center: DVec3, half_size: DVec3) -> OrientedBoundingBox {
        OrientedBoundingBox::new(center, DMat3::from_diagonal(half_size))
    }

    #[test]
    fn from_quat_identity() {
        let obb = OrientedBoundingBox::from_quat(
            DVec3::new(1.0, 2.0, 3.0),
            DQuat::IDENTITY,
            DVec3::new(4.0, 5.0, 6.0),
        );
        assert!((obb.center - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
        let lengths = DVec3::new(
            obb.half_axes.x_axis.length(),
            obb.half_axes.y_axis.length(),
            obb.half_axes.z_axis.length(),
        );
        assert!((lengths - DVec3::new(4.0, 5.0, 6.0)).length() < 1e-12);
    }

    #[test]
    fn from_array_roundtrip() {
        let a = [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0];
        let obb = OrientedBoundingBox::from_array(&a).unwrap();
        let b = obb.to_array();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }

    #[test]
    fn contains_center() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::new(1.0, 2.0, 3.0));
        assert!(obb.contains(DVec3::ZERO));
    }

    #[test]
    fn contains_inside() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::new(5.0, 5.0, 5.0));
        assert!(obb.contains(DVec3::new(3.0, 3.0, 3.0)));
    }

    #[test]
    fn not_contains_outside() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::new(1.0, 1.0, 1.0));
        assert!(!obb.contains(DVec3::new(2.0, 0.0, 0.0)));
    }

    #[test]
    fn contains_rotated() {
        let obb = OrientedBoundingBox::from_quat(
            DVec3::ZERO,
            DQuat::from_rotation_z(std::f64::consts::FRAC_PI_4),
            DVec3::new(10.0, 1.0, 1.0),
        );
        assert!(obb.contains(DVec3::new(5.0, 5.0, 0.0)));
        assert!(!obb.contains(DVec3::new(10.0, 0.0, 0.0)));
    }

    #[test]
    fn distance_to_inside_is_zero() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::new(5.0, 5.0, 5.0));
        assert!(obb.distance_to(DVec3::new(1.0, 1.0, 1.0)).abs() < 1e-10);
    }

    #[test]
    fn distance_to_outside() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::ONE);
        assert!((obb.distance_to(DVec3::new(3.0, 0.0, 0.0)) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn intersect_plane_inside() {
        let obb = axis_aligned(DVec3::new(0.0, 10.0, 0.0), DVec3::ONE);
        let p = Plane::from_point_normal(DVec3::ZERO, DVec3::Y);
        assert_eq!(obb.intersect_plane(&p), CullingResult::Inside);
    }

    #[test]
    fn intersect_plane_outside() {
        let obb = axis_aligned(DVec3::new(0.0, -10.0, 0.0), DVec3::ONE);
        let p = Plane::from_point_normal(DVec3::ZERO, DVec3::Y);
        assert_eq!(obb.intersect_plane(&p), CullingResult::Outside);
    }

    #[test]
    fn intersect_plane_intersecting() {
        let obb = axis_aligned(DVec3::new(0.0, 0.5, 0.0), DVec3::ONE);
        let p = Plane::from_point_normal(DVec3::ZERO, DVec3::Y);
        assert_eq!(obb.intersect_plane(&p), CullingResult::Intersecting);
    }

    #[test]
    fn to_aabb_identity() {
        let obb = axis_aligned(DVec3::new(5.0, 5.0, 5.0), DVec3::new(1.0, 2.0, 3.0));
        let aabb = obb.to_aabb();
        assert!((aabb.min - DVec3::new(4.0, 3.0, 2.0)).length() < 1e-12);
        assert!((aabb.max - DVec3::new(6.0, 7.0, 8.0)).length() < 1e-12);
    }

    #[test]
    fn to_sphere_half_diagonal() {
        // half-diagonal of (3, 4, 0) = sqrt(9+16+0) = 5
        let obb = axis_aligned(DVec3::new(1.0, 2.0, 3.0), DVec3::new(3.0, 4.0, 0.0));
        let s = obb.to_sphere();
        assert!((s.center - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
        assert!((s.radius - 5.0).abs() < 1e-12);
    }

    #[test]
    fn projected_area_decreases_with_distance() {
        let obb = axis_aligned(DVec3::ZERO, DVec3::new(10.0, 10.0, 10.0));
        let near = obb.projected_area(DVec3::new(0.0, 0.0, 100.0), 1080.0, 1.0);
        let far = obb.projected_area(DVec3::new(0.0, 0.0, 1000.0), 1080.0, 1.0);
        assert!(far < near);
    }

    #[test]
    fn from_aabb_conversion() {
        let aabb = AxisAlignedBoundingBox::new(DVec3::NEG_ONE, DVec3::ONE);
        let obb = OrientedBoundingBox::from(aabb);
        assert!(obb.contains(DVec3::ZERO));
        assert!(!obb.contains(DVec3::splat(2.0)));
    }
}

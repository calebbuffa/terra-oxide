//! `BoundingVolume` trait - a common interface for all bounding volume types.

use glam::DVec3;

use crate::culling::CullingResult;
use crate::plane::Plane;
use crate::ray::Ray;
use crate::sphere::BoundingSphere;

/// Common geometric queries shared by all bounding volume types.
///
/// Implemented by [`BoundingSphere`], [`AxisAlignedBoundingBox`],
/// [`OrientedBoundingBox`], and [`SpatialBounds`].
///
/// [`AxisAlignedBoundingBox`]: crate::AxisAlignedBoundingBox
/// [`OrientedBoundingBox`]: crate::OrientedBoundingBox
/// [`SpatialBounds`]: crate::SpatialBounds
pub trait BoundingVolume {
    /// Center point of the bounding volume.
    fn center(&self) -> DVec3;

    /// Returns `true` if `point` is inside or on the boundary.
    fn contains(&self, point: DVec3) -> bool;

    /// Squared distance to the nearest surface point. Returns `0.0` if inside.
    ///
    /// Implement this rather than [`distance_to`] to avoid an unnecessary
    /// `sqrt` when comparing distances.
    ///
    /// [`distance_to`]: BoundingVolume::distance_to
    fn distance_squared_to(&self, point: DVec3) -> f64;

    /// Non-negative distance to the nearest surface. Returns `0.0` if inside.
    ///
    /// Default impl is `self.distance_squared_to(point).sqrt()`.
    fn distance_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point).sqrt()
    }

    /// Classify this volume against a plane using the separating-axis theorem.
    ///
    /// Returns [`CullingResult::Inside`] when fully on the positive side,
    /// [`CullingResult::Outside`] when fully on the negative side, or
    /// [`CullingResult::Intersecting`] when straddling the plane.
    fn classify_plane(&self, plane: &Plane) -> CullingResult;

    /// Test a ray against the volume. Returns the parametric distance `t >= 0`
    /// to the first intersection, or `None` on a miss.
    fn intersect_ray(&self, ray: &Ray) -> Option<f64>;

    /// Compute the smallest enclosing bounding sphere.
    fn to_sphere(&self) -> BoundingSphere;
}

impl BoundingVolume for BoundingSphere {
    fn center(&self) -> DVec3 {
        self.center
    }

    fn contains(&self, point: DVec3) -> bool {
        self.center.distance_squared(point) <= self.radius * self.radius
    }

    fn distance_squared_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point)
    }

    fn classify_plane(&self, plane: &Plane) -> CullingResult {
        self.intersect_plane(plane)
    }

    fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        crate::intersection::ray_sphere(ray, self)
    }

    fn to_sphere(&self) -> BoundingSphere {
        *self
    }
}

impl BoundingVolume for crate::AxisAlignedBoundingBox {
    fn center(&self) -> DVec3 {
        self.center()
    }

    fn contains(&self, point: DVec3) -> bool {
        self.contains(point)
    }

    fn distance_squared_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point)
    }

    fn classify_plane(&self, plane: &Plane) -> CullingResult {
        self.intersect_plane(plane)
    }

    fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        crate::intersection::ray_aabb(ray, self)
    }

    fn to_sphere(&self) -> BoundingSphere {
        self.to_bounding_sphere()
    }
}

impl BoundingVolume for crate::OrientedBoundingBox {
    fn center(&self) -> DVec3 {
        self.center
    }

    fn contains(&self, point: DVec3) -> bool {
        self.contains(point)
    }

    fn distance_squared_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point)
    }

    fn classify_plane(&self, plane: &Plane) -> CullingResult {
        self.intersect_plane(plane)
    }

    fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        crate::intersection::ray_obb(ray, self)
    }

    fn to_sphere(&self) -> BoundingSphere {
        self.to_sphere()
    }
}

impl BoundingVolume for crate::SpatialBounds {
    fn center(&self) -> DVec3 {
        self.center()
    }

    fn contains(&self, point: DVec3) -> bool {
        self.contains(point)
    }

    fn distance_squared_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point)
    }

    fn classify_plane(&self, plane: &Plane) -> CullingResult {
        self.classify_plane(plane)
    }

    fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        self.intersect_ray(ray)
    }

    fn to_sphere(&self) -> BoundingSphere {
        self.to_sphere()
    }
}

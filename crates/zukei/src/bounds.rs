use glam::{DMat3, DVec2, DVec3};

use crate::aabb::AxisAlignedBoundingBox;
use crate::obb::OrientedBoundingBox;
use crate::polygon::{point_in_polygon_2d, polygon_boundary_distance_2d};
use crate::ray::Ray;
use crate::rectangle::Rectangle;
use crate::sphere::BoundingSphere;
use crate::tiling::{OctreeTileID, QuadtreeTileID};

/// Denominator for a given implicit tile level.
///
/// Divide the root tile's geometric error by this value to get the
/// geometric error for tiles on `level`. Divide each axis of a bounding
/// volume by this factor to get the child tile size at that level.
///
/// Equivalent to `2^level`.
#[inline]
pub fn compute_level_denominator(level: u32) -> f64 {
    (1u64 << level) as f64
}

/// The spatial extent of a tile or node.
///
/// Each variant wraps the corresponding concrete bounding-volume type.
/// Use the concrete type directly when you know the shape; use `SpatialBounds`
/// when the shape is determined at runtime.
#[derive(Clone, Debug, PartialEq)]
pub enum SpatialBounds {
    Sphere(BoundingSphere),
    Aabb(AxisAlignedBoundingBox),
    Obb(OrientedBoundingBox),
    Rectangle(Rectangle),
    Polygon(Vec<DVec2>),
    Empty,
}

impl SpatialBounds {
    /// Center point of the bounding volume, or `DVec3::ZERO` for `Empty`.
    pub fn center(&self) -> DVec3 {
        match self {
            Self::Sphere(s) => s.center,
            Self::Aabb(a) => a.center(),
            Self::Obb(o) => o.center,
            Self::Rectangle(r) => r.center().extend(0.0),
            Self::Polygon(verts) => {
                if verts.is_empty() {
                    return DVec3::ZERO;
                }
                let sum = verts.iter().fold(DVec2::ZERO, |a, v| a + *v);
                (sum / verts.len() as f64).extend(0.0)
            }
            Self::Empty => DVec3::ZERO,
        }
    }

    pub fn as_sphere(&self) -> Option<&BoundingSphere> {
        if let Self::Sphere(s) = self {
            Some(s)
        } else {
            None
        }
    }

    pub fn as_obb(&self) -> Option<&OrientedBoundingBox> {
        if let Self::Obb(o) = self {
            Some(o)
        } else {
            None
        }
    }

    pub fn as_aabb(&self) -> Option<&AxisAlignedBoundingBox> {
        if let Self::Aabb(a) = self {
            Some(a)
        } else {
            None
        }
    }
}

impl SpatialBounds {
    /// Non-negative distance from `point` to the nearest surface.
    /// Returns `0.0` when the point is inside.
    #[inline]
    pub fn distance_to(&self, point: DVec3) -> f64 {
        self.distance_squared_to(point).sqrt()
    }

    /// Non-negative squared distance from `point` to the nearest surface.
    /// Returns `0.0` when inside.
    ///
    /// Prefer this over [`Self::distance_to`] whenever the caller only needs
    /// a comparison (e.g., SSE ordering, priority scoring); it avoids the
    /// final `sqrt` on every variant and returns `INFINITY` for `Empty`.
    pub fn distance_squared_to(&self, point: DVec3) -> f64 {
        match self {
            Self::Sphere(s) => s.distance_squared_to(point),
            Self::Aabb(a) => a.distance_squared_to(point),
            Self::Obb(o) => o.distance_squared_to(point),
            Self::Rectangle(r) => {
                let ex = (r.minimum_x - point.x).max(point.x - r.maximum_x).max(0.0);
                let ey = (r.minimum_y - point.y).max(point.y - r.maximum_y).max(0.0);
                ex * ex + ey * ey
            }
            Self::Polygon(verts) => {
                let p2 = DVec2::new(point.x, point.y);
                if point_in_polygon_2d(p2, verts) {
                    0.0
                } else {
                    let d = polygon_boundary_distance_2d(p2, verts);
                    d * d
                }
            }
            Self::Empty => f64::INFINITY,
        }
    }

    /// Returns `true` if `point` is inside (or on the boundary of) this volume.
    pub fn contains(&self, point: DVec3) -> bool {
        match self {
            Self::Sphere(s) => s.contains(point),
            Self::Aabb(a) => a.contains(point),
            Self::Obb(o) => o.contains(point),
            Self::Rectangle(r) => {
                point.x >= r.minimum_x
                    && point.x <= r.maximum_x
                    && point.y >= r.minimum_y
                    && point.y <= r.maximum_y
            }
            Self::Polygon(verts) => point_in_polygon_2d(DVec2::new(point.x, point.y), verts),
            Self::Empty => false,
        }
    }

    /// Returns `true` when the volume is entirely on the negative side of `plane`.
    pub fn is_clipped_by(&self, plane: &crate::plane::Plane) -> bool {
        support_dot(self, plane.normal) + plane.distance < 0.0
    }

    /// Classify this volume against a plane using the separating-axis theorem.
    pub fn classify_plane(&self, plane: &crate::plane::Plane) -> crate::culling::CullingResult {
        use crate::culling::CullingResult;
        match self {
            Self::Sphere(s) => s.intersect_plane(plane),
            Self::Aabb(a) => a.intersect_plane(plane),
            Self::Obb(o) => o.intersect_plane(plane),
            _ => {
                if self.is_clipped_by(plane) {
                    CullingResult::Outside
                } else {
                    CullingResult::Intersecting
                }
            }
        }
    }

    /// Returns `true` when the horizontal (XZ) projection of `point` falls
    /// within this volume's footprint.
    pub fn is_over_footprint(&self, point: DVec3) -> bool {
        match self {
            Self::Sphere(s) => {
                let dx = point.x - s.center.x;
                let dz = point.z - s.center.z;
                (dx * dx + dz * dz).sqrt() <= s.radius
            }
            Self::Aabb(a) => {
                point.x >= a.min.x && point.x <= a.max.x && point.z >= a.min.z && point.z <= a.max.z
            }
            Self::Obb(o) => {
                let d = point - o.center;
                for col in [o.half_axes.x_axis, o.half_axes.z_axis] {
                    let len = col.length();
                    if len < f64::EPSILON {
                        continue;
                    }
                    if d.dot(col / len).abs() > len {
                        return false;
                    }
                }
                true
            }
            Self::Rectangle(r) => {
                point.x >= r.minimum_x
                    && point.x <= r.maximum_x
                    && point.y >= r.minimum_y
                    && point.y <= r.maximum_y
            }
            Self::Polygon(verts) => point_in_polygon_2d(DVec2::new(point.x, point.z), verts),
            Self::Empty => false,
        }
    }

    /// Test a ray against this bounding volume, returning the parametric
    /// distance `t >= 0` to the first intersection, or `None` on a miss.
    pub fn intersect_ray(&self, ray: &Ray) -> Option<f64> {
        use crate::intersection::{ray_aabb, ray_obb, ray_sphere};
        match self {
            Self::Sphere(s) => ray_sphere(ray, s),
            Self::Aabb(a) => ray_aabb(ray, a),
            Self::Obb(o) => ray_obb(ray, o),
            Self::Rectangle(r) => {
                let min3 = DVec3::new(r.minimum_x, r.minimum_y, -f64::EPSILON);
                let max3 = DVec3::new(r.maximum_x, r.maximum_y, f64::EPSILON);
                let aabb = AxisAlignedBoundingBox::new(min3, max3);
                ray_aabb(ray, &aabb)
            }
            Self::Polygon(verts) => ray_vs_polygon_2d(ray, verts),
            Self::Empty => None,
        }
    }

    /// Convert to the smallest enclosing bounding sphere.
    pub fn to_sphere(&self) -> BoundingSphere {
        match self {
            Self::Sphere(s) => *s,
            Self::Aabb(a) => a.to_bounding_sphere(),
            Self::Obb(o) => o.to_sphere(),
            Self::Rectangle(r) => {
                let cx = (r.minimum_x + r.maximum_x) * 0.5;
                let cy = (r.minimum_y + r.maximum_y) * 0.5;
                let center = DVec2::new(cx, cy);
                let radius = DVec2::new(r.maximum_x, r.maximum_y).distance(center);
                BoundingSphere::new(center.extend(0.0), radius)
            }
            Self::Polygon(verts) => {
                let min = verts
                    .iter()
                    .fold(DVec2::splat(f64::INFINITY), |a, v| a.min(*v));
                let max = verts
                    .iter()
                    .fold(DVec2::splat(f64::NEG_INFINITY), |a, v| a.max(*v));
                let center = (min + max) * 0.5;
                let radius = verts.iter().map(|v| v.distance(center)).fold(0.0, f64::max);
                BoundingSphere::new(center.extend(0.0), radius)
            }
            Self::Empty => BoundingSphere::new(DVec3::ZERO, 0.0),
        }
    }

    /// Subdivide a [`SpatialBounds`] for an octree tile.
    ///
    /// Port of `ImplicitTilingUtilities::computeBoundingVolume(OBB, OctreeTileID)`.
    pub fn subdivide_oct(&self, tile: OctreeTileID) -> SpatialBounds {
        let denom = compute_level_denominator(tile.level);

        match self {
            SpatialBounds::Obb(o) => {
                let x_dim = o.half_axes.col(0) * 2.0 / denom;
                let y_dim = o.half_axes.col(1) * 2.0 / denom;
                let z_dim = o.half_axes.col(2) * 2.0 / denom;
                let min_corner =
                    o.center - o.half_axes.col(0) - o.half_axes.col(1) - o.half_axes.col(2);
                let child_min = min_corner
                    + x_dim * tile.x as f64
                    + y_dim * tile.y as f64
                    + z_dim * tile.z as f64;
                let child_max = min_corner
                    + x_dim * (tile.x as f64 + 1.0)
                    + y_dim * (tile.y as f64 + 1.0)
                    + z_dim * (tile.z as f64 + 1.0);
                SpatialBounds::Obb(OrientedBoundingBox::new(
                    (child_min + child_max) * 0.5,
                    DMat3::from_cols(x_dim * 0.5, y_dim * 0.5, z_dim * 0.5),
                ))
            }
            SpatialBounds::Sphere(s) => {
                let obb = SpatialBounds::Obb(OrientedBoundingBox::new(
                    s.center,
                    DMat3::from_diagonal(DVec3::splat(s.radius)),
                ));
                obb.subdivide_oct(tile)
            }
            SpatialBounds::Aabb(a) => {
                let sizes = a.max - a.min;
                let x_step = sizes.x / denom;
                let y_step = sizes.y / denom;
                let z_step = sizes.z / denom;
                let child_min = a.min
                    + DVec3::new(
                        x_step * tile.x as f64,
                        y_step * tile.y as f64,
                        z_step * tile.z as f64,
                    );
                let child_max = child_min + DVec3::new(x_step, y_step, z_step);
                SpatialBounds::Aabb(AxisAlignedBoundingBox::new(child_min, child_max))
            }
            _ => SpatialBounds::Empty,
        }
    }

    /// Subdivide a [`SpatialBounds`] for a quadtree tile.
    /// Returns `SpatialBounds::Empty` for unsupported input variants.
    pub fn subdivide_quad(&self, tile: QuadtreeTileID) -> SpatialBounds {
        let denom = compute_level_denominator(tile.level);

        match self {
            SpatialBounds::Obb(o) => {
                let x_dim = o.half_axes.col(0) * 2.0 / denom;
                let y_dim = o.half_axes.col(1) * 2.0 / denom;
                let z_half = o.half_axes.col(2);
                let min_corner = o.center - o.half_axes.col(0) - o.half_axes.col(1) - z_half;
                let child_min = min_corner + x_dim * tile.x as f64 + y_dim * tile.y as f64;
                let child_max = min_corner
                    + x_dim * (tile.x as f64 + 1.0)
                    + y_dim * (tile.y as f64 + 1.0)
                    + z_half * 2.0;
                SpatialBounds::Obb(OrientedBoundingBox::new(
                    (child_min + child_max) * 0.5,
                    DMat3::from_cols(x_dim * 0.5, y_dim * 0.5, z_half),
                ))
            }
            SpatialBounds::Sphere(s) => {
                let obb = SpatialBounds::Obb(OrientedBoundingBox::new(
                    s.center,
                    DMat3::from_diagonal(DVec3::splat(s.radius)),
                ));
                obb.subdivide_quad(tile)
            }
            SpatialBounds::Aabb(a) => {
                let sizes = a.max - a.min;
                let x_step = sizes.x / denom;
                let y_step = sizes.y / denom;
                let child_min =
                    a.min + DVec3::new(x_step * tile.x as f64, y_step * tile.y as f64, 0.0);
                let child_max = DVec3::new(child_min.x + x_step, child_min.y + y_step, a.max.z);
                SpatialBounds::Aabb(AxisAlignedBoundingBox::new(child_min, child_max))
            }
            _ => SpatialBounds::Empty,
        }
    }
}

impl From<BoundingSphere> for SpatialBounds {
    fn from(s: BoundingSphere) -> Self {
        Self::Sphere(s)
    }
}
impl From<AxisAlignedBoundingBox> for SpatialBounds {
    fn from(a: AxisAlignedBoundingBox) -> Self {
        Self::Aabb(a)
    }
}
impl From<OrientedBoundingBox> for SpatialBounds {
    fn from(o: OrientedBoundingBox) -> Self {
        Self::Obb(o)
    }
}
impl From<Rectangle> for SpatialBounds {
    fn from(r: Rectangle) -> Self {
        Self::Rectangle(r)
    }
}

impl TryFrom<SpatialBounds> for OrientedBoundingBox {
    type Error = SpatialBounds;
    fn try_from(b: SpatialBounds) -> Result<Self, Self::Error> {
        if let SpatialBounds::Obb(o) = b {
            Ok(o)
        } else {
            Err(b)
        }
    }
}
impl TryFrom<SpatialBounds> for BoundingSphere {
    type Error = SpatialBounds;
    fn try_from(b: SpatialBounds) -> Result<Self, Self::Error> {
        if let SpatialBounds::Sphere(s) = b {
            Ok(s)
        } else {
            Err(b)
        }
    }
}
impl TryFrom<SpatialBounds> for AxisAlignedBoundingBox {
    type Error = SpatialBounds;
    fn try_from(b: SpatialBounds) -> Result<Self, Self::Error> {
        if let SpatialBounds::Aabb(a) = b {
            Ok(a)
        } else {
            Err(b)
        }
    }
}

/// Support function: `max_{p in bounds} (normal . p)`.
fn support_dot(bounds: &SpatialBounds, normal: DVec3) -> f64 {
    match bounds {
        SpatialBounds::Sphere(s) => normal.dot(s.center) + s.radius,
        SpatialBounds::Aabb(a) => {
            let cx = if normal.x >= 0.0 { a.max.x } else { a.min.x };
            let cy = if normal.y >= 0.0 { a.max.y } else { a.min.y };
            let cz = if normal.z >= 0.0 { a.max.z } else { a.min.z };
            normal.dot(DVec3::new(cx, cy, cz))
        }
        SpatialBounds::Obb(o) => {
            normal.dot(o.center)
                + normal.dot(o.half_axes.x_axis).abs()
                + normal.dot(o.half_axes.y_axis).abs()
                + normal.dot(o.half_axes.z_axis).abs()
        }
        SpatialBounds::Rectangle(r) => {
            let cx = if normal.x >= 0.0 {
                r.maximum_x
            } else {
                r.minimum_x
            };
            let cy = if normal.y >= 0.0 {
                r.maximum_y
            } else {
                r.minimum_y
            };
            normal.x * cx + normal.y * cy
        }
        SpatialBounds::Polygon(verts) => verts
            .iter()
            .map(|v| normal.x * v.x + normal.y * v.y)
            .fold(f64::NEG_INFINITY, f64::max),
        SpatialBounds::Empty => f64::NEG_INFINITY,
    }
}

fn ray_vs_polygon_2d(ray: &Ray, verts: &[DVec2]) -> Option<f64> {
    let n = verts.len();
    if n < 3 {
        return None;
    }
    for i in 0..n {
        let a = verts[i].extend(0.0);
        let b = verts[(i + 1) % n].extend(0.0);
        let c = if i + 2 < n {
            verts[i + 2]
        } else {
            verts[(i + 2) % n]
        }
        .extend(0.0);
        if let Some(t) = crate::intersection::ray_triangle(ray, a, b, c) {
            if t >= 0.0 {
                return Some(t);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_denominator() {
        assert_eq!(compute_level_denominator(0), 1.0);
        assert_eq!(compute_level_denominator(1), 2.0);
        assert_eq!(compute_level_denominator(4), 16.0);
    }
}

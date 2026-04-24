mod aabb;
mod bounds;
mod culling;
mod frustum;
mod intersection;
pub mod morton;
mod obb;
mod plane;
mod polygon;
mod ray;
mod rectangle;
pub mod s2;
mod sphere;
mod tiling;
mod transforms;
pub mod volume;

pub use aabb::AxisAlignedBoundingBox;
pub use bounds::SpatialBounds;
pub use culling::CullingResult;
pub use frustum::CullingVolume;
pub use intersection::{
    point_in_triangle_2d, point_in_triangle_3d, point_in_triangle_3d_barycentric, ray_aabb,
    ray_aabb_parametric, ray_ellipsoid, ray_obb, ray_plane, ray_sphere, ray_triangle,
    ray_triangle_parametric,
};
pub use obb::OrientedBoundingBox;
pub use plane::Plane;
pub use polygon::{
    cross2, ear_clip, point_in_polygon_2d, point_to_segment_dist_2d, polygon_boundary_distance_2d,
};
pub use ray::Ray;
pub use rectangle::Rectangle;
pub use s2::S2CellId;
pub use sphere::BoundingSphere;
pub use tiling::{
    OctreeTileID, OctreeTilingScheme, QuadtreeTileID, QuadtreeTileRectangularRange,
    QuadtreeTilingScheme,
};
pub use transforms::{
    Axis, X_UP_TO_Y_UP, X_UP_TO_Z_UP, Y_UP_TO_X_UP, Y_UP_TO_Z_UP, Z_UP_TO_X_UP, Z_UP_TO_Y_UP,
    apply_up_axis_correction, create_orthographic, create_perspective_fov,
    create_perspective_offcenter, create_translation_rotation_scale, create_view_matrix,
    decompose_translation_rotation_scale, get_up_axis_transform, mat3_to_quat,
    rotation_from_up_right,
};
pub use volume::BoundingVolume;

pub const ONE_PI: f64 = std::f64::consts::PI;
pub const TWO_PI: f64 = ONE_PI * 2.0;
pub const PI_OVER_TWO: f64 = ONE_PI / 2.0;
pub const PI_OVER_FOUR: f64 = ONE_PI / 4.0;

pub const RADIANS_PER_DEGREE: f64 = ONE_PI / 180.0;
pub const DEGREES_PER_RADIAN: f64 = 180.0 / ONE_PI;

/// Converts degrees to radians.
#[inline]
pub fn to_radians(degrees: f64) -> f64 {
    degrees * RADIANS_PER_DEGREE
}

/// Converts radians to degrees.
#[inline]
pub fn to_degrees(radians: f64) -> f64 {
    radians * DEGREES_PER_RADIAN
}

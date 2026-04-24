//! Bounding volume and geometric error helpers for terrain quadtree tiles.

use std::f64::consts::PI;

use glam::{DMat3, DVec3};
use zukei::{
    AxisAlignedBoundingBox, BoundingSphere, QuadtreeTileID, QuadtreeTilingScheme, SpatialBounds,
};

use crate::{Cartographic, Ellipsoid, calc_quadtree_max_geometric_error};

const MIN_H: f64 = -500.0;
const MAX_H: f64 = 9000.0;

/// Compute an oriented bounding box for a geodetic rectangle at conservative
/// terrain height extents (−500 m … +9 000 m).
pub fn obb_for_region(
    ellipsoid: &Ellipsoid,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
) -> SpatialBounds {
    let mut aabb_min = DVec3::splat(f64::MAX);
    let mut aabb_max = DVec3::splat(f64::MIN);

    for &lon in &[west, east] {
        for &lat in &[south, north] {
            for &h in &[MIN_H, MAX_H] {
                let p = ellipsoid.cartographic_to_ecef(Cartographic {
                    longitude: lon,
                    latitude: lat,
                    height: h,
                });
                aabb_min = aabb_min.min(p);
                aabb_max = aabb_max.max(p);
            }
        }
    }

    let center = (aabb_min + aabb_max) * 0.5;
    let half = (aabb_max - aabb_min) * 0.5;
    let half_axes = DMat3::from_diagonal(half);
    SpatialBounds::Obb(zukei::OrientedBoundingBox::new(center, half_axes))
}

/// Geometric error for a quadtree tile, scaled by its angular width.
///
/// Mirrors the error formula used by cesium-native's terrain loader.
pub fn tile_geometric_error(
    ellipsoid: &Ellipsoid,
    id: QuadtreeTileID,
    scheme: &QuadtreeTilingScheme,
) -> f64 {
    let max_err = calc_quadtree_max_geometric_error(ellipsoid);
    let nx = scheme.tiles_x_at_level(id.level) as f64;
    let angular_width = 2.0 * PI / nx;
    8.0 * max_err * angular_width
}

/// Compute an axis-aligned bounding box in ECEF for a geodetic rectangle at
/// the ellipsoid surface (height = 0).
///
/// Samples the four geographic corners and returns the AABB that encloses them.
pub fn aabb_for_region(
    ellipsoid: &Ellipsoid,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
) -> SpatialBounds {
    let corners = [(west, south), (east, south), (west, north), (east, north)];
    let mut min = DVec3::splat(f64::MAX);
    let mut max = DVec3::splat(f64::MIN);
    for (lon, lat) in corners {
        let p = ellipsoid.cartographic_to_ecef(Cartographic::new(lon, lat, 0.0));
        min = min.min(p);
        max = max.max(p);
    }
    SpatialBounds::Aabb(AxisAlignedBoundingBox::new(min, max))
}

/// Approximate a geodetic region as a bounding sphere using its eight ECEF
/// corner points (sampling both `min_height` and `max_height`).
///
/// The sphere centre is the ECEF point at the geographic midpoint and mid
/// height; the radius is the maximum distance from centre to any corner.
pub fn sphere_for_region(
    ellipsoid: &Ellipsoid,
    west: f64,
    south: f64,
    east: f64,
    north: f64,
    min_height: f64,
    max_height: f64,
) -> SpatialBounds {
    let mid = ellipsoid.cartographic_to_ecef(Cartographic::new(
        (west + east) * 0.5,
        (south + north) * 0.5,
        (min_height + max_height) * 0.5,
    ));
    let corners = [
        (west, south, min_height),
        (east, south, min_height),
        (west, north, min_height),
        (east, north, min_height),
        (west, south, max_height),
        (east, south, max_height),
        (west, north, max_height),
        (east, north, max_height),
    ];
    let radius = corners
        .iter()
        .map(|&(lon, lat, h)| {
            (ellipsoid.cartographic_to_ecef(Cartographic::new(lon, lat, h)) - mid).length()
        })
        .fold(0.0_f64, f64::max);
    SpatialBounds::Sphere(BoundingSphere::new(mid, radius))
}

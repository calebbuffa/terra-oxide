//! Local reference frame transforms relative to the ellipsoid surface.
//!
//! These functions produce 4x4 homogeneous matrices (column-major, matching
//! glam's `DMat4` convention) that transform from a local frame at a given
//! ECEF origin into ECEF world space.

use glam::{DMat4, DQuat, DVec3, DVec4};
use outil::EPSILON6;

use crate::Ellipsoid;

/// Build the East-North-Up (ENU) frame at `origin`, expressed as a 4x4
/// column-major matrix that transforms from ENU-local to ECEF world space.
///
/// The local axes are:
/// - **+X** = East
/// - **+Y** = North
/// - **+Z** = Up (geodetic surface normal at `origin`)
///
/// Equivalent to Cesium's `Transforms.eastNorthUpToFixedFrame`.
///
/// # Panics
/// Does not panic, but returns a degenerate matrix if `origin` is the
/// ellipsoid centre (surface normal undefined).
pub fn east_north_up_to_ecef(origin: DVec3, ellipsoid: &Ellipsoid) -> DMat4 {
    let up = ellipsoid.geodetic_surface_normal(origin);
    let (east, north, up) = enu_basis(up);
    DMat4::from_cols(
        DVec4::from((east, 0.0)),
        DVec4::from((north, 0.0)),
        DVec4::from((up, 0.0)),
        DVec4::from((origin, 1.0)),
    )
}

/// Build an orthonormal ENU basis whose +Z is `up`.
///
/// Returns `(east, north, up)` in ECEF, all unit length. Near the poles the
/// east axis is derived from the ECEF X-axis instead of Z to avoid a
/// degenerate cross product. If `up` itself is too short to use as a
/// reference (numerically zero), the basis falls back to world-aligned axes.
#[inline]
fn enu_basis(up: DVec3) -> (DVec3, DVec3, DVec3) {
    const POLE_THRESHOLD: f64 = EPSILON6;
    let up = if up.length_squared() < POLE_THRESHOLD * POLE_THRESHOLD {
        DVec3::Z
    } else {
        up.normalize()
    };
    let reference = if up.x.abs() < POLE_THRESHOLD && up.y.abs() < POLE_THRESHOLD {
        // Near the geographic poles - use X-axis as reference.
        DVec3::X
    } else {
        DVec3::Z
    };
    let east = reference.cross(up).normalize();
    let north = up.cross(east);
    (east, north, up)
}

/// Compute the ENU-aligned rotation quaternion for an instance at ECEF
/// position `p` using double precision.
///
/// Double precision is essential for instances placed far from the tileset
/// origin (e.g. 10,000 km offsets in I3DM batches) - f32 math accumulates
/// ~10 cm of error there which is visible as rotational jitter. Callers
/// holding `f32` data should convert at the boundary with
/// [`Vec3::as_dvec3`](glam::Vec3::as_dvec3) and
/// [`DQuat::as_quat`](glam::DQuat::as_quat).
///
/// Falls back to [`DQuat::IDENTITY`] near the poles or at the origin.
pub fn enu_quaternion(p: DVec3) -> DQuat {
    if p.length_squared() < EPSILON6 * EPSILON6 {
        return DQuat::IDENTITY;
    }
    let (east, north, up) = enu_basis(p);
    DQuat::from_mat3(&glam::DMat3::from_cols(east, north, up))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cartographic;
    use glam::DVec3;

    #[test]
    fn enu_at_equator_prime_meridian() {
        let ellipsoid = Ellipsoid::wgs84();
        let origin = ellipsoid.cartographic_to_ecef(Cartographic::from_degrees(0.0, 0.0, 0.0));
        let m = east_north_up_to_ecef(origin, &ellipsoid);

        // East axis at (lon=0, lat=0) should be +Y in ECEF.
        let east = m.col(0).truncate();
        assert!((east - DVec3::Y).length() < 1e-10, "east = {east:?}");

        // Up axis should be ~+X in ECEF (towards origin on equator at lon=0).
        let up = m.col(2).truncate();
        let expected_up = origin.normalize();
        assert!((up - expected_up).length() < 1e-10, "up = {up:?}");
    }

    #[test]
    fn enu_at_north_pole_does_not_panic() {
        let ellipsoid = Ellipsoid::wgs84();
        let origin = ellipsoid.cartographic_to_ecef(Cartographic::from_degrees(0.0, 90.0, 0.0));
        let _m = east_north_up_to_ecef(origin, &ellipsoid);
    }
}

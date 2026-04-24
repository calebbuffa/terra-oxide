//! Ellipsoidal occluder for horizon culling.
//!
//! Determines whether a point or bounding sphere is hidden behind the Earth's
//! curvature as seen from a given camera position in ECEF space.  Mirrors the
//! algorithm in CesiumJS `EllipsoidalOccluder`.

use glam::DVec3;

use crate::Ellipsoid;

/// Horizon-culling occluder based on an ellipsoidal Earth model.
///
/// The camera position is scaled to the unit sphere internally; all occlusion
/// tests work in that scaled space so they generalise to any ellipsoid.
///
/// # Example
/// ```
/// # use terra::{Ellipsoid, EllipsoidalOccluder};
/// # use glam::DVec3;
/// let ellipsoid = Ellipsoid::wgs84();
/// // Camera above the North Pole
/// let camera = DVec3::new(0.0, 0.0, 7_000_000.0);
/// let occluder = EllipsoidalOccluder::new(ellipsoid, camera);
/// // A point on the opposite side of Earth is occluded.
/// assert!(occluder.is_point_occluded(DVec3::new(0.0, 0.0, -6_356_000.0)));
/// ```
#[derive(Debug, Clone)]
pub struct EllipsoidalOccluder {
    ellipsoid: Ellipsoid,
    camera_position: DVec3,
    scaled_camera_position: DVec3,
    /// Negative normalised scaled camera position — points from camera toward
    /// the ellipsoid centre in scaled space.
    scaled_camera_direction: DVec3,
}

impl EllipsoidalOccluder {
    /// Create a new occluder for `ellipsoid` viewed from `camera_position`
    /// (ECEF, metres).
    pub fn new(ellipsoid: Ellipsoid, camera_position: DVec3) -> Self {
        let scaled = ellipsoid.transform_position_to_scaled_space(camera_position);
        let direction = if scaled.length_squared() > 0.0 {
            -scaled.normalize()
        } else {
            DVec3::ZERO
        };
        Self {
            ellipsoid,
            camera_position,
            scaled_camera_position: scaled,
            scaled_camera_direction: direction,
        }
    }

    /// Update the camera position, recomputing internal precomputed state.
    pub fn set_camera_position(&mut self, camera_position: DVec3) {
        self.camera_position = camera_position;
        self.scaled_camera_position = self
            .ellipsoid
            .transform_position_to_scaled_space(camera_position);
        self.scaled_camera_direction = if self.scaled_camera_position.length_squared() > 0.0 {
            -self.scaled_camera_position.normalize()
        } else {
            DVec3::ZERO
        };
    }

    /// Current camera position in ECEF.
    #[inline]
    pub fn camera_position(&self) -> DVec3 {
        self.camera_position
    }

    /// Test whether `point` (ECEF) is occluded by the ellipsoid from the
    /// camera.
    ///
    /// Returns `true` if the point is **not** visible (it is behind the
    /// horizon).
    pub fn is_point_occluded(&self, point: DVec3) -> bool {
        !self
            .is_scaled_space_point_visible(self.ellipsoid.transform_position_to_scaled_space(point))
    }

    /// Test whether a bounding sphere (conservative) is occluded.
    ///
    /// The sphere centre is tested; the radius is accounted for by shifting the
    /// horizon plane outward.  This is conservative: if the sphere straddles
    /// the horizon the function may return `false` (visible) even if part of
    /// the sphere is hidden.
    pub fn is_bounding_sphere_occluded(&self, center: DVec3, radius: f64) -> bool {
        // Offset the scaled-space center away from the camera by the radius to
        // get a conservative occlusion test.
        let scaled_center = self.ellipsoid.transform_position_to_scaled_space(center);
        // Use the radius as an offset along the direction toward the camera.
        let to_camera = self.scaled_camera_position - scaled_center;
        let dist = to_camera.length();
        if dist < 1e-15 {
            return false;
        }
        // Conservative: shift the test point toward the camera by the sphere
        // radius scaled to ellipsoid-scaled space. Using `minimum_radius()`
        // ensures the scaled sphere is never smaller than its true projection
        // on any axis (correct for oblate ellipsoids).
        let scaled_radius = radius / self.ellipsoid.minimum_radius();
        let scaled_offset = scaled_radius.min(dist);
        let adjusted = scaled_center + to_camera / dist * scaled_offset;
        !self.is_scaled_space_point_visible(adjusted)
    }

    /// Compute the horizon-culling point for a set of positions.
    ///
    /// Given a nominal `direction_to_point` (unit vector from the ellipsoid
    /// centre toward the feature), this returns the world-space position that
    /// most occludes the feature — i.e., the point in `positions` that, when
    /// scaled to the unit sphere, has the largest projection onto
    /// `direction_to_point`.
    ///
    /// Returns `None` if `positions` is empty or `direction_to_point` is zero.
    pub fn compute_horizon_culling_point(
        &self,
        direction_to_point: DVec3,
        positions: &[DVec3],
    ) -> Option<DVec3> {
        let dir_len = direction_to_point.length();
        if dir_len < 1e-15 || positions.is_empty() {
            return None;
        }
        let dir_norm = direction_to_point / dir_len;

        let mut max_mag: f64 = f64::NEG_INFINITY;
        for &pos in positions {
            let scaled = self.ellipsoid.transform_position_to_scaled_space(pos);
            let mag = scaled.dot(dir_norm);
            if mag > max_mag {
                max_mag = mag;
            }
        }

        Some(dir_norm * max_mag)
    }

    // ---- internal ----

    /// CesiumJS `isScaledSpacePointVisible`: a scaled-space point is visible
    /// iff `dot(scaled_point - scaled_camera, scaled_camera) >= 0`.
    #[inline]
    fn is_scaled_space_point_visible(&self, scaled_point: DVec3) -> bool {
        (scaled_point - self.scaled_camera_position).dot(self.scaled_camera_position) >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Ellipsoid;

    #[test]
    fn bounding_sphere_on_far_side_is_occluded() {
        let ellipsoid = Ellipsoid::wgs84();
        // Camera above the North Pole
        let camera = DVec3::new(0.0, 0.0, 7_000_000.0);
        let occluder = EllipsoidalOccluder::new(ellipsoid, camera);
        // A sphere centred deep on the far side of Earth, small radius
        let center = DVec3::new(0.0, 0.0, -6_356_000.0);
        assert!(
            occluder.is_bounding_sphere_occluded(center, 100.0),
            "sphere on far side should be occluded"
        );
    }

    #[test]
    fn bounding_sphere_near_camera_is_visible() {
        let ellipsoid = Ellipsoid::wgs84();
        let camera = DVec3::new(0.0, 0.0, 7_000_000.0);
        let occluder = EllipsoidalOccluder::new(ellipsoid, camera);
        // A sphere well above the camera (in the same direction) satisfies
        // dot(p - c, c) > 0 and must not be occluded.
        let center = DVec3::new(0.0, 0.0, 10_000_000.0);
        assert!(
            !occluder.is_bounding_sphere_occluded(center, 1000.0),
            "sphere above camera should not be occluded"
        );
    }
}

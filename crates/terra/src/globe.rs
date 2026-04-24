use std::f64::consts::PI;

use glam::{DMat3, DMat4};
use outil;

use crate::{Cartographic, Ellipsoid, LocalHorizontalCoordinateSystem};

/// An object anchored to the globe at a specific ECEF transform.
///
/// # Example
/// ```
/// # use terra::{Ellipsoid, GlobeAnchor};
/// # use glam::DMat4;
/// let anchor = GlobeAnchor::from_anchor_to_fixed(DMat4::IDENTITY);
/// assert_eq!(anchor.anchor_to_fixed(), DMat4::IDENTITY);
/// ```
#[derive(Debug, Clone)]
pub struct GlobeAnchor {
    anchor_to_fixed: DMat4,
}

impl GlobeAnchor {
    /// Create a `GlobeAnchor` from an explicit `anchor -> ECEF` matrix.
    pub fn from_anchor_to_fixed(anchor_to_fixed: DMat4) -> Self {
        Self { anchor_to_fixed }
    }

    /// Create a `GlobeAnchor` from an `anchor -> local` matrix and the
    /// `LocalHorizontalCoordinateSystem` that defines the local space.
    pub fn from_anchor_to_local(
        local: &LocalHorizontalCoordinateSystem,
        anchor_to_local: DMat4,
    ) -> Self {
        let anchor_to_fixed = local.local_to_ecef_matrix() * anchor_to_local;
        Self { anchor_to_fixed }
    }

    /// The current `anchor -> ECEF` transform.
    #[inline]
    pub fn anchor_to_fixed(&self) -> DMat4 {
        self.anchor_to_fixed
    }

    /// Update the `anchor -> ECEF` transform.
    ///
    /// When `adjust_orientation` is `true`, the rotational part of the matrix
    /// is adjusted so the object remains upright at its new location on the
    /// globe (i.e., the local "up" direction is kept aligned with the geodetic
    /// surface normal).  Pass `false` if you are already accounting for globe
    /// curvature in the caller.
    pub fn set_anchor_to_fixed(
        &mut self,
        new_anchor_to_fixed: DMat4,
        adjust_orientation: bool,
        ellipsoid: &Ellipsoid,
    ) {
        if adjust_orientation {
            self.anchor_to_fixed = adjust_orientation_for_curvature(
                &self.anchor_to_fixed,
                new_anchor_to_fixed,
                ellipsoid,
            );
        } else {
            self.anchor_to_fixed = new_anchor_to_fixed;
        }
    }

    /// Compute the `anchor -> local` matrix for the given coordinate system.
    pub fn anchor_to_local(&self, local: &LocalHorizontalCoordinateSystem) -> DMat4 {
        local.ecef_to_local_matrix() * self.anchor_to_fixed
    }

    /// Update the `anchor -> ECEF` transform by supplying a new
    /// `anchor -> local` matrix.
    pub fn set_anchor_to_local(
        &mut self,
        local: &LocalHorizontalCoordinateSystem,
        new_anchor_to_local: DMat4,
        adjust_orientation: bool,
        ellipsoid: &Ellipsoid,
    ) {
        let new_anchor_to_fixed = local.local_to_ecef_matrix() * new_anchor_to_local;
        self.set_anchor_to_fixed(new_anchor_to_fixed, adjust_orientation, ellipsoid);
    }
}

/// Rotate the orientation component of `new_transform` so that the local "up"
/// vector (the old ECEF surface normal at the old position) maps to the new
/// surface normal at the new position.
///
/// Algorithm:
/// 1. Extract the old surface normal from `old_transform`'s translation.
/// 2. Extract the new surface normal from `new_transform`'s translation.
/// 3. Compute the rotation `R` that takes old -> new normal (axis-angle via
///    cross product).
/// 4. Premultiply the rotation-only part of `new_transform` by `R`.
fn adjust_orientation_for_curvature(
    old_transform: &DMat4,
    mut new_transform: DMat4,
    ellipsoid: &Ellipsoid,
) -> DMat4 {
    let old_position = old_transform.col(3).truncate();
    let new_position = new_transform.col(3).truncate();

    let old_normal = ellipsoid.geodetic_surface_normal(old_position);
    let new_normal = ellipsoid.geodetic_surface_normal(new_position);

    // Rotation that takes old_normal -> new_normal.
    let rot = rotation_from_normals(old_normal, new_normal);

    // Apply the rotation to the upper-left 3x3 of new_transform.
    let upper3x3 = DMat3::from_cols(
        new_transform.col(0).truncate(),
        new_transform.col(1).truncate(),
        new_transform.col(2).truncate(),
    );
    let rotated = rot * upper3x3;

    // Bulk copy each rotated column into the upper-left 3x3 of `new_transform`,
    // preserving column 3's existing translation. `col_mut` returns a mutable
    // `DVec4` reference; `extend(w)` promotes `DVec3 -> DVec4`.
    for i in 0..3 {
        let w = new_transform.col(i).w;
        *new_transform.col_mut(i) = rotated.col(i).extend(w);
    }

    new_transform
}

/// Compute the rotation matrix that takes unit vector `from` to unit vector `to`.
///
/// Uses Rodrigues' rotation formula.  If `from` and `to` are parallel,
/// returns `DMat3::IDENTITY`.
fn rotation_from_normals(from: glam::DVec3, to: glam::DVec3) -> DMat3 {
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if (dot - 1.0).abs() < 1e-10 {
        return DMat3::IDENTITY;
    }
    if (dot + 1.0).abs() < 1e-10 {
        // Anti-parallel: 180 degree rotation around any perpendicular axis.
        let perp = from.any_orthogonal_vector().normalize();
        return rotation_axis_angle(perp, std::f64::consts::PI);
    }
    let axis = from.cross(to).normalize();
    let angle = dot.acos();
    rotation_axis_angle(axis, angle)
}

/// Rodrigues-formula rotation matrix around `axis` by `angle` radians.
fn rotation_axis_angle(axis: glam::DVec3, angle: f64) -> DMat3 {
    let (sin, cos) = angle.sin_cos();
    let t = 1.0 - cos;
    let (x, y, z) = (axis.x, axis.y, axis.z);
    DMat3::from_cols(
        glam::DVec3::new(t * x * x + cos, t * x * y + sin * z, t * x * z - sin * y),
        glam::DVec3::new(t * x * y - sin * z, t * y * y + cos, t * y * z + sin * x),
        glam::DVec3::new(t * x * z + sin * y, t * y * z - sin * x, t * z * z + cos),
    )
}

/// An axis-aligned geodetic bounding rectangle defined by geodetic extent
/// `[west, south, east, north]` in **radians**.
///
/// Used to represent the horizontal extent of a tile or dataset.
/// Heights are handled separately by [`crate::BoundingRegion`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GlobeRectangle {
    /// West boundary longitude (radians).
    pub west: f64,
    /// South boundary latitude (radians).
    pub south: f64,
    /// East boundary longitude (radians).
    pub east: f64,
    /// North boundary latitude (radians).
    pub north: f64,
}
impl GlobeRectangle {
    pub const WEB_MERCATOR: Self = Self::new(-PI, -PI / 2.0, PI, PI / 2.0);

    /// The full surface of the globe.
    pub const MAX: Self = Self::new(-PI, -PI / 2.0, PI, PI / 2.0);

    /// An empty/degenerate rectangle at the origin.
    pub const EMPTY: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    /// Construct from boundary values already in radians.
    #[inline]
    pub const fn new(west: f64, south: f64, east: f64, north: f64) -> Self {
        Self {
            west,
            south,
            east,
            north,
        }
    }

    /// Construct from boundary values in degrees.
    pub fn from_degrees(west_deg: f64, south_deg: f64, east_deg: f64, north_deg: f64) -> Self {
        Self {
            west: west_deg.to_radians(),
            south: south_deg.to_radians(),
            east: east_deg.to_radians(),
            north: north_deg.to_radians(),
        }
    }

    /// Parse from an array of at least 4 radians values.
    ///
    /// Layout: `[west_rad, south_rad, east_rad, north_rad, ...]` - trailing
    /// elements (e.g. height) are ignored.
    /// Returns `None` if the slice has fewer than 4 elements.
    pub fn from_array(region: &[f64]) -> Option<Self> {
        if region.len() < 4 {
            return None;
        }
        Some(Self {
            west: region[0],
            south: region[1],
            east: region[2],
            north: region[3],
        })
    }

    /// Return the `[west, south, east, north]` values in radians.
    #[inline]
    pub fn to_array(self) -> [f64; 4] {
        [self.west, self.south, self.east, self.north]
    }

    /// Return true if the rectangle contains the given cartographic position.
    ///
    /// Handles antimeridian-crossing rectangles (where `east < west`).
    pub fn contains_cartographic(&self, c: Cartographic) -> bool {
        let lon_ok = if self.east >= self.west {
            c.longitude >= self.west && c.longitude <= self.east
        } else {
            // Crosses the antimeridian.
            c.longitude >= self.west || c.longitude <= self.east
        };
        lon_ok && c.latitude >= self.south && c.latitude <= self.north
    }

    /// Return the intersection of this rectangle with `other`, or `None` if
    /// they do not overlap.
    ///
    /// Handles antimeridian-crossing rectangles (where `east < west`) correctly.
    ///
    /// Mirrors `CesiumGeospatial::GlobeRectangle::computeIntersection`.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let mut rect_east = self.east;
        let mut rect_west = self.west;
        let mut other_east = other.east;
        let mut other_west = other.west;

        // Unwrap antimeridian-crossing rectangles into the extended range
        // so that longitude arithmetic is monotonic.
        if rect_east < rect_west && other_east > 0.0 {
            rect_east += 2.0 * PI;
        } else if other_east < other_west && rect_east > 0.0 {
            other_east += 2.0 * PI;
        }

        if rect_east < rect_west && other_west < 0.0 {
            other_west += 2.0 * PI;
        } else if other_east < other_west && rect_west < 0.0 {
            rect_west += 2.0 * PI;
        }

        let west = outil::negative_pi_to_pi(rect_west.max(other_west));
        let east = outil::negative_pi_to_pi(rect_east.min(other_east));

        // If neither input crosses the IDL and the result is degenerate, no intersection.
        if (self.west < self.east || other.west < other.east) && east <= west {
            return None;
        }

        let south = self.south.max(other.south);
        let north = self.north.min(other.north);
        if south >= north {
            return None;
        }

        Some(Self::new(west, south, east, north))
    }

    /// Geodetic centre of this rectangle as a [`Cartographic`] at height 0.
    pub fn center(&self) -> Cartographic {
        let lon = if self.east >= self.west {
            (self.west + self.east) * 0.5
        } else {
            // Antimeridian crossing: average wraps around.
            let mid = (self.west + self.east + 2.0 * PI) * 0.5;
            if mid > PI { mid - 2.0 * PI } else { mid }
        };
        Cartographic::new(lon, (self.south + self.north) * 0.5, 0.0)
    }

    /// East-west angular width in radians. Handles antimeridian crossing.
    #[inline]
    pub fn width(&self) -> f64 {
        if self.east >= self.west {
            self.east - self.west
        } else {
            self.east - self.west + 2.0 * PI
        }
    }

    /// North-south angular height in radians.
    #[inline]
    pub fn height(&self) -> f64 {
        self.north - self.south
    }

    /// Return true if the rectangle covers the full globe.
    #[inline]
    pub fn is_full_globe(&self) -> bool {
        self.west <= -PI && self.east >= PI && self.south <= -PI / 2.0 && self.north >= PI / 2.0
    }

    /// Return true if this is an empty / degenerate rectangle.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.width() <= 0.0 || self.height() <= 0.0
    }

    /// Split this rectangle at the antimeridian (±\pi longitude).
    ///
    /// If the rectangle does not cross the antimeridian (`west <= east`), returns
    /// `(self, None)`. Otherwise returns two non-crossing rectangles; the larger
    /// piece is always first (same convention as cesium-native).
    ///
    /// Mirrors `CesiumGeospatial::GlobeRectangle::splitAtAntiMeridian`.
    pub fn split_at_anti_meridian(&self) -> (Self, Option<Self>) {
        if self.west <= self.east {
            return (*self, None);
        }
        // Eastern piece: [west, \pi]
        let a = Self::new(self.west, self.south, PI, self.north);
        // Western piece: [-\pi, east]
        let b = Self::new(-PI, self.south, self.east, self.north);
        if a.width() >= b.width() {
            (a, Some(b))
        } else {
            (b, Some(a))
        }
    }

    /// Map a [`Cartographic`] position to normalised (0 -> 1) coordinates
    /// within this rectangle.
    ///
    /// Returns `(u, v)` where `u = 0` at the west edge and `v = 0` at the
    /// south edge. Antimeridian-crossing rectangles are handled correctly.
    ///
    /// Mirrors `CesiumGeospatial::GlobeRectangle::computeNormalizedCoordinates`.
    pub fn compute_normalized_coordinates(&self, c: Cartographic) -> [f64; 2] {
        let mut east = self.east;
        let mut lon = c.longitude;
        if east < self.west {
            east += 2.0 * PI;
            if lon < self.west {
                lon += 2.0 * PI;
            }
        }
        let u = (lon - self.west) / (east - self.west);
        let v = (c.latitude - self.south) / (self.north - self.south);
        [u, v]
    }

    /// Returns the south-west corner as a [`Cartographic`] at height 0.
    #[inline]
    pub fn southwest(&self) -> Cartographic {
        Cartographic::new(self.west, self.south, 0.0)
    }

    /// Returns the south-east corner as a [`Cartographic`] at height 0.
    #[inline]
    pub fn southeast(&self) -> Cartographic {
        Cartographic::new(self.east, self.south, 0.0)
    }

    /// Returns the north-west corner as a [`Cartographic`] at height 0.
    #[inline]
    pub fn northwest(&self) -> Cartographic {
        Cartographic::new(self.west, self.north, 0.0)
    }

    /// Returns the north-east corner as a [`Cartographic`] at height 0.
    #[inline]
    pub fn northeast(&self) -> Cartographic {
        Cartographic::new(self.east, self.north, 0.0)
    }
}

impl Default for GlobeRectangle {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl TryFrom<&[f64]> for GlobeRectangle {
    type Error = ();
    fn try_from(a: &[f64]) -> Result<Self, Self::Error> {
        Self::from_array(a).ok_or(())
    }
}

impl Into<[f64; 4]> for GlobeRectangle {
    fn into(self) -> [f64; 4] {
        self.to_array()
    }
}

impl From<&[f64; 4]> for GlobeRectangle {
    fn from(a: &[f64; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cartographic, LocalDirection};
    use glam::{DMat3, DMat4, DVec3, DVec4};

    #[test]
    fn split_at_anti_meridian_non_crossing() {
        let r = GlobeRectangle::from_degrees(10.0, -10.0, 20.0, 10.0);
        let (a, b) = r.split_at_anti_meridian();
        assert_eq!(a, r);
        assert!(b.is_none());
    }

    #[test]
    fn split_at_anti_meridian_crossing() {
        let deg = PI / 180.0;
        let r = GlobeRectangle::new(170.0 * deg, -10.0 * deg, -175.0 * deg, 10.0 * deg);
        let (primary, secondary) = r.split_at_anti_meridian();
        let sec = secondary.unwrap();
        assert!(
            (primary.width() - 10.0 * deg).abs() < 1e-12,
            "primary_w={}",
            primary.width() / deg
        );
        assert!(
            (sec.width() - 5.0 * deg).abs() < 1e-12,
            "sec_w={}",
            sec.width() / deg
        );
        assert!(primary.west <= primary.east);
        assert!(sec.west <= sec.east);
    }

    #[test]
    fn compute_normalized_coordinates_center() {
        let r = GlobeRectangle::from_degrees(0.0, 0.0, 10.0, 10.0);
        let c = Cartographic::from_degrees(5.0, 5.0, 0.0);
        let [u, v] = r.compute_normalized_coordinates(c);
        assert!((u - 0.5).abs() < 1e-12);
        assert!((v - 0.5).abs() < 1e-12);
    }

    #[test]
    fn compute_normalized_coordinates_antimeridian() {
        let deg = PI / 180.0;
        let r = GlobeRectangle::new(170.0 * deg, -10.0 * deg, -175.0 * deg, 10.0 * deg);
        let c = Cartographic::new(-179.0 * deg, 0.0, 0.0);
        let [u, _v] = r.compute_normalized_coordinates(c);
        assert!((u - 11.0 / 15.0).abs() < 1e-12, "u={u}");
    }

    #[test]
    fn globe_rectangle_corners() {
        let r = GlobeRectangle::from_degrees(-10.0, -20.0, 10.0, 20.0);
        let sw = r.southwest();
        assert!((sw.longitude - r.west).abs() < 1e-12);
        assert!((sw.latitude - r.south).abs() < 1e-12);
        let ne = r.northeast();
        assert!((ne.longitude - r.east).abs() < 1e-12);
        assert!((ne.latitude - r.north).abs() < 1e-12);
    }

    // --- GlobeRectangle::intersection tests ---

    #[test]
    fn intersection_normal_overlap() {
        let a = GlobeRectangle::from_degrees(-10.0, -10.0, 10.0, 10.0);
        let b = GlobeRectangle::from_degrees(0.0, -5.0, 20.0, 5.0);
        let i = a.intersection(&b).unwrap();
        assert!((i.west.to_degrees() - 0.0).abs() < 1e-10);
        assert!((i.east.to_degrees() - 10.0).abs() < 1e-10);
        assert!((i.south.to_degrees() - (-5.0)).abs() < 1e-10);
        assert!((i.north.to_degrees() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn intersection_no_overlap() {
        let a = GlobeRectangle::from_degrees(-10.0, -10.0, 0.0, 10.0);
        let b = GlobeRectangle::from_degrees(5.0, -10.0, 15.0, 10.0);
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn intersection_self_crosses_antimeridian() {
        // Self: 170° to -175° (crosses IDL, 15° wide). Other: 168° to 175°.
        // Intersection should be 170° to 175°.
        let a = GlobeRectangle::from_degrees(170.0, -10.0, -175.0, 10.0);
        let b = GlobeRectangle::from_degrees(168.0, -5.0, 175.0, 5.0);
        let i = a.intersection(&b).unwrap();
        assert!(
            (i.west.to_degrees() - 170.0).abs() < 1e-9,
            "west={}",
            i.west.to_degrees()
        );
        assert!(
            (i.east.to_degrees() - 175.0).abs() < 1e-9,
            "east={}",
            i.east.to_degrees()
        );
        assert!(i.west <= i.east, "result must not cross IDL");
    }

    #[test]
    fn intersection_other_crosses_antimeridian() {
        // Other: 170° to -175° (crosses IDL). Self: 168° to 175°.
        let a = GlobeRectangle::from_degrees(168.0, -5.0, 175.0, 5.0);
        let b = GlobeRectangle::from_degrees(170.0, -10.0, -175.0, 10.0);
        let i = a.intersection(&b).unwrap();
        assert!(
            (i.west.to_degrees() - 170.0).abs() < 1e-9,
            "west={}",
            i.west.to_degrees()
        );
        assert!(
            (i.east.to_degrees() - 175.0).abs() < 1e-9,
            "east={}",
            i.east.to_degrees()
        );
    }

    #[test]
    fn intersection_both_cross_antimeridian() {
        // a: 160° to -170° (30° wide), b: 150° to -160° (50° wide).
        // a covers [160°,180°]∪[-180°,-170°], b covers [150°,180°]∪[-180°,-160°].
        // Intersection: [160°,180°]∪[-180°,-170°] = 30° wide, west=160°, east=-170°.
        let a = GlobeRectangle::from_degrees(160.0, -10.0, -170.0, 10.0);
        let b = GlobeRectangle::from_degrees(150.0, -5.0, -160.0, 5.0);
        let i = a.intersection(&b).unwrap();
        assert!(
            (i.west.to_degrees() - 160.0).abs() < 1e-9
                && (i.east.to_degrees() - (-170.0)).abs() < 1e-9,
            "west={} east={}",
            i.west.to_degrees(),
            i.east.to_degrees()
        );
        // The result itself crosses the IDL (east < west).
        assert!(
            i.east < i.west,
            "intersection of two IDL-crossing rects must also cross IDL"
        );
    }

    #[test]
    fn intersection_crossing_and_non_crossing_no_overlap() {
        // Self crosses IDL: 170° to -170°. Other is entirely in the western hemisphere.
        let a = GlobeRectangle::from_degrees(170.0, -10.0, -170.0, 10.0);
        let b = GlobeRectangle::from_degrees(-100.0, -5.0, -80.0, 5.0);
        assert!(a.intersection(&b).is_none());
    }

    fn wgs84() -> Ellipsoid {
        Ellipsoid::wgs84()
    }

    fn lhcs_at(lon_deg: f64, lat_deg: f64) -> LocalHorizontalCoordinateSystem {
        let e = wgs84();
        LocalHorizontalCoordinateSystem::from_cartographic(
            Cartographic::from_degrees(lon_deg, lat_deg, 0.0),
            LocalDirection::East,
            LocalDirection::North,
            LocalDirection::Up,
            1.0,
            &e,
        )
    }

    #[test]
    fn from_anchor_to_fixed_stores_matrix() {
        let m = DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0));
        let anchor = GlobeAnchor::from_anchor_to_fixed(m);
        assert_eq!(anchor.anchor_to_fixed(), m);
    }

    #[test]
    fn from_anchor_to_local_round_trip() {
        let local = lhcs_at(0.0, 0.0);
        let anchor_to_local = DMat4::from_translation(DVec3::new(10.0, 0.0, 0.0));
        let anchor = GlobeAnchor::from_anchor_to_local(&local, anchor_to_local);
        let recovered = anchor.anchor_to_local(&local);
        for col in 0..4 {
            let diff = (recovered.col(col) - anchor_to_local.col(col)).length();
            assert!(diff < 1e-6, "col {} diff={diff}", col);
        }
    }

    #[test]
    fn set_anchor_no_orientation_adjust() {
        let m1 = DMat4::IDENTITY;
        let m2 = DMat4::from_translation(DVec3::new(6_378_137.0, 0.0, 0.0));
        let mut anchor = GlobeAnchor::from_anchor_to_fixed(m1);
        anchor.set_anchor_to_fixed(m2, false, &wgs84());
        assert_eq!(anchor.anchor_to_fixed(), m2);
    }

    #[test]
    fn set_anchor_with_orientation_adjust_changes_rotation() {
        // Move from equator/0 degree to equator/90 degree - the up-direction rotates 90 degree.
        let e = wgs84();
        let c0 = Cartographic::from_degrees(0.0, 0.0, 0.0);
        let c1 = Cartographic::from_degrees(90.0, 0.0, 0.0);
        let p0 = e.cartographic_to_ecef(c0);
        let p1 = e.cartographic_to_ecef(c1);

        let m0 = DMat4::from_cols(
            DVec4::new(1.0, 0.0, 0.0, 0.0),
            DVec4::new(0.0, 1.0, 0.0, 0.0),
            DVec4::new(0.0, 0.0, 1.0, 0.0),
            DVec4::from((p0, 1.0)),
        );
        let m1 = DMat4::from_cols(
            DVec4::new(1.0, 0.0, 0.0, 0.0),
            DVec4::new(0.0, 1.0, 0.0, 0.0),
            DVec4::new(0.0, 0.0, 1.0, 0.0),
            DVec4::from((p1, 1.0)),
        );
        let mut anchor = GlobeAnchor::from_anchor_to_fixed(m0);
        anchor.set_anchor_to_fixed(m1, true, &e);
        // The rotation part should be different from the naive `m1`.
        let naive_col0: DVec3 = m1.col(0).truncate();
        let adjusted_col0: DVec3 = anchor.anchor_to_fixed().col(0).truncate();
        // They should differ because orientation was rotated.
        let same = (adjusted_col0 - naive_col0).length() < 1e-6;
        assert!(!same, "orientation should have been adjusted");
    }

    #[test]
    fn set_anchor_to_local_round_trip() {
        let e = wgs84();
        let local = lhcs_at(10.0, 20.0);
        let anchor_to_local = DMat4::from_translation(DVec3::new(5.0, 5.0, 0.0));
        let mut anchor = GlobeAnchor::from_anchor_to_local(&local, anchor_to_local);
        // Update via set_anchor_to_local without orientation adjustment.
        let new_local_mat = DMat4::from_translation(DVec3::new(20.0, 0.0, 0.0));
        anchor.set_anchor_to_local(&local, new_local_mat, false, &e);
        let recovered = anchor.anchor_to_local(&local);
        for col in 0..4 {
            let diff = (recovered.col(col) - new_local_mat.col(col)).length();
            assert!(diff < 1e-4, "col {} diff={diff}", col);
        }
    }

    #[test]
    fn rotation_from_same_normal_is_identity() {
        let n = DVec3::new(0.0, 0.0, 1.0);
        let r = rotation_from_normals(n, n);
        let identity = DMat3::IDENTITY;
        for c in 0..3 {
            let diff = (r.col(c) - identity.col(c)).length();
            assert!(diff < 1e-10, "col {} not identity", c);
        }
    }

    #[test]
    fn rotation_from_antipodal_normals_is_180_degrees() {
        let r = rotation_from_normals(DVec3::Z, -DVec3::Z);
        // Z should map to -Z.
        let mapped = r * DVec3::Z;
        assert!((mapped - (-DVec3::Z)).length() < 1e-6, "mapped={mapped}");
    }
}

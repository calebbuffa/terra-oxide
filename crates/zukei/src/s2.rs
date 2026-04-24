//! S2 geometry cell identifiers.
//!
//! Implements the core of the S2 cell system used by the
//! `3DTILES_bounding_volume_S2` implicit tiling extension.
//!
//! An [`S2CellId`] is a 64-bit integer that encodes:
//! - bits 63–61: face (0–5)
//! - bits 60–1:  Hilbert-curve position within that face
//! - the level is inferred from the trailing sentinel bit
//!
//! # References
//! - Google S2 Geometry Library (open-source)
//! - CesiumJS `S2Cell.js`
//! - 3D Tiles spec §3DTILES_bounding_volume_S2

/// Lookup table for Hilbert curve decode: (di, dj, new_orientation).
///
/// Row = orientation * 4, column = 2-bit input.
/// Orientations: 0 = normal, 1 = right, 2 = flipped, 3 = left.
const HILBERT_LOOKUP: [(u32, u32, u32); 16] = [
    // orientation 0 (normal)
    (0, 0, 1), // 00 -> SW -> rotate right
    (0, 1, 0), // 01 -> NW -> no rotation
    (1, 1, 0), // 10 -> NE -> no rotation
    (1, 0, 3), // 11 -> SE -> rotate left
    // orientation 1 (right)
    (0, 0, 0), // 00
    (1, 0, 1), // 01
    (1, 1, 1), // 10
    (0, 1, 2), // 11
    // orientation 2 (flipped)
    (1, 1, 3), // 00
    (1, 0, 2), // 01
    (0, 0, 2), // 10
    (0, 1, 1), // 11
    // orientation 3 (left)
    (1, 1, 2), // 00
    (0, 1, 3), // 01
    (0, 0, 3), // 10
    (1, 0, 0), // 11
];

/// A 64-bit S2 cell identifier.
///
/// The root (level 0) cell for face `f` is `S2CellId(f << 61 | 0x1000000000000000)`.
/// The maximum subdivision depth is 30.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct S2CellId(pub u64);

impl S2CellId {
    /// Maximum level (deepest subdivision).
    pub const MAX_LEVEL: u32 = 30;

    /// Parse an S2 token (compact hex with trailing zeros stripped).
    ///
    /// Returns `None` if the token is empty or longer than 16 hex characters.
    pub fn from_token(token: &str) -> Option<Self> {
        if token.is_empty() || token.len() > 16 {
            return None;
        }
        // Parse as hex, then shift left so the significant bits occupy the
        // most-significant positions of a 16-char (64-bit) hex string.
        let shift = (16 - token.len()) * 4;
        let id = u64::from_str_radix(token, 16)
            .ok()?
            .checked_shl(shift as u32)?;
        Some(Self(id))
    }

    /// Render the cell as a compact S2 token (trailing zero nibbles stripped).
    pub fn to_token(self) -> String {
        if self.0 == 0 {
            return "X".to_owned();
        }
        let trailing_nibbles = self.0.trailing_zeros() / 4;
        let hex = format!("{:016x}", self.0);
        hex[..16 - trailing_nibbles as usize].to_owned()
    }

    /// Construct directly from a raw 64-bit S2 cell id.
    #[inline]
    pub const fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Raw 64-bit cell id.
    #[inline]
    pub const fn raw(self) -> u64 {
        self.0
    }

    /// Face number (0–5).
    #[inline]
    pub fn face(self) -> u32 {
        (self.0 >> 61) as u32
    }

    /// Subdivision level (0 = face cell, 30 = finest).
    #[inline]
    pub fn level(self) -> u32 {
        // The level is determined by the position of the trailing 1-bit
        // (the sentinel): level = 30 - (trailing_zeros / 2).
        let tz = self.0.trailing_zeros();
        debug_assert_eq!(tz % 2, 0, "invalid S2CellId: trailing zeros must be even");
        30 - tz / 2
    }

    /// The lowest-set bit (the sentinel / LSB of the Hilbert position at this level).
    #[inline]
    fn lsb(self) -> u64 {
        self.0 & self.0.wrapping_neg()
    }

    /// Parent cell (one level coarser).  Returns `None` for a face cell (level 0).
    pub fn parent(self) -> Option<Self> {
        let level = self.level();
        if level == 0 {
            return None;
        }
        let new_lsb = self.lsb() << 2;
        Some(Self((self.0 & new_lsb.wrapping_neg()) | new_lsb))
    }

    /// Parent at the given `level`.  `level` must be ≤ `self.level()`.
    pub fn parent_at_level(self, level: u32) -> Self {
        debug_assert!(level <= self.level());
        let new_lsb = 1u64 << (2 * (Self::MAX_LEVEL - level) + 1);
        Self((self.0 & new_lsb.wrapping_neg()) | new_lsb)
    }

    /// All four children, ordered by Hilbert position (0–3).
    pub fn children(self) -> [Self; 4] {
        let lsb = self.lsb();
        let child_lsb = lsb >> 2;
        // offset of child i from this cell's id (before clearing the sentinel)
        let base = self.0 & !(lsb - 1) & !(lsb);
        [
            Self(base | child_lsb * 1 | child_lsb),
            Self(base | child_lsb * 3 | child_lsb),
            Self(base | child_lsb * 5 | child_lsb),
            Self(base | child_lsb * 7 | child_lsb),
        ]
    }

    /// Which child position (0–3) this cell occupies within its parent.
    ///
    /// Panics if called on a face cell (level 0).
    pub fn child_position(self) -> u32 {
        let level = self.level();
        debug_assert!(level > 0, "face cells have no child position");
        // The two bits above the current LSB encode the child index.
        ((self.0 >> (2 * (Self::MAX_LEVEL - level) + 1)) & 3) as u32
    }

    /// Hilbert position of this tile within the subtree rooted at `subtree_root`.
    ///
    /// This is used for the availability bit index in implicit S2 tilesets.
    /// The position is the interleaved child-position bits from the subtree
    /// root's level + 1 down to this tile's level.
    pub fn subtree_hilbert_position(self, subtree_root: S2CellId) -> u64 {
        let root_level = subtree_root.level();
        let tile_level = self.level();
        debug_assert!(tile_level >= root_level);
        let depth = tile_level - root_level; // number of levels inside the subtree

        let mut pos = 0u64;
        let mut cur = self;
        for i in 0..depth {
            // We accumulate from the deepest level upward.
            let cp = cur.child_position() as u64;
            pos |= cp << (2 * i);
            if let Some(p) = cur.parent() {
                cur = p;
            }
        }
        pos
    }

    /// Geographic bounds in radians (west, south, east, north).
    ///
    /// Uses the S2 face->UV->ST->lat/lon projection.
    pub fn lat_lon_bounds_radians(self) -> (f64, f64, f64, f64) {
        // Sample the four corners of the cell in ST space and convert to lat/lon.
        let level = self.level();
        let (st_min, st_max) = self.st_bounds(level);
        let face = self.face();

        // Walk all four corners (and midpoints for faces that wrap) to find the
        // true geographic extent.
        let corners = [
            (st_min.0, st_min.1),
            (st_max.0, st_min.1),
            (st_min.0, st_max.1),
            (st_max.0, st_max.1),
        ];

        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;

        for (s, t) in corners {
            let u = st_to_uv(s);
            let v = st_to_uv(t);
            let xyz = face_uv_to_xyz(face, u, v);
            let (lat, lon) = xyz_to_lat_lon(xyz);
            min_lat = min_lat.min(lat);
            max_lat = max_lat.max(lat);
            min_lon = min_lon.min(lon);
            max_lon = max_lon.max(lon);
        }

        (min_lon, min_lat, max_lon, max_lat)
    }

    /// ST bounds for this cell at its level.  Returns ((s_min, t_min), (s_max, t_max)).
    fn st_bounds(&self, level: u32) -> ((f64, f64), (f64, f64)) {
        // At level L there are 2^L cells along each ST axis per face.
        // Compute the IJ coordinates from the Hilbert curve position.
        let (i, j) = self.to_ij();
        let scale = 1.0 / (1u64 << level) as f64;
        let s_min = i as f64 * scale;
        let t_min = j as f64 * scale;
        (
            (s_min.clamp(0.0, 1.0), t_min.clamp(0.0, 1.0)),
            (
                (s_min + scale).clamp(0.0, 1.0),
                (t_min + scale).clamp(0.0, 1.0),
            ),
        )
    }

    /// Convert the cell centre to integer IJ grid coordinates at its level.
    fn to_ij(self) -> (u32, u32) {
        let level = self.level();
        let n = 1u64 << level;
        // Extract the Hilbert curve position at this level.
        // The S2 Hilbert position is stored in bits 2..2*(level+1).
        let bits = 2 * level;
        let pos = (self.0 >> (2 * (Self::MAX_LEVEL - level) + 1)) & ((1u64 << bits) - 1);
        hilbert_pos_to_ij(level, pos, n)
    }
}

impl std::fmt::Display for S2CellId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_token())
    }
}

/// Quadratic ST -> UV mapping (same as used in the reference S2 library).
#[inline]
fn st_to_uv(s: f64) -> f64 {
    if s >= 0.5 {
        (1.0 / 3.0) * (4.0 * s * s - 1.0)
    } else {
        (1.0 / 3.0) * (1.0 - 4.0 * (1.0 - s) * (1.0 - s))
    }
}

/// Map (face, u, v) to a unit ECEF vector.
///
/// S2 face numbering: 0=+x, 1=+y, 2=+z, 3=-x, 4=-y, 5=-z
#[inline]
fn face_uv_to_xyz(face: u32, u: f64, v: f64) -> (f64, f64, f64) {
    match face {
        0 => (1.0, u, v),
        1 => (-u, 1.0, v),
        2 => (-u, -v, 1.0),
        3 => (-1.0, -v, -u),
        4 => (v, -1.0, -u),
        5 => (v, u, -1.0),
        _ => unreachable!("S2 face must be 0–5"),
    }
}

/// Convert an ECEF unit vector to (latitude, longitude) in radians.
#[inline]
fn xyz_to_lat_lon(xyz: (f64, f64, f64)) -> (f64, f64) {
    let (x, y, z) = xyz;
    let lat = z.atan2((x * x + y * y).sqrt());
    let lon = y.atan2(x);
    (lat, lon)
}

/// Decode a Hilbert curve position at `level` into (i, j) grid coordinates.
///
/// This implements the classic in-place bit-reversal decode for the S2/Peano
/// Hilbert curve.  Based on the reference C++ `S2CellId::ToIJK`.
fn hilbert_pos_to_ij(level: u32, pos: u64, _n: u64) -> (u32, u32) {
    // Use the Hilbert curve decode table.
    // The S2 curve is a sequence of transformations on a 2-bit pair (the two
    // bits of the Hilbert position at each level), each of which selects a
    // quadrant and rotates/reflects the sub-curve.
    let mut i = 0u32;
    let mut j = 0u32;
    let mut orientation = 0u32; // initial orientation

    for k in (0..level).rev() {
        let bits = ((pos >> (2 * k)) & 3) as u32;
        let (di, dj, new_orientation) = HILBERT_LOOKUP[orientation as usize * 4 + bits as usize];
        i = (i << 1) | di;
        j = (j << 1) | dj;
        orientation = new_orientation;
    }
    (i, j)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::PI;

    #[test]
    fn token_roundtrip() {
        let tokens = ["89c25c", "1", "3754", "4b59a"];
        for token in &tokens {
            let cell = S2CellId::from_token(token).expect("valid token");
            assert_eq!(&cell.to_token(), token, "roundtrip failed for {token}");
        }
    }

    #[test]
    fn level_from_token() {
        // Token "1" is face 0, level 0 (the whole face).
        let face_cell = S2CellId::from_token("1").unwrap();
        assert_eq!(face_cell.level(), 0);

        // Token "3" is face 1, level 0.
        let face1 = S2CellId::from_token("3").unwrap();
        assert_eq!(face1.level(), 0);
        assert_eq!(face1.face(), 1);
    }

    #[test]
    fn face_zero_level_zero() {
        // Face 0 level 0: id = 0x1000000000000000
        let cell = S2CellId::from_raw(0x1000000000000000);
        assert_eq!(cell.face(), 0);
        assert_eq!(cell.level(), 0);
        assert_eq!(cell.to_token(), "1");
    }

    #[test]
    fn parent_level() {
        let token = "89c25c";
        let cell = S2CellId::from_token(token).unwrap();
        let level = cell.level();
        let parent = cell.parent().unwrap();
        assert_eq!(parent.level(), level - 1);
    }

    #[test]
    fn children_roundtrip() {
        let cell = S2CellId::from_token("89c25c").unwrap();
        let children = cell.children();
        for c in &children {
            assert_eq!(c.parent().unwrap(), cell);
        }
    }

    #[test]
    fn child_position_range() {
        let cell = S2CellId::from_token("89c25c").unwrap();
        for (i, c) in cell.children().iter().enumerate() {
            assert_eq!(c.child_position(), i as u32);
        }
    }

    #[test]
    fn subtree_hilbert_position_root() {
        let cell = S2CellId::from_token("89c25c").unwrap();
        // The cell relative to itself should be position 0.
        assert_eq!(cell.subtree_hilbert_position(cell), 0);
    }

    #[test]
    fn subtree_hilbert_position_children() {
        let parent = S2CellId::from_token("89c25c").unwrap();
        let children = parent.children();
        for (i, c) in children.iter().enumerate() {
            assert_eq!(c.subtree_hilbert_position(parent), i as u64);
        }
    }

    #[test]
    fn bounds_plausible() {
        // Face 0 level 0 should cover roughly ±45 degrees latitude and -45..45 longitude.
        let cell = S2CellId::from_raw(0x1000000000000000);
        let (west, south, east, north) = cell.lat_lon_bounds_radians();
        assert!(west < east, "west < east");
        assert!(south < north, "south < north");
        assert!(south > -PI / 2.0 - 0.01);
        assert!(north < PI / 2.0 + 0.01);
    }
}

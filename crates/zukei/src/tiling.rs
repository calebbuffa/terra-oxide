use crate::{AxisAlignedBoundingBox, Rectangle, morton};
use glam::{DVec2, DVec3};

/// A rectangular range of available tiles at one zoom level in a quadtree.
///
/// Mirrors `CesiumGeometry::QuadtreeTileRectangularRange`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuadtreeTileRectangularRange {
    pub level: u32,
    pub start_x: u32,
    pub start_y: u32,
    pub end_x: u32,
    pub end_y: u32,
}

///
/// The root tile is `{ level: 0, x: 0, y: 0 }`. At each subsequent level the
/// tile count doubles in both axes, so a tile at level `L` covers
/// `1/2^L` of the root bounding volume along each axis.
/// `x` increases west-to-east; `y` increases south-to-north.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct QuadtreeTileID {
    /// Depth from the root (0 = root).
    pub level: u32,
    /// Column within the level-grid. Range: `[0, 2^level)`.
    pub x: u32,
    /// Row within the level-grid. Range: `[0, 2^level)`.
    pub y: u32,
}

impl QuadtreeTileID {
    #[inline]
    pub const fn new(level: u32, x: u32, y: u32) -> Self {
        Self { level, x, y }
    }

    /// Return the parent tile ID, or `None` if this is the root.
    #[inline]
    pub const fn parent(self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            Some(Self::new(self.level - 1, self.x >> 1, self.y >> 1))
        }
    }

    /// Return the four children of this tile as a lazy iterable.
    #[inline]
    pub fn children(self) -> QuadtreeChildren {
        QuadtreeChildren::new(self)
    }

    /// Compute the absolute Morton index for a quadtree tile at its level.
    ///
    /// The Morton (Z-order) index interleaves the bits of `x` and `y`.
    #[inline]
    pub fn morton_index(&self) -> u64 {
        morton::spread_bits_2d(self.x) | (morton::spread_bits_2d(self.y) << 1)
    }

    /// Return the root tile of the subtree that contains `tile`.
    ///
    /// `subtree_levels` is the number of levels in each subtree (the
    /// `subtreeLevels` field in the `ImplicitTiling` JSON object).
    pub fn subtree_root(&self, subtree_levels: u32) -> QuadtreeTileID {
        let subtree_level = self.level / subtree_levels;
        let levels_left = self.level % subtree_levels;
        QuadtreeTileID::new(
            subtree_level * subtree_levels,
            self.x >> levels_left,
            self.y >> levels_left,
        )
    }

    /// Morton index of `tile` relative to the subtree rooted at `subtree_root`.
    pub fn relative_morton_index(&self, tile: QuadtreeTileID) -> u64 {
        let rel = self.relative_to(tile);
        rel.morton_index()
    }

    /// Convert an absolute tile ID to one relative to `self` assuming self is the root.
    pub fn relative_to(&self, tile: QuadtreeTileID) -> QuadtreeTileID {
        let relative_level = tile.level - self.level;
        QuadtreeTileID::new(
            relative_level,
            tile.x - (self.x << relative_level),
            tile.y - (self.y << relative_level),
        )
    }
}

impl std::fmt::Display for QuadtreeTileID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}", self.level, self.x, self.y)
    }
}

/// Unique identifier for a tile in an implicit octree.
///
/// The root tile is `{ level: 0, x: 0, y: 0, z: 0 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct OctreeTileID {
    /// Depth from the root (0 = root).
    pub level: u32,
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl OctreeTileID {
    #[inline]
    pub const fn new(level: u32, x: u32, y: u32, z: u32) -> Self {
        Self { level, x, y, z }
    }

    /// Return the parent tile ID, or `None` if this is the root.
    #[inline]
    pub const fn parent(self) -> Option<Self> {
        if self.level == 0 {
            None
        } else {
            Some(Self::new(
                self.level - 1,
                self.x >> 1,
                self.y >> 1,
                self.z >> 1,
            ))
        }
    }

    /// Return the eight children of this tile as a lazy iterable.
    #[inline]
    pub fn children(self) -> OctreeChildren {
        OctreeChildren::new(self)
    }

    /// Compute the absolute Morton index for an octree tile at its level.
    ///
    /// Interleaves the bits of `x`, `y`, and `z`.
    #[inline]
    pub fn morton_index(&self) -> u64 {
        morton::spread_bits_3d(self.x)
            | (morton::spread_bits_3d(self.y) << 1)
            | (morton::spread_bits_3d(self.z) << 2)
    }

    /// Return the root tile of the subtree that contains this tile.
    #[inline]
    pub fn subtree_root(&self, subtree_levels: u32) -> Self {
        let subtree_level = self.level / subtree_levels;
        let levels_left = self.level % subtree_levels;
        OctreeTileID {
            level: subtree_level * subtree_levels,
            x: self.x >> levels_left,
            y: self.y >> levels_left,
            z: self.z >> levels_left,
        }
    }

    /// Morton index of `tile` relative to the subtree rooted at `subtree_root`.
    pub fn relative_morton_index(&self, tile: OctreeTileID) -> u64 {
        let rel = self.relative_to(tile);
        rel.morton_index()
    }

    /// Convert an absolute tile ID to one relative to `self` assuming self is the root.
    pub fn relative_to(&self, tile: OctreeTileID) -> OctreeTileID {
        let relative_level = tile.level - self.level;
        OctreeTileID::new(
            relative_level,
            tile.x - (self.x << relative_level),
            tile.y - (self.y << relative_level),
            tile.z - (self.z << relative_level),
        )
    }
}

impl std::fmt::Display for OctreeTileID {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}/{}/{}", self.level, self.x, self.y, self.z)
    }
}

/// A lazy, non-allocating container yielding the four child [`QuadtreeTileID`]s.
///
/// Children are ordered: `(x*2, y*2)`, `(x*2+1, y*2)`, `(x*2, y*2+1)`,
/// `(x*2+1, y*2+1)`.
#[derive(Debug, Clone, Copy)]
pub struct QuadtreeChildren {
    parent: QuadtreeTileID,
}

impl QuadtreeChildren {
    pub(crate) fn new(parent: QuadtreeTileID) -> Self {
        Self { parent }
    }

    /// Always 4.
    pub const fn len(&self) -> usize {
        4
    }

    /// Never empty.
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// Iterator over the four children of a quadtree tile.
pub struct QuadtreeChildrenIter {
    parent: QuadtreeTileID,
    index: u32,
}

impl Iterator for QuadtreeChildrenIter {
    type Item = QuadtreeTileID;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= 4 {
            return None;
        }
        let i = self.index;
        self.index += 1;
        Some(QuadtreeTileID::new(
            self.parent.level + 1,
            self.parent.x * 2 + (i & 1),
            self.parent.y * 2 + (i >> 1),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (4 - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for QuadtreeChildrenIter {}

impl IntoIterator for QuadtreeChildren {
    type Item = QuadtreeTileID;
    type IntoIter = QuadtreeChildrenIter;

    fn into_iter(self) -> Self::IntoIter {
        QuadtreeChildrenIter {
            parent: self.parent,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a QuadtreeChildren {
    type Item = QuadtreeTileID;
    type IntoIter = QuadtreeChildrenIter;

    fn into_iter(self) -> Self::IntoIter {
        QuadtreeChildrenIter {
            parent: self.parent,
            index: 0,
        }
    }
}

/// A lazy, non-allocating container yielding the eight child [`OctreeTileID`]s.
///
/// Children are ordered by `(dx, dy, dz)` in bit order 0–7.
#[derive(Debug, Clone, Copy)]
pub struct OctreeChildren {
    parent: OctreeTileID,
}

impl OctreeChildren {
    pub(crate) fn new(parent: OctreeTileID) -> Self {
        Self { parent }
    }

    /// Always 8.
    pub const fn len(&self) -> usize {
        8
    }

    /// Never empty.
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// Iterator over the eight children of an octree tile.
pub struct OctreeChildrenIter {
    parent: OctreeTileID,
    index: u32,
}

impl Iterator for OctreeChildrenIter {
    type Item = OctreeTileID;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= 8 {
            return None;
        }
        let i = self.index;
        self.index += 1;
        Some(OctreeTileID::new(
            self.parent.level + 1,
            self.parent.x * 2 + (i & 1),
            self.parent.y * 2 + ((i >> 1) & 1),
            self.parent.z * 2 + (i >> 2),
        ))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (8 - self.index) as usize;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for OctreeChildrenIter {}

impl IntoIterator for OctreeChildren {
    type Item = OctreeTileID;
    type IntoIter = OctreeChildrenIter;

    fn into_iter(self) -> Self::IntoIter {
        OctreeChildrenIter {
            parent: self.parent,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a OctreeChildren {
    type Item = OctreeTileID;
    type IntoIter = OctreeChildrenIter;

    fn into_iter(self) -> Self::IntoIter {
        OctreeChildrenIter {
            parent: self.parent,
            index: 0,
        }
    }
}

/// Defines how a rectangular region of projected space is divided into an
/// implicit quadtree.
#[derive(Debug, Clone)]
pub struct QuadtreeTilingScheme {
    rectangle: Rectangle,
    root_tiles_x: u32,
    root_tiles_y: u32,
}

impl QuadtreeTilingScheme {
    /// Create a new tiling scheme.
    ///
    /// # Panics
    /// Panics if `root_tiles_x` or `root_tiles_y` is zero.
    pub fn new(rectangle: Rectangle, root_tiles_x: u32, root_tiles_y: u32) -> Self {
        assert!(
            root_tiles_x > 0 && root_tiles_y > 0,
            "root tile counts must be positive"
        );
        Self {
            rectangle,
            root_tiles_x,
            root_tiles_y,
        }
    }

    pub fn rectangle(&self) -> &Rectangle {
        &self.rectangle
    }

    pub fn root_tiles_x(&self) -> u32 {
        self.root_tiles_x
    }

    pub fn root_tiles_y(&self) -> u32 {
        self.root_tiles_y
    }

    /// Total number of tiles in the X direction at `level`.
    pub fn tiles_x_at_level(&self, level: u32) -> u32 {
        self.root_tiles_x << level
    }

    /// Total number of tiles in the Y direction at `level`.
    pub fn tiles_y_at_level(&self, level: u32) -> u32 {
        self.root_tiles_y << level
    }

    /// Return the projected [`Rectangle`] for a given tile.
    ///
    /// Returns `None` if `x` or `y` are out of range for the given `level`.
    pub fn tile_to_rectangle(&self, tile: QuadtreeTileID) -> Option<Rectangle> {
        let nx = self.tiles_x_at_level(tile.level) as f64;
        let ny = self.tiles_y_at_level(tile.level) as f64;
        if tile.x as f64 >= nx || tile.y as f64 >= ny {
            return None;
        }
        let w = self.rectangle.width() / nx;
        let h = self.rectangle.height() / ny;
        let min_x = self.rectangle.minimum_x + tile.x as f64 * w;
        let min_y = self.rectangle.minimum_y + tile.y as f64 * h;
        Some(Rectangle::new(min_x, min_y, min_x + w, min_y + h))
    }

    /// Return the tile ID that contains the given projected position at `level`.
    ///
    /// Returns `None` if the position is outside the root rectangle.
    pub fn position_to_tile(&self, x: f64, y: f64, level: u32) -> Option<QuadtreeTileID> {
        if !self.rectangle.contains(DVec2::new(x, y)) {
            return None;
        }
        let nx = self.tiles_x_at_level(level) as f64;
        let ny = self.tiles_y_at_level(level) as f64;
        let tx = ((x - self.rectangle.minimum_x) / self.rectangle.width() * nx)
            .min(nx - 1.0)
            .max(0.0) as u32;
        let ty = ((y - self.rectangle.minimum_y) / self.rectangle.height() * ny)
            .min(ny - 1.0)
            .max(0.0) as u32;
        Some(QuadtreeTileID::new(level, tx, ty))
    }

    /// Infer the tile that contains a geographic position from an angular width.
    ///
    /// Derives the tile level from `lon_width` (width of the tile in radians
    /// relative to the root tile width), then locates `(lon_west, lat_south)`
    /// within the grid at that level.  Returns `None` if the position maps
    /// outside this scheme's rectangle or the inferred tile's rectangle does
    /// not match `(lon_west, lat_south)` within a 1 × 10⁻⁹ radian tolerance.
    pub fn tile_for_angular_width(
        &self,
        lon_west: f64,
        lat_south: f64,
        lon_east: f64,
        lat_north: f64,
    ) -> Option<QuadtreeTileID> {
        let _ = lat_north; // only the width and SW corner are used
        let tile_lon_width = lon_east - lon_west;
        let root_lon_width = self.rectangle.width() / self.root_tiles_x as f64;

        // level = log2(root_lon_width / tile_lon_width)
        let ratio = root_lon_width / tile_lon_width;
        let level = ratio.log2().round() as u32;

        let nx = self.tiles_x_at_level(level) as f64;
        let ny = self.tiles_y_at_level(level) as f64;

        let tx = ((lon_west - self.rectangle.minimum_x) / self.rectangle.width() * nx)
            .round()
            .clamp(0.0, nx - 1.0) as u32;
        let ty = ((lat_south - self.rectangle.minimum_y) / self.rectangle.height() * ny)
            .round()
            .clamp(0.0, ny - 1.0) as u32;

        let id = QuadtreeTileID::new(level, tx, ty);
        // Sanity-check: verify the reconstructed rectangle matches.
        let rect = self.tile_to_rectangle(id)?;
        const EPS: f64 = 1e-9;
        if (rect.minimum_x - lon_west).abs() > EPS || (rect.minimum_y - lat_south).abs() > EPS {
            return None;
        }
        Some(id)
    }

    /// Geographic tiling scheme: covers `[-\pi, -\pi/2] -> [\pi, \pi/2]` with a 2 x 1
    /// root grid (two 90-degree-wide tiles at the root).
    pub fn geographic() -> Self {
        use std::f64::consts::PI;
        Self::new(Rectangle::new(-PI, -PI / 2.0, PI, PI / 2.0), 2, 1)
    }

    /// Web Mercator tiling scheme: covers `[-\pi, -\pi] -> [\pi, \pi]` in
    /// easting/northing with a 1 x 1 root grid.
    pub fn web_mercator() -> Self {
        const HALF_SIZE: f64 = 20_037_508.342_789_244;
        Self::new(
            Rectangle::new(-HALF_SIZE, -HALF_SIZE, HALF_SIZE, HALF_SIZE),
            1,
            1,
        )
    }
}

/// Defines how an axis-aligned box is divided into an implicit octree.
#[derive(Debug, Clone)]
pub struct OctreeTilingScheme {
    bounding_box: AxisAlignedBoundingBox,
    root_tiles_x: u32,
    root_tiles_y: u32,
    root_tiles_z: u32,
}

impl OctreeTilingScheme {
    /// Create a new octree tiling scheme.
    ///
    /// # Panics
    /// Panics if any root tile count is zero.
    pub fn new(
        bounding_box: AxisAlignedBoundingBox,
        root_tiles_x: u32,
        root_tiles_y: u32,
        root_tiles_z: u32,
    ) -> Self {
        assert!(
            root_tiles_x > 0 && root_tiles_y > 0 && root_tiles_z > 0,
            "root tile counts must be positive"
        );
        Self {
            bounding_box,
            root_tiles_x,
            root_tiles_y,
            root_tiles_z,
        }
    }

    pub fn bounding_box(&self) -> &AxisAlignedBoundingBox {
        &self.bounding_box
    }

    pub fn root_tiles_x(&self) -> u32 {
        self.root_tiles_x
    }
    pub fn root_tiles_y(&self) -> u32 {
        self.root_tiles_y
    }
    pub fn root_tiles_z(&self) -> u32 {
        self.root_tiles_z
    }

    pub fn tiles_x_at_level(&self, level: u32) -> u32 {
        self.root_tiles_x << level
    }
    pub fn tiles_y_at_level(&self, level: u32) -> u32 {
        self.root_tiles_y << level
    }
    pub fn tiles_z_at_level(&self, level: u32) -> u32 {
        self.root_tiles_z << level
    }

    /// Return the AABB for `tile`, or `None` if any coordinate is out of range.
    pub fn tile_to_box(&self, tile: OctreeTileID) -> Option<AxisAlignedBoundingBox> {
        let nx = self.tiles_x_at_level(tile.level) as f64;
        let ny = self.tiles_y_at_level(tile.level) as f64;
        let nz = self.tiles_z_at_level(tile.level) as f64;
        if tile.x as f64 >= nx || tile.y as f64 >= ny || tile.z as f64 >= nz {
            return None;
        }
        let size = self.bounding_box.max - self.bounding_box.min;
        let tw = DVec3::new(size.x / nx, size.y / ny, size.z / nz);
        let min = self.bounding_box.min
            + DVec3::new(
                tile.x as f64 * tw.x,
                tile.y as f64 * tw.y,
                tile.z as f64 * tw.z,
            );
        Some(AxisAlignedBoundingBox::new(min, min + tw))
    }

    /// Return the tile that contains `position` at `level`, or `None` if
    /// the position is outside the bounding box.
    pub fn position_to_tile(&self, position: DVec3, level: u32) -> Option<OctreeTileID> {
        if !self.bounding_box.contains(position) {
            return None;
        }
        let size = self.bounding_box.max - self.bounding_box.min;
        let nx = self.tiles_x_at_level(level) as f64;
        let ny = self.tiles_y_at_level(level) as f64;
        let nz = self.tiles_z_at_level(level) as f64;
        let rel = position - self.bounding_box.min;
        let tx = ((rel.x / size.x * nx).min(nx - 1.0).max(0.0)) as u32;
        let ty = ((rel.y / size.y * ny).min(ny - 1.0).max(0.0)) as u32;
        let tz = ((rel.z / size.z * nz).min(nz - 1.0).max(0.0)) as u32;
        Some(OctreeTileID {
            level,
            x: tx,
            y: ty,
            z: tz,
        })
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn morton_quad_root() {
        let id = QuadtreeTileID::new(0, 0, 0);
        assert_eq!(id.morton_index(), 0);
    }

    #[test]
    fn morton_quad_level1() {
        // Morton index interleaves x (LSB) and y: x=bit0, y=bit1.
        // (x=0,y=0)->0, (x=1,y=0)->1, (x=0,y=1)->2, (x=1,y=1)->3
        let expected: &[(u64, u32, u32)] = &[(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)];
        for &(want, x, y) in expected {
            let id = QuadtreeTileID::new(1, x, y);
            assert_eq!(id.morton_index(), want, "x={x} y={y}");
        }
    }

    #[test]
    fn morton_oct_root() {
        let id = OctreeTileID::new(0, 0, 0, 0);
        assert_eq!(id.morton_index(), 0);
    }

    #[test]
    fn morton_oct_level1() {
        // 8 children; Morton index is the 3-bit interleaved index of (z,y,x).
        for i in 0u32..8 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = i >> 2;
            let id = OctreeTileID::new(1, x, y, z);
            assert_eq!(id.morton_index(), i as u64, "i={i} x={x} y={y} z={z}");
        }
    }

    #[test]
    fn test_get_subtree_root_quad() {
        // Subtree levels = 4: tiles 0-3 in first subtree, 4-7 in next.
        let tile = QuadtreeTileID::new(5, 6, 7);
        let root = tile.subtree_root(4);
        assert_eq!(root.level, 4);
        assert_eq!(root.x, 6 >> 1); // 5 % 4 = 1 level below subtree root
        assert_eq!(root.y, 7 >> 1);
    }

    #[test]
    fn test_get_subtree_root_oct() {
        let tile = OctreeTileID::new(5, 6, 7, 4);
        let root = tile.subtree_root(4);
        assert_eq!(root.level, 4);
        assert_eq!(root.x, 6 >> 1);
        assert_eq!(root.y, 7 >> 1);
        assert_eq!(root.z, 4 >> 1);
    }

    #[test]
    fn test_absolute_to_relative_quad() {
        let id = QuadtreeTileID::new(3, 5, 6);
        let rel = id.relative_to(id);
        assert_eq!(rel, QuadtreeTileID::new(0, 0, 0));
    }
}

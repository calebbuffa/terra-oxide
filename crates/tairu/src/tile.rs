//! Tile transform and bounding volume helpers.

use glam::{DMat4, DVec3};
use terra::BoundingRegion;
use zukei::{BoundingSphere, OrientedBoundingBox};

use crate::generated::{BoundingVolume, Tile};

/// Functions for reading and writing a [`Tile`]'s transform.
pub struct TileTransform;

impl TileTransform {
    /// Parse the tile's `transform` array into a [`DMat4`].
    ///
    /// Returns `None` if the array has fewer than 16 elements.
    /// Extra elements beyond index 15 are silently ignored.
    pub fn get_transform(tile: &Tile) -> Option<DMat4> {
        let a = &tile.transform;
        if a.len() < 16 {
            return None;
        }
        Some(DMat4::from_cols_array(&[
            a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13],
            a[14], a[15],
        ]))
    }

    /// Write a [`DMat4`] into a tile's `transform` array, replacing any
    /// existing value.
    pub fn set_transform(tile: &mut Tile, transform: DMat4) {
        let a = transform.to_cols_array();
        tile.transform = a.to_vec();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{DMat4, DVec3};

    fn tile_with_identity() -> Tile {
        let mut tile = Tile::default();
        TileTransform::set_transform(&mut tile, DMat4::IDENTITY);
        tile
    }

    #[test]
    fn identity_round_trip() {
        let tile = tile_with_identity();
        let m = TileTransform::get_transform(&tile).unwrap();
        assert!((m - DMat4::IDENTITY).abs_diff_eq(DMat4::ZERO, 1e-15));
    }

    #[test]
    fn get_transform_none_on_empty() {
        let tile = Tile::default();
        assert!(TileTransform::get_transform(&tile).is_none());
    }

    #[test]
    fn translation_round_trip() {
        let t = DMat4::from_translation(DVec3::new(100.0, 200.0, 300.0));
        let mut tile = Tile::default();
        TileTransform::set_transform(&mut tile, t);
        let back = TileTransform::get_transform(&tile).unwrap();
        assert!((back.w_axis.truncate() - DVec3::new(100.0, 200.0, 300.0)).length() < 1e-10);
    }
}

/// Functions for extracting and setting typed bounding volumes on
/// [`BoundingVolume`] values.
pub struct TileBoundingVolumes;

impl TileBoundingVolumes {
    /// Parse the `box` field of a [`BoundingVolume`] into an
    /// [`OrientedBoundingBox`].
    ///
    /// Returns `None` if `bounding_volume.box` has fewer than 12 elements.
    pub fn get_oriented_bounding_box(
        bounding_volume: &BoundingVolume,
    ) -> Option<OrientedBoundingBox> {
        OrientedBoundingBox::from_array(&bounding_volume.r#box)
    }

    /// Write an [`OrientedBoundingBox`] into the `box` field of a
    /// [`BoundingVolume`], replacing any existing value.
    pub fn set_oriented_bounding_box(
        bounding_volume: &mut BoundingVolume,
        obb: OrientedBoundingBox,
    ) {
        let arr = obb.to_array();
        bounding_volume.r#box = arr.to_vec();
    }

    /// Parse the `region` field of a [`BoundingVolume`] into a
    /// [`BoundingRegion`].
    ///
    /// The six floats are `[west_rad, south_rad, east_rad, north_rad,
    /// min_height_m, max_height_m]`.
    ///
    /// Returns `None` if `bounding_volume.region` has fewer than 6 elements.
    pub fn get_bounding_region(bounding_volume: &BoundingVolume) -> Option<BoundingRegion> {
        BoundingRegion::from_array(&bounding_volume.region)
    }

    /// Write a [`BoundingRegion`] into the `region` field of a
    /// [`BoundingVolume`], replacing any existing value.
    pub fn set_bounding_region(bounding_volume: &mut BoundingVolume, region: BoundingRegion) {
        bounding_volume.region = region.to_array().to_vec();
    }

    /// Parse the `sphere` field of a [`BoundingVolume`] into a
    /// [`BoundingSphere`].
    ///
    /// The four floats are `[cx, cy, cz, radius]`.
    ///
    /// Returns `None` if `bounding_volume.sphere` has fewer than 4 elements.
    pub fn get_bounding_sphere(bounding_volume: &BoundingVolume) -> Option<BoundingSphere> {
        let s = &bounding_volume.sphere;
        if s.len() < 4 {
            return None;
        }
        Some(BoundingSphere::new(DVec3::new(s[0], s[1], s[2]), s[3]))
    }

    /// Write a [`BoundingSphere`] into the `sphere` field of a
    /// [`BoundingVolume`], replacing any existing value.
    pub fn set_bounding_sphere(bounding_volume: &mut BoundingVolume, sphere: BoundingSphere) {
        bounding_volume.sphere = vec![
            sphere.center.x,
            sphere.center.y,
            sphere.center.z,
            sphere.radius,
        ];
    }
}

/// Binary tile format detected from the URL or magic bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFormat {
    /// glTF binary blob (magic `glTF`).
    Glb,
    /// Batched 3D Model (magic `b3dm`).
    B3dm,
    /// Instanced 3D Model (magic `i3dm`).
    I3dm,
    /// Composite (magic `cmpt`).
    Cmpt,
    /// Point cloud (magic `pnts`).
    Pnts,
    /// External tileset (JSON).
    Json,
    /// Unknown / unrecognised.
    Unknown,
}

impl TileFormat {
    /// Detect the format from the first four bytes of the response body.
    /// Falls back to URL-based detection when the magic is not recognised.
    pub fn detect(url: &str, data: &[u8]) -> Self {
        if data.len() >= 4 {
            match &data[..4] {
                b"glTF" => return Self::Glb,
                b"b3dm" => return Self::B3dm,
                b"i3dm" => return Self::I3dm,
                b"cmpt" => return Self::Cmpt,
                b"pnts" => return Self::Pnts,
                _ => {}
            }
        }
        // Fallback to extension.
        let path = url.split('?').next().unwrap_or(url);
        match path
            .rsplit('.')
            .next()
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("glb") => Self::Glb,
            Some("b3dm") => Self::B3dm,
            Some("i3dm") => Self::I3dm,
            Some("cmpt") => Self::Cmpt,
            Some("pnts") => Self::Pnts,
            Some("json") => Self::Json,
            _ => Self::Unknown,
        }
    }
}

#[cfg(test)]
mod bounding_volume_tests {
    use super::*;

    #[test]
    fn get_oriented_bounding_box_none_short() {
        let bv = BoundingVolume::default();
        assert!(TileBoundingVolumes::get_oriented_bounding_box(&bv).is_none());
    }

    #[test]
    fn get_bounding_region_round_trip() {
        use std::f64::consts::PI;
        let raw = [-PI, -PI / 2.0, PI, PI / 2.0, -100.0, 500.0];
        let mut bv = BoundingVolume {
            region: raw.to_vec(),
            ..Default::default()
        };

        let region = TileBoundingVolumes::get_bounding_region(&bv).unwrap();
        assert!((region.rectangle.west + PI).abs() < 1e-15);
        assert!((region.maximum_height - 500.0).abs() < 1e-10);

        TileBoundingVolumes::set_bounding_region(&mut bv, region);
        let region2 = TileBoundingVolumes::get_bounding_region(&bv).unwrap();
        assert!((region2.minimum_height + 100.0).abs() < 1e-10);
    }

    #[test]
    fn get_bounding_sphere_round_trip() {
        let mut bv = BoundingVolume {
            sphere: vec![1.0, 2.0, 3.0, 500.0],
            ..Default::default()
        };
        let sphere = TileBoundingVolumes::get_bounding_sphere(&bv).unwrap();
        assert!((sphere.center - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-10);
        assert!((sphere.radius - 500.0).abs() < 1e-10);

        TileBoundingVolumes::set_bounding_sphere(&mut bv, sphere);
        let sphere2 = TileBoundingVolumes::get_bounding_sphere(&bv).unwrap();
        assert!((sphere2.radius - 500.0).abs() < 1e-10);
    }

    #[test]
    fn get_bounding_sphere_none_short() {
        let bv = BoundingVolume::default();
        assert!(TileBoundingVolumes::get_bounding_sphere(&bv).is_none());
    }
}

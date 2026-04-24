//! Cesium `heightmap-1.0` terrain tile codec.
//!
//! Reference: <https://github.com/CesiumGS/quantized-mesh>
//! (heightmap-1.0 section)
//!
//! # Wire format (uncompressed)
//!
//! ```text
//! [0 .. 2*65*65)   heights     u16 LE × 4225  (row-major, south-to-north)
//! [8450]           child_flags u8              SW=bit0 SE=bit1 NW=bit2 NE=bit3
//! [8451 ..)        water_mask  1 byte  (all-land=0x00 / all-water=0x01)
//!                           OR 256×256 bytes (per-pixel mask)
//! ```
//!
//! Tiles are typically stored gzip-compressed on disk; this module handles both
//! compressed and raw input transparently.

use std::io::Read;

use thiserror::Error;

use outil::io::{BufferReader, BufferWriter, UnexpectedEndOfData};

/// Width / height (in samples) of a heightmap tile, including the one-pixel
/// border shared with adjacent tiles.
pub const TILE_SIZE: usize = 65;

/// Total number of height samples per tile (`65 × 65`).
pub const TILE_CELL_SIZE: usize = TILE_SIZE * TILE_SIZE;

/// Width / height (in pixels) of the full water mask.
pub const MASK_SIZE: usize = 256;

/// Total pixels in the full water mask (`256 × 256`).
pub const MASK_CELL_SIZE: usize = MASK_SIZE * MASK_SIZE;

/// Minimum uncompressed size: heights + child byte + 1-byte mask sentinel.
const MIN_TERRAIN_SIZE: usize = TILE_CELL_SIZE * 2 + 1 + 1;

/// Maximum uncompressed size: heights + child byte + full mask.
const MAX_TERRAIN_SIZE: usize = TILE_CELL_SIZE * 2 + 1 + MASK_CELL_SIZE;

/// gzip magic bytes.
const GZIP_MAGIC: [u8; 2] = [0x1F, 0x8B];

/// Bit flags indicating which child tiles exist.
///
/// Matches the CTB / Cesium spec: SW=bit 0, SE=bit 1, NW=bit 2, NE=bit 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ChildFlags(pub u8);

impl ChildFlags {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0b0000_1111);

    pub fn sw(self) -> bool {
        self.0 & 0b0001 != 0
    }
    pub fn se(self) -> bool {
        self.0 & 0b0010 != 0
    }
    pub fn nw(self) -> bool {
        self.0 & 0b0100 != 0
    }
    pub fn ne(self) -> bool {
        self.0 & 0b1000 != 0
    }

    pub fn with_sw(self, on: bool) -> Self {
        self.set_bit(0, on)
    }
    pub fn with_se(self, on: bool) -> Self {
        self.set_bit(1, on)
    }
    pub fn with_nw(self, on: bool) -> Self {
        self.set_bit(2, on)
    }
    pub fn with_ne(self, on: bool) -> Self {
        self.set_bit(3, on)
    }

    fn set_bit(self, bit: u8, on: bool) -> Self {
        if on {
            Self(self.0 | (1 << bit))
        } else {
            Self(self.0 & !(1 << bit))
        }
    }
}

/// The water mask embedded in a heightmap tile.
#[derive(Debug, Clone, PartialEq)]
pub enum WaterMask {
    /// Tile is entirely land (mask byte = 0x00).
    Land,
    /// Tile is entirely water (mask byte = 0x01).
    Water,
    /// Per-pixel mask: 256×256 bytes, row-major south-to-north.
    /// `0` = land, `255` = water, values in between indicate partial coverage.
    Mask(Box<[u8; MASK_CELL_SIZE]>),
}

impl Default for WaterMask {
    fn default() -> Self {
        Self::Land
    }
}

/// A decoded `heightmap-1.0` terrain tile.
///
/// Heights are row-major, ordered south-to-north, west-to-east.
/// Each value is a `u16` in the range `[0, 32767]` where `0` maps to
/// −1000 m and `32767` maps to +9000 m (linear interpolation).
#[derive(Debug, Clone)]
pub struct HeightmapTile {
    pub heights: [u16; TILE_CELL_SIZE],
    pub children: ChildFlags,
    pub water_mask: WaterMask,
}

impl HeightmapTile {
    /// Create an all-zero (sea-level), all-land tile with no children.
    pub fn empty() -> Self {
        Self {
            heights: [0u16; TILE_CELL_SIZE],
            children: ChildFlags::NONE,
            water_mask: WaterMask::Land,
        }
    }
}

/// Errors produced while decoding a heightmap tile.
#[derive(Debug, Error)]
pub enum HeightmapError {
    #[error("unexpected end of data")]
    Truncated,

    /// The uncompressed payload is the wrong size to be a valid heightmap.
    #[error("invalid heightmap size: {0} bytes (expected {MIN_TERRAIN_SIZE}..={MAX_TERRAIN_SIZE})")]
    BadSize(usize),

    #[error("gzip decompression failed: {0}")]
    Gzip(#[source] std::io::Error),
}

impl From<UnexpectedEndOfData> for HeightmapError {
    fn from(_: UnexpectedEndOfData) -> Self {
        Self::Truncated
    }
}

/// Decode a `heightmap-1.0` tile from `data`.
///
/// Accepts both gzip-compressed (as served by terrain servers) and raw
/// uncompressed bytes.
pub fn decode_heightmap(data: &[u8]) -> Result<HeightmapTile, HeightmapError> {
    if data.starts_with(&GZIP_MAGIC) {
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut buf = Vec::with_capacity(MAX_TERRAIN_SIZE);
        decoder
            .read_to_end(&mut buf)
            .map_err(HeightmapError::Gzip)?;
        decode_raw(&buf)
    } else {
        decode_raw(data)
    }
}

fn decode_raw(data: &[u8]) -> Result<HeightmapTile, HeightmapError> {
    match data.len() {
        n if n == MIN_TERRAIN_SIZE || n == MAX_TERRAIN_SIZE => {}
        n => return Err(HeightmapError::BadSize(n)),
    }

    let mut r = BufferReader::new(data);

    // Heights: 4225 × u16 LE
    let mut heights = [0u16; TILE_CELL_SIZE];
    for h in heights.iter_mut() {
        *h = r.read_le::<u16>()?;
    }

    // Child flags
    let children = ChildFlags(r.read_le::<u8>()?);

    // Water mask
    let water_mask = match r.remaining() {
        1 => {
            let byte = r.read_le::<u8>()?;
            if byte == 0 {
                WaterMask::Land
            } else {
                WaterMask::Water
            }
        }
        MASK_CELL_SIZE => {
            let bytes = r.read_bytes(MASK_CELL_SIZE)?;
            let mut mask = Box::new([0u8; MASK_CELL_SIZE]);
            mask.copy_from_slice(bytes);
            WaterMask::Mask(mask)
        }
        n => return Err(HeightmapError::BadSize(n)),
    };

    Ok(HeightmapTile {
        heights,
        children,
        water_mask,
    })
}

/// Encode a [`HeightmapTile`] to raw (uncompressed) bytes.
///
/// The returned bytes match the wire format exactly. Compress with gzip before
/// writing to disk or serving over HTTP if required.
pub fn encode_heightmap(tile: &HeightmapTile) -> Vec<u8> {
    let mask_len = match &tile.water_mask {
        WaterMask::Mask(_) => MASK_CELL_SIZE,
        _ => 1,
    };
    let capacity = TILE_CELL_SIZE * 2 + 1 + mask_len;

    let mut w = BufferWriter::with_capacity(capacity);

    for &h in &tile.heights {
        w.write_le(h);
    }

    w.write_le(tile.children.0);

    match &tile.water_mask {
        WaterMask::Land => w.write_le(0u8),
        WaterMask::Water => w.write_le(1u8),
        WaterMask::Mask(mask) => w.write_bytes(mask.as_ref()),
    }

    w.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(tile: &HeightmapTile) -> HeightmapTile {
        let encoded = encode_heightmap(tile);
        decode_heightmap(&encoded).expect("round-trip failed")
    }

    #[test]
    fn round_trip_land_tile() {
        let mut tile = HeightmapTile::empty();
        tile.heights[0] = 100;
        tile.heights[TILE_CELL_SIZE - 1] = 32767;
        tile.children = ChildFlags::ALL;

        let rt = round_trip(&tile);
        assert_eq!(rt.heights[0], 100);
        assert_eq!(rt.heights[TILE_CELL_SIZE - 1], 32767);
        assert_eq!(rt.children, ChildFlags::ALL);
        assert_eq!(rt.water_mask, WaterMask::Land);
    }

    #[test]
    fn round_trip_water_tile() {
        let mut tile = HeightmapTile::empty();
        tile.water_mask = WaterMask::Water;
        tile.children = ChildFlags::NONE.with_sw(true).with_ne(true);

        let rt = round_trip(&tile);
        assert_eq!(rt.water_mask, WaterMask::Water);
        assert!(rt.children.sw());
        assert!(!rt.children.se());
        assert!(!rt.children.nw());
        assert!(rt.children.ne());
    }

    #[test]
    fn round_trip_full_water_mask() {
        let mut tile = HeightmapTile::empty();
        let mut mask = Box::new([0u8; MASK_CELL_SIZE]);
        mask[0] = 255;
        mask[MASK_CELL_SIZE - 1] = 128;
        tile.water_mask = WaterMask::Mask(mask);

        let rt = round_trip(&tile);
        let WaterMask::Mask(m) = &rt.water_mask else {
            panic!("expected Mask variant");
        };
        assert_eq!(m[0], 255);
        assert_eq!(m[MASK_CELL_SIZE - 1], 128);
    }

    #[test]
    fn child_flags_individual_bits() {
        let f = ChildFlags::NONE.with_sw(true).with_ne(true);
        assert!(f.sw());
        assert!(!f.se());
        assert!(!f.nw());
        assert!(f.ne());
        assert_eq!(f.0, 0b0000_1001);
    }

    #[test]
    fn decode_bad_size_errors() {
        let bad = vec![0u8; 100];
        assert!(matches!(
            decode_heightmap(&bad),
            Err(HeightmapError::BadSize(100))
        ));
    }

    #[test]
    fn encode_size_land() {
        let tile = HeightmapTile::empty();
        assert_eq!(encode_heightmap(&tile).len(), MIN_TERRAIN_SIZE);
    }

    #[test]
    fn encode_size_full_mask() {
        let mut tile = HeightmapTile::empty();
        tile.water_mask = WaterMask::Mask(Box::new([0u8; MASK_CELL_SIZE]));
        assert_eq!(encode_heightmap(&tile).len(), MAX_TERRAIN_SIZE);
    }
}

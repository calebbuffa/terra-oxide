//! Compressed vertex-attribute utilities.
//!
//! Mirrors `CesiumUtility::AttributeCompression`.

use glam::DVec3;

use crate::{from_snorm, sign_not_zero};

/// Decode an oct-encoded unit-length vector from a SNORM range of `[0, range_max]`.
///
/// The `x` and `y` parameters are the oct-encoded integer components.
/// `range_max` is the maximum value of the integer range
/// (e.g. `255` for u8, `65535` for u16).
///
/// Returns the decoded, normalised `DVec3`.
///
/// Mirrors `CesiumUtility::AttributeCompression::octDecodeInRange`.
pub fn oct_decode_in_range(x: u32, y: u32, range_max: u32) -> DVec3 {
    let rm = range_max as f64;
    let mut vx = from_snorm(x as f64, rm);
    let mut vy = from_snorm(y as f64, rm);
    let vz = 1.0 - (vx.abs() + vy.abs());
    if vz < 0.0 {
        let old_vx = vx;
        vx = (1.0 - vy.abs()) * sign_not_zero(old_vx);
        vy = (1.0 - old_vx.abs()) * sign_not_zero(vy);
    }
    DVec3::new(vx, vy, vz).normalize_or_zero()
}

/// Decode a 1-byte-per-component oct-encoded unit-length normal.
///
/// Each component is in `[0, 255]` (u8 SNORM range).
///
/// Mirrors `CesiumUtility::AttributeCompression::octDecode`.
#[inline]
pub fn oct_decode(x: u8, y: u8) -> DVec3 {
    oct_decode_in_range(x as u32, y as u32, 255)
}

/// Decode a 2-byte-per-component oct-encoded unit-length normal.
///
/// Each component is in `[0, 65535]` (u16 SNORM range).
/// This corresponds to the `NORMAL_OCT16P` semantic in legacy 3D Tiles PNTS.
///
/// Mirrors `CesiumUtility::AttributeCompression::octDecodeInRange<uint16_t, 65535>`.
#[inline]
pub fn oct_decode_16p(x: u16, y: u16) -> DVec3 {
    oct_decode_in_range(x as u32, y as u32, 65535)
}

/// Decode an RGB565-packed colour into normalised linear `[0, 1]` per-channel values.
///
/// **Note:** the three channels use different bit widths (5/6/5), so the
/// normalisation denominators differ: red/blue divide by 31, green divides by 63.
/// The values are in sRGB space — apply gamma correction separately if linear
/// output is required.
///
/// Mirrors `CesiumUtility::AttributeCompression::decodeRGB565`.
#[inline]
pub fn decode_rgb565(value: u16) -> DVec3 {
    let r = ((value >> 11) & 0x1F) as f64 / 31.0;
    let g = ((value >> 5) & 0x3F) as f64 / 63.0;
    let b = (value & 0x1F) as f64 / 31.0;
    DVec3::new(r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit vector along +Z oct-encodes as (range_max/2, range_max/2).
    #[test]
    fn oct_decode_in_range_z_axis() {
        // x=y=127 (approximately half of 255) -> decoded normal should point ~+Z.
        let v = oct_decode(127, 127);
        // Not exactly (0,0,1) due to SNORM rounding, but should be close.
        assert!(v.z > 0.9, "z={}", v.z);
        assert!((v.length() - 1.0).abs() < 1e-6, "not unit: len={}", v.length());
    }

    /// Convenience wrapper must agree with the range-generic function.
    #[test]
    fn oct_decode_u8_matches_in_range() {
        let a = oct_decode(200, 100);
        let b = oct_decode_in_range(200, 100, 255);
        assert!((a - b).length() < 1e-12);
    }

    #[test]
    fn oct_decode_16p_unit_length() {
        let v = oct_decode_16p(32768, 32768);
        assert!((v.length() - 1.0).abs() < 1e-6, "not unit: len={}", v.length());
    }

    #[test]
    fn decode_rgb565_black() {
        let v = decode_rgb565(0x0000);
        assert_eq!(v, DVec3::ZERO);
    }

    #[test]
    fn decode_rgb565_white() {
        let v = decode_rgb565(0xFFFF);
        // All channels should be 1.0 (31/31, 63/63, 31/31).
        assert!((v.x - 1.0).abs() < 1e-10);
        assert!((v.y - 1.0).abs() < 1e-10);
        assert!((v.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn decode_rgb565_channel_isolation() {
        // Pure red: bits [15:11] = 0b11111, rest zero -> 0xF800.
        let v = decode_rgb565(0xF800);
        assert!((v.x - 1.0).abs() < 1e-10, "r={}", v.x);
        assert!(v.y.abs() < 1e-10, "g={}", v.y);
        assert!(v.z.abs() < 1e-10, "b={}", v.z);

        // Pure green: bits [10:5] = 0b111111, rest zero -> 0x07E0.
        let v = decode_rgb565(0x07E0);
        assert!(v.x.abs() < 1e-10, "r={}", v.x);
        assert!((v.y - 1.0).abs() < 1e-10, "g={}", v.y);
        assert!(v.z.abs() < 1e-10, "b={}", v.z);

        // Pure blue: bits [4:0] = 0b11111, rest zero -> 0x001F.
        let v = decode_rgb565(0x001F);
        assert!(v.x.abs() < 1e-10, "r={}", v.x);
        assert!(v.y.abs() < 1e-10, "g={}", v.y);
        assert!((v.z - 1.0).abs() < 1e-10, "b={}", v.z);
    }
}

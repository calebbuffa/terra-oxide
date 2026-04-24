//! Morton (Z-order) curve encoding.
//!
//! Interleaves the bits of 2D or 3D integer coordinates into a single 64-bit
//! Morton index, preserving spatial locality for quadtree and octree tile
//! enumeration.

/// Spread the bits of a 32-bit integer into the even bits of a 64-bit word,
/// leaving zeros in the odd bits. Used for 2D Morton (Z-order) encoding.
///
/// For n bits of input the output occupies bits 0, 2, 4, … 2*(n-1).
/// Handles up to 32-bit inputs safely.
#[inline]
pub fn spread_bits_2d(n: u32) -> u64 {
    let mut x = n as u64;
    x = (x | (x << 16)) & 0x0000_FFFF_0000_FFFF;
    x = (x | (x << 8)) & 0x00FF_00FF_00FF_00FF;
    x = (x | (x << 4)) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

/// Spread the bits of a 21-bit integer into every third bit of a 64-bit word.
/// The output occupies bits 0, 3, 6, … 60. Handles octree levels 0–21.
#[inline]
pub fn spread_bits_3d(n: u32) -> u64 {
    // Limit to 21 bits — octree coordinates at level 21 are at most 2^21−1.
    let mut x = (n & 0x001F_FFFF) as u64;
    x = (x | (x << 32)) & 0x001F_0000_0000_FFFF;
    x = (x | (x << 16)) & 0x001F_0000_FF00_00FF;
    x = (x | (x << 8)) & 0x100F_00F0_0F00_F00F;
    x = (x | (x << 4)) & 0x10C3_0C30_C30C_30C3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

/// Compute the 2D Morton index for a quadtree tile at `(x, y)`.
///
/// Interleaves the bits of `x` (even bits) and `y` (odd bits).
#[inline]
pub fn morton_2d(x: u32, y: u32) -> u64 {
    spread_bits_2d(x) | (spread_bits_2d(y) << 1)
}

/// Compute the 3D Morton index for an octree tile at `(x, y, z)`.
///
/// Interleaves the bits of `x`, `y`, and `z`.
#[inline]
pub fn morton_3d(x: u32, y: u32, z: u32) -> u64 {
    spread_bits_3d(x) | (spread_bits_3d(y) << 1) | (spread_bits_3d(z) << 2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spread_bits_2d_zero() {
        assert_eq!(spread_bits_2d(0), 0);
    }

    #[test]
    fn spread_bits_2d_one() {
        assert_eq!(spread_bits_2d(1), 1);
    }

    #[test]
    fn spread_bits_2d_two() {
        // 0b10 -> bits spread to positions 0,2 -> 0b0100 = 4
        assert_eq!(spread_bits_2d(2), 4);
    }

    #[test]
    fn morton_2d_level1() {
        // (x=0,y=0)->0, (x=1,y=0)->1, (x=0,y=1)->2, (x=1,y=1)->3
        assert_eq!(morton_2d(0, 0), 0);
        assert_eq!(morton_2d(1, 0), 1);
        assert_eq!(morton_2d(0, 1), 2);
        assert_eq!(morton_2d(1, 1), 3);
    }

    #[test]
    fn morton_3d_root() {
        assert_eq!(morton_3d(0, 0, 0), 0);
    }

    #[test]
    fn morton_3d_level1_all_children() {
        for i in 0u32..8 {
            let x = i & 1;
            let y = (i >> 1) & 1;
            let z = i >> 2;
            assert_eq!(morton_3d(x, y, z), i as u64, "i={i}");
        }
    }
}

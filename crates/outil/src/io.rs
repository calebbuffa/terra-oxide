//! Zero-copy binary buffer reader and in-memory buffer writer.

use thiserror::Error;

/// Returned by [`BufferReader`] when a read would exceed the buffer bounds.
#[derive(Debug, Error)]
#[error("unexpected end of data")]
pub struct UnexpectedEndOfData;

// ── LeBytes ──────────────────────────────────────────────────────────────

/// Little-endian encoding and decoding for fixed-size numeric types.
///
/// Implemented for all primitive numeric types. Implement this for your own
/// fixed-size types, keeping `SIZE`, `from_le`, and `write_le` consistent.
pub trait LeBytes: Sized + Copy {
    /// Byte width of this type.
    const SIZE: usize;
    /// Decode `Self` from a little-endian byte slice of exactly `SIZE` bytes.
    fn from_le(bytes: &[u8]) -> Self;
    /// Encode `self` as little-endian bytes and append them to `buf`.
    fn write_le(self, buf: &mut Vec<u8>);
}

macro_rules! impl_le_bytes {
    ($($t:ty),*) => {
        $(impl LeBytes for $t {
            const SIZE: usize = std::mem::size_of::<$t>();
            fn from_le(bytes: &[u8]) -> Self {
                Self::from_le_bytes(bytes.try_into().unwrap())
            }
            fn write_le(self, buf: &mut Vec<u8>) {
                buf.extend_from_slice(&self.to_le_bytes());
            }
        })*
    };
}

impl_le_bytes!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);

/// Zero-copy cursor over a byte slice.
///
/// Use [`BufferReader::read_le`] to read any [`LeBytes`] primitive and
/// [`BufferReader::read_bytes`] to borrow a sub-slice without allocation.
/// All reads advance the internal position or return [`UnexpectedEndOfData`].
///
/// The lifetime `'a` is threaded through [`BufferReader::read_bytes`], allowing
/// zero-allocation sub-slices that borrow directly from the original buffer.
pub struct BufferReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufferReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Current byte offset.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Number of bytes left after the current position.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    /// Jump to an absolute byte offset.
    ///
    /// # Panics (debug only)
    /// Panics if `pos > data.len()`.
    pub fn seek(&mut self, pos: usize) {
        debug_assert!(pos <= self.data.len(), "seek past end");
        self.pos = pos;
    }

    /// Advance to the next 4-byte boundary (no-op if already aligned).
    pub fn align_to_4(&mut self) {
        let rem = self.pos % 4;
        if rem != 0 {
            self.pos += 4 - rem;
        }
    }

    fn require(&self, n: usize) -> Result<(), UnexpectedEndOfData> {
        if self.pos + n <= self.data.len() {
            Ok(())
        } else {
            Err(UnexpectedEndOfData)
        }
    }

    /// Read a single little-endian `T` and advance by `T::SIZE` bytes.
    ///
    /// # Example
    /// ```
    /// # use outil::io::BufferReader;
    /// let mut r = BufferReader::new(&[0x01, 0x00, 0x02, 0x00]);
    /// assert_eq!(r.read_le::<u16>().unwrap(), 1_u16);
    /// assert_eq!(r.read_le::<u16>().unwrap(), 2_u16);
    /// ```
    pub fn read_le<T: LeBytes>(&mut self) -> Result<T, UnexpectedEndOfData> {
        self.require(T::SIZE)?;
        let v = T::from_le(&self.data[self.pos..self.pos + T::SIZE]);
        self.pos += T::SIZE;
        Ok(v)
    }

    /// Read `count` consecutive little-endian `T` values into a `Vec<T>`.
    ///
    /// Checks bounds once up-front before allocating.
    ///
    /// # Example
    /// ```
    /// # use outil::io::BufferReader;
    /// let bytes: Vec<u8> = [1u16, 2, 3].iter().flat_map(|v| v.to_le_bytes()).collect();
    /// let mut r = BufferReader::new(&bytes);
    /// assert_eq!(r.read_le_vec::<u16>(3).unwrap(), vec![1, 2, 3]);
    /// ```
    pub fn read_le_vec<T: LeBytes>(&mut self, count: usize) -> Result<Vec<T>, UnexpectedEndOfData> {
        let byte_len = count * T::SIZE;
        self.require(byte_len)?;
        let values = self.data[self.pos..self.pos + byte_len]
            .chunks_exact(T::SIZE)
            .map(T::from_le)
            .collect();
        self.pos += byte_len;
        Ok(values)
    }

    /// Return a sub-slice of `n` bytes, borrowing from the original `'a` buffer.
    pub fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], UnexpectedEndOfData> {
        self.require(n)?;
        let slice = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }
}

// ── BufferWriter ─────────────────────────────────────────────────────────────

/// In-memory little-endian byte buffer builder.
///
/// Symmetric counterpart to [`BufferReader`]: use [`BufferWriter::write_le`]
/// to append any [`LeBytes`] value and [`BufferWriter::write_bytes`] to append
/// raw slices. Call [`BufferWriter::finish`] to consume the writer and get the
/// underlying `Vec<u8>`.
///
/// # Example
/// ```
/// # use outil::io::BufferWriter;
/// let mut w = BufferWriter::new();
/// w.write_le(0x01u8);
/// w.write_le(0x0200u16);
/// assert_eq!(w.finish(), vec![0x01, 0x00, 0x02]);
/// ```
pub struct BufferWriter {
    buf: Vec<u8>,
}

impl BufferWriter {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    /// Create a writer pre-allocated to hold `capacity` bytes.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: Vec::with_capacity(capacity),
        }
    }

    /// Current byte length of the buffer.
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    /// Returns `true` if no bytes have been written yet.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Append `value` as little-endian bytes.
    pub fn write_le<T: LeBytes>(&mut self, value: T) {
        value.write_le(&mut self.buf);
    }

    /// Append a slice of little-endian `T` values.
    pub fn write_le_slice<T: LeBytes>(&mut self, values: &[T]) {
        for &v in values {
            v.write_le(&mut self.buf);
        }
    }

    /// Append raw bytes.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
    }

    /// Pad with `pad_byte` until `len()` is a multiple of `align`.
    /// No-op if already aligned or `align` is 0.
    pub fn align_to(&mut self, align: usize, pad_byte: u8) {
        if align == 0 {
            return;
        }
        let rem = self.buf.len() % align;
        if rem != 0 {
            self.buf
                .extend(std::iter::repeat(pad_byte).take(align - rem));
        }
    }

    /// Consume the writer and return the underlying buffer.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

impl Default for BufferWriter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_primitives() {
        let data: &[u8] = &[0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00];
        let mut r = BufferReader::new(data);
        assert_eq!(r.read_le::<u8>().unwrap(), 0x01);
        assert_eq!(r.read_le::<u8>().unwrap(), 0x02);
        assert_eq!(r.read_le::<u32>().unwrap(), 0x0000_0300);
        // 1 byte remains — a u8 read succeeds but a u32 read does not
        assert!(r.read_le::<u8>().is_ok());
        assert!(r.read_le::<u32>().is_err());
    }

    #[test]
    fn reads_float() {
        let v = 1.5f32;
        let bytes = v.to_le_bytes();
        let mut r = BufferReader::new(&bytes);
        assert_eq!(r.read_le::<f32>().unwrap(), 1.5f32);
    }

    #[test]
    fn reads_signed() {
        let v = -1i16;
        let bytes = v.to_le_bytes();
        let mut r = BufferReader::new(&bytes);
        assert_eq!(r.read_le::<i16>().unwrap(), -1i16);
    }

    #[test]
    fn seek_and_position() {
        let data = [0u8; 16];
        let mut r = BufferReader::new(&data);
        r.seek(8);
        assert_eq!(r.position(), 8);
        assert_eq!(r.remaining(), 8);
    }

    #[test]
    fn align_to_4() {
        let data = [0u8; 16];
        let mut r = BufferReader::new(&data);
        r.seek(3);
        r.align_to_4();
        assert_eq!(r.position(), 4);
        r.seek(4);
        r.align_to_4();
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn read_bytes_borrows_original_lifetime() {
        let data: Vec<u8> = (0u8..8).collect();
        let mut r = BufferReader::new(&data);
        let slice = r.read_bytes(4).unwrap();
        assert_eq!(slice, &[0, 1, 2, 3]);
        assert_eq!(r.position(), 4);
    }

    #[test]
    fn writer_round_trips() {
        let mut w = BufferWriter::new();
        w.write_le(0x01u8);
        w.write_le(0x0200u16);
        w.write_le(-1i32);
        w.write_le(1.5f32);
        let bytes = w.finish();

        let mut r = BufferReader::new(&bytes);
        assert_eq!(r.read_le::<u8>().unwrap(), 0x01);
        assert_eq!(r.read_le::<u16>().unwrap(), 0x0200);
        assert_eq!(r.read_le::<i32>().unwrap(), -1);
        assert_eq!(r.read_le::<f32>().unwrap(), 1.5f32);
    }

    #[test]
    fn writer_align_to() {
        let mut w = BufferWriter::new();
        w.write_le(0x01u8); // len = 1
        w.align_to(4, 0x20); // pad to 4 with spaces
        assert_eq!(w.len(), 4);
        let bytes = w.finish();
        assert_eq!(bytes, &[0x01, 0x20, 0x20, 0x20]);
    }

    #[test]
    fn writer_write_le_slice() {
        let vals = [1u16, 2, 3];
        let mut w = BufferWriter::new();
        w.write_le_slice(&vals);
        let bytes = w.finish();
        let mut r = BufferReader::new(&bytes);
        assert_eq!(r.read_le_vec::<u16>(3).unwrap(), vec![1, 2, 3]);
    }
}

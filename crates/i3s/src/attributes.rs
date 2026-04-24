//! I3S attribute binary buffer decoder.
//!
//! # Binary format
//!
//! Each attribute resource has the following layout:
//! ```text
//! [uint32 count]                             ← feature count from header
//! [uint32[] byte_counts]  (string only)      ← byte length of each string value,
//!                                               including the null terminator
//! [packed typed values]                      ← count x sizeof(type) bytes
//! ```
//!
//! The value type is read from `AttributeStorageInfo::attribute_values.valueType`.
//! String attributes have a `attribute_byte_counts` section between the count
//! header and the UTF-8 payload.

use crate::cmn::{AttributeStorageInfo, HeaderValueType};
use outil::io::{BufferReader, UnexpectedEndOfData};
use std::collections::HashMap;

/// A decoded attribute buffer for one field of one node.
#[derive(Debug, Clone)]
pub struct AttributeBuffer {
    /// Field name matching `AttributeStorageInfo::name`.
    pub field_name: String,
    /// Decoded values for all features in the node.
    pub values: AttributeValues,
}

/// The decoded value sequence for an attribute field.
#[derive(Debug, Clone)]
pub enum AttributeValues {
    Int32(Vec<i32>),
    UInt32(Vec<u32>),
    Int64(Vec<i64>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Utf8(Vec<String>),
}

impl AttributeValues {
    /// Number of decoded values.
    pub fn len(&self) -> usize {
        match self {
            Self::Int32(v) => v.len(),
            Self::UInt32(v) => v.len(),
            Self::Int64(v) => v.len(),
            Self::UInt64(v) => v.len(),
            Self::Float32(v) => v.len(),
            Self::Float64(v) => v.len(),
            Self::Utf8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Errors from the attribute decode pipeline.
#[derive(Debug, thiserror::Error)]
pub enum AttributeDecodeError {
    /// Buffer is shorter than the header/payload requires.
    #[error("attribute buffer truncated")]
    Truncated,
    /// Value type string in the descriptor is not recognised.
    #[error("unknown attribute value type: {0}")]
    UnknownValueType(String),
    /// String payload contains invalid UTF-8.
    #[error("invalid UTF-8 at feature {feature_index}: {error}")]
    InvalidUtf8 {
        feature_index: usize,
        #[source]
        error: std::str::Utf8Error,
    },
    /// Attribute descriptor has no `attributeValues` section.
    #[error("attributeStorageInfo has no attributeValues descriptor")]
    MissingAttributeValues,
    /// `count` from the binary header exceeds the safety limit.
    #[error("attribute count {count} exceeds maximum allowed ({max})")]
    CountTooLarge { count: usize, max: usize },
    /// Byte-length arithmetic overflowed.
    #[error("attribute buffer arithmetic overflow")]
    Overflow,
}

impl From<UnexpectedEndOfData> for AttributeDecodeError {
    fn from(_: UnexpectedEndOfData) -> Self {
        AttributeDecodeError::Truncated
    }
}

/// Decode a raw I3S attribute binary buffer into an [`AttributeBuffer`].
///
/// # Arguments
///
/// * `data` - raw bytes fetched from
///   `layers/{id}/nodes/{n}/attributes/{info.key}/0`.
/// * `info` - the `AttributeStorageInfo` descriptor for this field from
///   `layer.attributeStorageInfo`.
pub fn decode_attribute(
    data: &[u8],
    info: &AttributeStorageInfo,
) -> Result<AttributeBuffer, AttributeDecodeError> {
    let mut r = BufferReader::new(data);
    let count = r.read_le::<u32>()? as usize;

    // Reject absurdly large counts before allocating anything.
    const MAX_ATTRIBUTE_COUNT: usize = 10_000_000;
    if count > MAX_ATTRIBUTE_COUNT {
        return Err(AttributeDecodeError::CountTooLarge {
            count,
            max: MAX_ATTRIBUTE_COUNT,
        });
    }

    let value_type = extract_value_type(info)?;

    let values = match value_type {
        HeaderValueType::Int32 => AttributeValues::Int32(r.read_le_vec::<i32>(count)?),
        HeaderValueType::UInt32 => AttributeValues::UInt32(r.read_le_vec::<u32>(count)?),
        HeaderValueType::Int16 => {
            // Widen i16 -> i32 - no explicit Int16 variant needed by callers.
            AttributeValues::Int32(
                r.read_le_vec::<i16>(count)?
                    .into_iter()
                    .map(|x| x as i32)
                    .collect(),
            )
        }
        HeaderValueType::UInt16 => AttributeValues::UInt32(
            r.read_le_vec::<u16>(count)?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        ),
        HeaderValueType::Int8 => AttributeValues::Int32(
            r.read_le_vec::<i8>(count)?
                .into_iter()
                .map(|x| x as i32)
                .collect(),
        ),
        HeaderValueType::UInt8 => AttributeValues::UInt32(
            r.read_le_vec::<u8>(count)?
                .into_iter()
                .map(|x| x as u32)
                .collect(),
        ),
        HeaderValueType::Float32 => AttributeValues::Float32(r.read_le_vec::<f32>(count)?),
        HeaderValueType::Float64 => AttributeValues::Float64(r.read_le_vec::<f64>(count)?),
        HeaderValueType::String => {
            // Strings: byte-count array first, then raw UTF-8 bytes.
            // byte_counts[i] includes the null terminator.
            let byte_counts = r.read_le_vec::<u32>(count)?;
            let mut strings = Vec::with_capacity(count);
            for (i, &byte_count) in byte_counts.iter().enumerate() {
                let slice = r.read_bytes(byte_count as usize)?;
                // Strip trailing null if present.
                let slice = if slice.last().copied() == Some(0) {
                    &slice[..slice.len() - 1]
                } else {
                    slice
                };
                let s =
                    std::str::from_utf8(slice).map_err(|e| AttributeDecodeError::InvalidUtf8 {
                        feature_index: i,
                        error: e,
                    })?;
                strings.push(s.to_owned());
            }
            AttributeValues::Utf8(strings)
        }
        HeaderValueType::Unknown => {
            return Err(AttributeDecodeError::UnknownValueType("Unknown".into()));
        }
    };

    Ok(AttributeBuffer {
        field_name: info.name.clone(),
        values,
    })
}

/// A single decoded property value from an I3S attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    Int32(i32),
    UInt32(u32),
    Int64(i64),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
}

/// Decode an I3S attribute binary buffer and return a map of
/// `attribute_name -> Vec<PropertyValue>`.
///
/// This is a convenience wrapper around [`decode_attribute`] that converts
/// the typed [`AttributeValues`] into a uniform [`PropertyValue`] vec.
pub fn decode_i3s_attributes(
    data: &[u8],
    storage_info: &AttributeStorageInfo,
) -> Result<HashMap<String, Vec<PropertyValue>>, AttributeDecodeError> {
    let buf = decode_attribute(data, storage_info)?;
    let values: Vec<PropertyValue> = match buf.values {
        AttributeValues::Int32(v) => v.into_iter().map(PropertyValue::Int32).collect(),
        AttributeValues::UInt32(v) => v.into_iter().map(PropertyValue::UInt32).collect(),
        AttributeValues::Int64(v) => v.into_iter().map(PropertyValue::Int64).collect(),
        AttributeValues::UInt64(v) => v.into_iter().map(PropertyValue::UInt64).collect(),
        AttributeValues::Float32(v) => v.into_iter().map(PropertyValue::Float32).collect(),
        AttributeValues::Float64(v) => v.into_iter().map(PropertyValue::Float64).collect(),
        AttributeValues::Utf8(v) => v.into_iter().map(PropertyValue::String).collect(),
    };
    let mut map = HashMap::new();
    map.insert(buf.field_name, values);
    Ok(map)
}

/// Extract the value type from `AttributeStorageInfo::attribute_values.value_type`.
fn extract_value_type(
    info: &AttributeStorageInfo,
) -> Result<HeaderValueType, AttributeDecodeError> {
    let val = info
        .attribute_values
        .as_ref()
        .ok_or(AttributeDecodeError::MissingAttributeValues)?;

    let type_str = val.value_type.as_str();

    let vt = match type_str {
        "Int8" => HeaderValueType::Int8,
        "UInt8" => HeaderValueType::UInt8,
        "Int16" => HeaderValueType::Int16,
        "UInt16" => HeaderValueType::UInt16,
        "Int32" => HeaderValueType::Int32,
        "UInt32" => HeaderValueType::UInt32,
        "Float32" => HeaderValueType::Float32,
        "Float64" => HeaderValueType::Float64,
        "String" | "Oid32" | "Oid64" => HeaderValueType::String,
        _ => return Err(AttributeDecodeError::UnknownValueType(type_str.to_owned())),
    };
    Ok(vt)
}

/// Whether the `ordering` field includes `AttributeByteCounts`, indicating
/// a string-type attribute.
pub fn has_byte_counts(info: &AttributeStorageInfo) -> bool {
    info.ordering.as_deref().unwrap_or(&[]).iter().any(|o| {
        matches!(
            o,
            crate::cmn::AttributeStorageInfoOrdering::AttributeByteCounts
        )
    })
}

/// Build a minimal `AttributeStorageInfo` for testing without a JSON layer doc.
#[cfg(test)]
fn make_info(name: &str, value_type: &str) -> AttributeStorageInfo {
    use crate::cmn::Value;
    AttributeStorageInfo {
        key: format!("f_{name}"),
        name: name.to_owned(),
        header: vec![],
        ordering: None,
        attribute_values: Some(Value {
            value_type: value_type.to_owned(),
            encoding: None,
            time_encoding: None,
            values_per_element: Some(1.0),
        }),
        attribute_byte_counts: None,
        object_ids: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn u32_le(v: u32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn i32_le(v: i32) -> [u8; 4] {
        v.to_le_bytes()
    }
    fn f64_le(v: f64) -> [u8; 8] {
        v.to_le_bytes()
    }
    fn f32_le(v: f32) -> [u8; 4] {
        v.to_le_bytes()
    }

    /// Build a buffer: [count_u32] ++ payload bytes
    fn buf(count: u32, payload: &[u8]) -> Vec<u8> {
        let mut v = u32_le(count).to_vec();
        v.extend_from_slice(payload);
        v
    }

    #[test]
    fn i32_three_values() {
        let info = make_info("population", "Int32");
        let payload: Vec<u8> = [i32_le(10), i32_le(-5), i32_le(999)]
            .iter()
            .flatten()
            .copied()
            .collect();
        let attr = decode_attribute(&buf(3, &payload), &info).unwrap();
        assert_eq!(attr.field_name, "population");
        let AttributeValues::Int32(v) = attr.values else {
            panic!("expected Int32");
        };
        assert_eq!(v, [10, -5, 999]);
    }

    #[test]
    fn f64_two_values() {
        let info = make_info("height", "Float64");
        let payload: Vec<u8> = [f64_le(3.14), f64_le(2.72)]
            .iter()
            .flatten()
            .copied()
            .collect();
        let attr = decode_attribute(&buf(2, &payload), &info).unwrap();
        let AttributeValues::Float64(v) = attr.values else {
            panic!("expected Float64");
        };
        assert!((v[0] - 3.14).abs() < 1e-9);
        assert!((v[1] - 2.72).abs() < 1e-9);
    }

    #[test]
    fn f32_values() {
        let info = make_info("temp", "Float32");
        let payload: Vec<u8> = [f32_le(1.0), f32_le(2.0)]
            .iter()
            .flatten()
            .copied()
            .collect();
        let attr = decode_attribute(&buf(2, &payload), &info).unwrap();
        let AttributeValues::Float32(v) = attr.values else {
            panic!("expected Float32");
        };
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn uint32_values() {
        let info = make_info("oid", "UInt32");
        let payload: Vec<u8> = [u32_le(1), u32_le(2), u32_le(3)]
            .iter()
            .flatten()
            .copied()
            .collect();
        let attr = decode_attribute(&buf(3, &payload), &info).unwrap();
        let AttributeValues::UInt32(v) = attr.values else {
            panic!("expected UInt32");
        };
        assert_eq!(v, [1, 2, 3]);
    }

    #[test]
    fn string_values_with_null_terminator() {
        let info = make_info("name", "String");
        // "Hello\0" (6 bytes), "Hi\0" (3 bytes)
        let s1 = b"Hello\0";
        let s2 = b"Hi\0";
        let mut payload: Vec<u8> = Vec::new();
        // byte counts array
        payload.extend_from_slice(&u32_le(s1.len() as u32));
        payload.extend_from_slice(&u32_le(s2.len() as u32));
        // string payloads
        payload.extend_from_slice(s1);
        payload.extend_from_slice(s2);

        let attr = decode_attribute(&buf(2, &payload), &info).unwrap();
        let AttributeValues::Utf8(v) = attr.values else {
            panic!("expected Utf8");
        };
        assert_eq!(v, ["Hello", "Hi"]);
    }

    #[test]
    fn string_without_null_terminator() {
        let info = make_info("code", "String");
        let s = b"ABC"; // 3 bytes, no null
        let mut payload: Vec<u8> = Vec::new();
        payload.extend_from_slice(&u32_le(s.len() as u32));
        payload.extend_from_slice(s);

        let attr = decode_attribute(&buf(1, &payload), &info).unwrap();
        let AttributeValues::Utf8(v) = attr.values else {
            panic!("expected Utf8");
        };
        assert_eq!(v, ["ABC"]);
    }

    #[test]
    fn truncated_header_returns_error() {
        let info = make_info("x", "Int32");
        let result = decode_attribute(&[], &info); // no count header
        assert!(matches!(
            result,
            Err(AttributeDecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn truncated_payload_returns_error() {
        let info = make_info("x", "Int32");
        // count = 3, but only 4 bytes of payload (need 12)
        let result = decode_attribute(&buf(3, &[0u8; 4]), &info);
        assert!(matches!(
            result,
            Err(AttributeDecodeError::Truncated { .. })
        ));
    }

    #[test]
    fn missing_attribute_values_returns_error() {
        let info = AttributeStorageInfo {
            key: "f_x".into(),
            name: "x".into(),
            header: vec![],
            ordering: None,
            attribute_values: None, // absent
            attribute_byte_counts: None,
            object_ids: None,
        };
        let result = decode_attribute(&buf(0, &[]), &info);
        assert!(matches!(
            result,
            Err(AttributeDecodeError::MissingAttributeValues)
        ));
    }

    #[test]
    fn unknown_value_type_returns_error() {
        let info = make_info("x", "Binary"); // unsupported
        let result = decode_attribute(&buf(0, &[]), &info);
        assert!(matches!(
            result,
            Err(AttributeDecodeError::UnknownValueType(_))
        ));
    }

    #[test]
    fn zero_count_returns_empty_values() {
        let info = make_info("x", "Int32");
        let attr = decode_attribute(&buf(0, &[]), &info).unwrap();
        assert!(attr.values.is_empty());
    }

    #[test]
    fn i16_widened_to_i32() {
        let info = make_info("val", "Int16");
        let mut payload = Vec::new();
        payload.extend_from_slice(&(-100i16).to_le_bytes());
        payload.extend_from_slice(&(200i16).to_le_bytes());
        let attr = decode_attribute(&buf(2, &payload), &info).unwrap();
        let AttributeValues::Int32(v) = attr.values else {
            panic!("expected Int32 (widened from i16)");
        };
        assert_eq!(v, [-100, 200]);
    }

    #[test]
    fn u8_widened_to_u32() {
        let info = make_info("flags", "UInt8");
        let attr = decode_attribute(&buf(3, &[0u8, 128u8, 255u8]), &info).unwrap();
        let AttributeValues::UInt32(v) = attr.values else {
            panic!("expected UInt32 (widened from u8)");
        };
        assert_eq!(v, [0, 128, 255]);
    }
}

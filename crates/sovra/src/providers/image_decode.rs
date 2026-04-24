//! Decode raster overlay tile payloads (PNG/JPEG/WebP) into raw RGBA.
//!
//! Kept provider-agnostic so URL-template, WMS, WMTS, and any future raster
//! source can share the same decode path without reaching into each other's
//! modules.

pub(crate) struct DecodedImage {
    pub(crate) pixels: Vec<u8>,
    pub(crate) width: u32,
    pub(crate) height: u32,
}

/// Decode a PNG/JPEG/WebP byte buffer into raw RGBA pixel data.
pub(crate) fn decode_image_to_rgba(data: &[u8]) -> Result<DecodedImage, image::ImageError> {
    let img = image::load_from_memory(data)?;
    let rgba = img.into_rgba8();
    Ok(DecodedImage {
        width: rgba.width(),
        height: rgba.height(),
        pixels: rgba.into_raw(),
    })
}

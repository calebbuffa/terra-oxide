use super::CodecRegistry;
use crate::reader::{
    Warning, Warnings, apply_texture_transforms, decode_data_urls, dequantize,
    resolve_external_refs,
};
use crate::{GltfModel, reader::GltfReaderOptions};

/// Run the full post-processing pipeline on a freshly parsed glTF model.
///
/// Order:
/// 1. External file URI resolution (filesystem)
/// 2. Data URL decoding
/// 3. Codec registry decode (image -> draco -> meshopt -> spz -> ktx2 by default)
/// 4. Dequantization
/// 5. Texture transform
pub fn decode_model(
    options: &GltfReaderOptions,
    registry: &CodecRegistry,
    model: &mut GltfModel,
    warnings: &mut Warnings,
) {
    // 1. External file URI resolution (non-data: URIs loaded from disk).
    if options.images.resolve_external_references {
        if let Some(base_path) = &options.images.base_path {
            resolve_external_refs(model, base_path, warnings);
        }
    }

    // 2. Data URL decoding.
    if options.images.decode_data_urls {
        decode_data_urls(model, options.images.clear_decoded_data_urls, warnings);
    }

    // 3. Run all registered codecs (image decode, draco, meshopt, spz, ktx2…).
    if options.images.decode_embedded_images {
        {
            for msg in registry.decode_all(model) {
                warnings.push(Warning(msg));
            }
        }
    }

    // 4. Dequantization.
    if options.mesh.dequantize {
        dequantize(model, warnings);
    }

    // 5. Texture transform.
    if options.mesh.apply_texture_transform {
        apply_texture_transforms(model, warnings);
    }
}

use crate::cmn::{GeometryDefinitionTopology, SceneLayerInfo};

/// Select the best geometry buffer for a given layer.
///
/// Returns `(definition_index, buffer_index)` into
/// `layer.geometry_definitions[def].geometry_buffers[buf]`.
///
/// Selection criteria (highest priority first):
/// 1. Triangle topology only - other topologies are not supported.
/// 2. Prefer a buffer with `compressed_attributes` (Draco) - smallest on the wire.
/// 3. Otherwise prefer the buffer with the most vertex attributes present
///    (normal + uv0 + color + uv_region each count as 1 point).
///
/// Returns `None` if no triangle-topology geometry definition exists.
pub fn select_geometry_buffer(layer: &SceneLayerInfo) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, i32)> = None; // (def, buf, score)

    for (def_idx, def) in layer.geometry_definitions.iter().enumerate() {
        // Skip non-triangle topologies.
        let topo = def
            .topology
            .as_ref()
            .unwrap_or(&GeometryDefinitionTopology::Triangle);
        if *topo != GeometryDefinitionTopology::Triangle {
            continue;
        }

        for (buf_idx, buf) in def.geometry_buffers.iter().enumerate() {
            let score = if buf.compressed_attributes.is_some() {
                // Draco wins outright - give it a very high score.
                1000
            } else {
                // Score = number of optional vertex attributes present.
                let mut s: i32 = 0;
                if buf.normal.is_some() {
                    s += 1;
                }
                if buf.uv0.is_some() {
                    s += 1;
                }
                if buf.color.is_some() {
                    s += 1;
                }
                if buf.uv_region.is_some() {
                    s += 1;
                }
                s
            };

            let better = best.map_or(true, |(_, _, prev)| score > prev);
            if better {
                best = Some((def_idx, buf_idx, score));
            }
        }
    }

    best.map(|(def, buf, _)| (def, buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cmn::{
        CompressedAttributes, CompressedAttributesAttributes, CompressedAttributesEncoding,
        GeometryBuffer, GeometryDefinition, GeometryNormal, SceneLayerInfo,
    };

    fn layer_with_buffers(bufs: Vec<GeometryBuffer>) -> SceneLayerInfo {
        SceneLayerInfo {
            geometry_definitions: vec![GeometryDefinition {
                topology: Some(GeometryDefinitionTopology::Triangle),
                geometry_buffers: bufs,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn prefers_draco_over_uncompressed() {
        let layer = layer_with_buffers(vec![
            GeometryBuffer {
                normal: Some(GeometryNormal::default()),
                uv0: Some(Default::default()),
                ..Default::default()
            },
            GeometryBuffer {
                compressed_attributes: Some(CompressedAttributes {
                    encoding: CompressedAttributesEncoding::Draco,
                    attributes: vec![CompressedAttributesAttributes::Position],
                }),
                ..Default::default()
            },
        ]);
        assert_eq!(select_geometry_buffer(&layer), Some((0, 1)));
    }

    #[test]
    fn picks_richest_uncompressed() {
        let layer = layer_with_buffers(vec![
            GeometryBuffer::default(),
            GeometryBuffer {
                normal: Some(Default::default()),
                uv0: Some(Default::default()),
                ..Default::default()
            },
        ]);
        assert_eq!(select_geometry_buffer(&layer), Some((0, 1)));
    }

    #[test]
    fn returns_none_for_no_triangle() {
        let layer = SceneLayerInfo {
            geometry_definitions: vec![GeometryDefinition {
                topology: Some(GeometryDefinitionTopology::Unknown),
                geometry_buffers: vec![GeometryBuffer::default()],
            }],
            ..Default::default()
        };
        assert_eq!(select_geometry_buffer(&layer), None);
    }
}

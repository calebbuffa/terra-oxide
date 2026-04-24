//! Typed extension wrappers for glTF extensions.
//!
//! Provides type-safe read/write access to extension data via the
//! [`HasExtensions`] trait - no downcasting or string-matching required.
//!
//! Built-in extensions ([`KhrDracoMeshCompression`], [`KhrTextureTransform`],
//! etc.) implement [`GltfExtension`], but any user-defined type that derives
//! `Serialize + Deserialize` and provides a `NAME` const can participate:
//!
//! ```ignore
//! #[derive(Serialize, Deserialize)]
//! struct MyExt { value: f32 }
//! impl GltfExtension for MyExt { const NAME: &'static str = "MY_custom_ext"; }
//!
//! node.set_extension(MyExt { value: 1.0 }).unwrap();
//! let ext: Option<MyExt> = node.get_extension();
//! ```

use crate::generated::*;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::collections::HashMap;

/// Trait for types that represent a named glTF extension.
///
/// Implementors must derive (or manually implement) [`Serialize`] and
/// [`Deserialize`]. The [`HasExtensions`] helpers use `serde_json` for
/// round-tripping, so no manual conversion code is needed.
pub trait GltfExtension: Sized + Serialize + DeserializeOwned {
    /// glTF extension name string (e.g. `"KHR_draco_mesh_compression"`).
    const NAME: &'static str;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrDracoMeshCompression {
    pub buffer_view: usize,
    pub attributes: HashMap<String, u32>,
}

impl GltfExtension for KhrDracoMeshCompression {
    const NAME: &'static str = "KHR_draco_mesh_compression";
}

/// Data bag for the `KHR_texture_transform` extension.
///
/// For UV-math operations (applying the transform to texture coordinates)
/// see [`crate::TextureTransform`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrTextureTransform {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rotation: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale: Option<[f32; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tex_coord: Option<u32>,
}

impl GltfExtension for KhrTextureTransform {
    const NAME: &'static str = "KHR_texture_transform";
}

/// Data bag for the `KHR_mesh_quantization` extension.
///
/// This extension has no additional JSON fields beyond its presence in
/// `extensionsUsed`/`extensionsRequired`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KhrMeshQuantization;

impl GltfExtension for KhrMeshQuantization {
    const NAME: &'static str = "KHR_mesh_quantization";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KhrLightsPunctual {
    pub light: usize,
}

impl GltfExtension for KhrLightsPunctual {
    const NAME: &'static str = "KHR_lights_punctual";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KhrMaterialsUnlit;

impl GltfExtension for KhrMaterialsUnlit {
    const NAME: &'static str = "KHR_materials_unlit";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtMeshGpuInstancing {
    pub attributes: HashMap<String, usize>,
}

impl GltfExtension for ExtMeshGpuInstancing {
    const NAME: &'static str = "EXT_mesh_gpu_instancing";
}

// ---------------------------------------------------------------------------
// KHR material extensions
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsClearcoat {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub clearcoat_factor: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub clearcoat_roughness_factor: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clearcoat_texture: Option<TextureInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clearcoat_roughness_texture: Option<TextureInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clearcoat_normal_texture: Option<MaterialNormalTextureInfo>,
}

impl Default for KhrMaterialsClearcoat {
    fn default() -> Self {
        Self {
            clearcoat_factor: 0.0,
            clearcoat_roughness_factor: 0.0,
            clearcoat_texture: None,
            clearcoat_roughness_texture: None,
            clearcoat_normal_texture: None,
        }
    }
}

impl GltfExtension for KhrMaterialsClearcoat {
    const NAME: &'static str = "KHR_materials_clearcoat";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsTransmission {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub transmission_factor: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transmission_texture: Option<TextureInfo>,
}

impl Default for KhrMaterialsTransmission {
    fn default() -> Self {
        Self {
            transmission_factor: 0.0,
            transmission_texture: None,
        }
    }
}

impl GltfExtension for KhrMaterialsTransmission {
    const NAME: &'static str = "KHR_materials_transmission";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsIor {
    #[serde(default = "default_ior")]
    pub ior: f32,
}

fn default_ior() -> f32 {
    1.5
}

impl Default for KhrMaterialsIor {
    fn default() -> Self {
        Self { ior: 1.5 }
    }
}

impl GltfExtension for KhrMaterialsIor {
    const NAME: &'static str = "KHR_materials_ior";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsSheen {
    #[serde(default = "default_sheen_color")]
    pub sheen_color_factor: [f32; 3],
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub sheen_roughness_factor: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sheen_color_texture: Option<TextureInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sheen_roughness_texture: Option<TextureInfo>,
}

fn default_sheen_color() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

impl Default for KhrMaterialsSheen {
    fn default() -> Self {
        Self {
            sheen_color_factor: [0.0, 0.0, 0.0],
            sheen_roughness_factor: 0.0,
            sheen_color_texture: None,
            sheen_roughness_texture: None,
        }
    }
}

impl GltfExtension for KhrMaterialsSheen {
    const NAME: &'static str = "KHR_materials_sheen";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsIridescence {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub iridescence_factor: f32,
    #[serde(default = "default_iridescence_ior")]
    pub iridescence_ior: f32,
    #[serde(default = "default_iridescence_thickness_min")]
    pub iridescence_thickness_minimum: f32,
    #[serde(default = "default_iridescence_thickness_max")]
    pub iridescence_thickness_maximum: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iridescence_texture: Option<TextureInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iridescence_thickness_texture: Option<TextureInfo>,
}

fn default_iridescence_ior() -> f32 {
    1.3
}
fn default_iridescence_thickness_min() -> f32 {
    100.0
}
fn default_iridescence_thickness_max() -> f32 {
    400.0
}

impl Default for KhrMaterialsIridescence {
    fn default() -> Self {
        Self {
            iridescence_factor: 0.0,
            iridescence_ior: 1.3,
            iridescence_thickness_minimum: 100.0,
            iridescence_thickness_maximum: 400.0,
            iridescence_texture: None,
            iridescence_thickness_texture: None,
        }
    }
}

impl GltfExtension for KhrMaterialsIridescence {
    const NAME: &'static str = "KHR_materials_iridescence";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsAnisotropy {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub anisotropy_strength: f32,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub anisotropy_rotation: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anisotropy_texture: Option<TextureInfo>,
}

impl Default for KhrMaterialsAnisotropy {
    fn default() -> Self {
        Self {
            anisotropy_strength: 0.0,
            anisotropy_rotation: 0.0,
            anisotropy_texture: None,
        }
    }
}

impl GltfExtension for KhrMaterialsAnisotropy {
    const NAME: &'static str = "KHR_materials_anisotropy";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterialVariant {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extensions: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extras: Option<serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsVariants {
    pub variants: Vec<MaterialVariant>,
}

impl GltfExtension for KhrMaterialsVariants {
    const NAME: &'static str = "KHR_materials_variants";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VariantMapping {
    pub material: usize,
    pub variants: Vec<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// `KHR_materials_variants` mappings extension on a `MeshPrimitive`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsVariantsMappings {
    pub mappings: Vec<VariantMapping>,
}

impl GltfExtension for KhrMaterialsVariantsMappings {
    const NAME: &'static str = "KHR_materials_variants";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsEmissiveStrength {
    #[serde(default = "default_emissive_strength")]
    pub emissive_strength: f32,
}

fn default_emissive_strength() -> f32 {
    1.0
}

impl Default for KhrMaterialsEmissiveStrength {
    fn default() -> Self {
        Self {
            emissive_strength: 1.0,
        }
    }
}

impl GltfExtension for KhrMaterialsEmissiveStrength {
    const NAME: &'static str = "KHR_materials_emissive_strength";
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KhrMaterialsVolume {
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    pub thickness_factor: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thickness_texture: Option<TextureInfo>,
    #[serde(default = "default_attenuation_distance")]
    pub attenuation_distance: f32,
    #[serde(default = "default_attenuation_color")]
    pub attenuation_color: [f32; 3],
}

fn default_attenuation_distance() -> f32 {
    f32::INFINITY
}
fn default_attenuation_color() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

impl Default for KhrMaterialsVolume {
    fn default() -> Self {
        Self {
            thickness_factor: 0.0,
            thickness_texture: None,
            attenuation_distance: f32::INFINITY,
            attenuation_color: [1.0, 1.0, 1.0],
        }
    }
}

impl GltfExtension for KhrMaterialsVolume {
    const NAME: &'static str = "KHR_materials_volume";
}

// ---------------------------------------------------------------------------
// EXT_instance_features
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstanceFeatureId {
    pub feature_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub attribute: Option<u32>,
    #[serde(rename = "propertyTable", skip_serializing_if = "Option::is_none")]
    pub property_table: Option<u32>,
    #[serde(rename = "nullFeatureId", skip_serializing_if = "Option::is_none")]
    pub null_feature_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtInstanceFeatures {
    #[serde(rename = "featureIds")]
    pub feature_ids: Vec<InstanceFeatureId>,
}

impl GltfExtension for ExtInstanceFeatures {
    const NAME: &'static str = "EXT_instance_features";
}

fn is_zero_f32(v: &f32) -> bool {
    *v == 0.0
}

/// Typed read/write access to extension data on any glTF object that carries
/// an `extensions: HashMap<String, serde_json::Value>` field.
///
/// Implement this for your own types by providing `extensions_map` /
/// `extensions_map_mut`. All glTF model objects generated by `moderu` already
/// implement this trait.
pub trait HasExtensions {
    fn extensions_map(&self) -> &HashMap<String, serde_json::Value>;
    fn extensions_map_mut(&mut self) -> &mut HashMap<String, serde_json::Value>;

    /// Deserialize an extension value by type. Returns `None` if the extension
    /// is absent or its JSON cannot be deserialized as `E`.
    fn get_extension<E: GltfExtension>(&self) -> Option<E> {
        self.extensions_map()
            .get(E::NAME)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Returns `true` if the extension key is present (regardless of whether
    /// the value can be deserialized as any particular type).
    fn has_extension<E: GltfExtension>(&self) -> bool {
        self.extensions_map().contains_key(E::NAME)
    }

    /// Serialize `ext` and insert it under `E::NAME`. Overwrites any existing
    /// value for that key.
    ///
    /// # Errors
    /// Returns `Err` if `serde_json::to_value` fails for `ext`.
    fn set_extension<E: GltfExtension>(&mut self, ext: E) -> Result<(), serde_json::Error> {
        let v = serde_json::to_value(ext)?;
        self.extensions_map_mut().insert(E::NAME.to_string(), v);
        Ok(())
    }

    /// Remove the extension entry for `E`. Returns `true` if a value was present.
    fn remove_extension<E: GltfExtension>(&mut self) -> bool {
        self.extensions_map_mut().remove(E::NAME).is_some()
    }
}

macro_rules! impl_has_extensions {
    ($($T:ty),+ $(,)?) => {
        $(
            impl HasExtensions for $T {
                #[inline]
                fn extensions_map(&self) -> &HashMap<String, serde_json::Value> {
                    &self.extensions
                }
                #[inline]
                fn extensions_map_mut(&mut self) -> &mut HashMap<String, serde_json::Value> {
                    &mut self.extensions
                }
            }
        )+
    };
}

impl_has_extensions!(
    Accessor,
    AccessorSparse,
    AccessorSparseIndices,
    AccessorSparseValues,
    Animation,
    AnimationChannel,
    AnimationChannelTarget,
    AnimationSampler,
    Asset,
    Buffer,
    BufferView,
    Camera,
    CameraOrthographic,
    CameraPerspective,
    GltfModel,
    Image,
    Material,
    MaterialNormalTextureInfo,
    MaterialOcclusionTextureInfo,
    MaterialPbrMetallicRoughness,
    Mesh,
    MeshPrimitive,
    Node,
    Sampler,
    Scene,
    Skin,
    Texture,
    TextureInfo,
);

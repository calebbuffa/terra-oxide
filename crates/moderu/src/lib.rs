//! # moderu - glTF 2.0 model toolkit
//!
//! Parse, inspect, build, and transform [glTF 2.0](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
//! models in memory.
//!
//! * **Generated types** - auto-generated Rust structs for the full glTF JSON schema (re-exported
//!   from [`generated`]).
//! * **Accessor utilities** - type-safe views and iterators over buffer data
//!   ([`resolve_accessor`], [`AccessorView`], …).
//! * **Builder** - construct models programmatically via [`GltfModelBuilder`].
//! * **Scene graph** - traverse node hierarchies and compute world transforms
//!   ([`SceneGraph`], [`TransformCache`]).
//! * **Metadata** - access `EXT_structural_metadata` property tables, textures,
//!   and attributes.
//! * **Extensions** - typed extension wrappers ([`extensions`]).

mod generated;

mod accessor;
mod builder;
pub mod codec;
mod compaction;
mod copyright;
pub mod extensions;
mod geometry;
mod image;
pub mod io;
mod merge;
mod metadata;
mod property;
pub mod reader;
mod sampler;
mod scene;
mod semantics;
mod skirt;
mod texture;
pub mod writer;

pub use generated::*;

pub use accessor::{
    AccessorIter, AccessorView, AccessorViewError, AccessorWriter, ComponentType, resolve_accessor,
    resolve_accessor_mut, resolve_accessor_owned,
};

pub use builder::{
    AccessorIndex, BufferViewIndex, FlatData, GltfData, GltfModelBuilder, IndexData, MaterialIndex,
    MeshBuilder, MeshIndex, NodeBuilder, NodeIndex, PrimitiveBuilder, UpAxis,
};

pub use geometry::{TangentError, flat_normals, generate_tangents};

// extensions - individual types accessible via moderu::extensions::*
pub use extensions::{
    ExtInstanceFeatures, ExtMeshGpuInstancing, GltfExtension, HasExtensions, InstanceFeatureId,
    KhrDracoMeshCompression, KhrLightsPunctual, KhrMaterialsAnisotropy, KhrMaterialsClearcoat,
    KhrMaterialsEmissiveStrength, KhrMaterialsIor, KhrMaterialsIridescence, KhrMaterialsSheen,
    KhrMaterialsTransmission, KhrMaterialsUnlit, KhrMaterialsVariants,
    KhrMaterialsVariantsMappings, KhrMaterialsVolume, KhrMeshQuantization, KhrTextureTransform,
    MaterialVariant, VariantMapping,
};

pub use image::{
    BlitError, GpuCompressedPixelFormat, ImageData, Ktx2TranscodeTargets, MipPosition, Rectangle,
    SupportedGpuCompressedPixelFormats,
};

pub use metadata::{
    Class, ClassProperty, EXT_STRUCTURAL_METADATA, EnumValue, ExtStructuralMetadata, FeatureId,
    MetadataError, MetadataViewError, PropertyAttribute, PropertyAttributeIter,
    PropertyAttributeProperty, PropertyAttributePropertyView, PropertyAttributeView, PropertyTable,
    PropertyTableIter, PropertyTableProperty, PropertyTablePropertyView, PropertyTableView,
    PropertyTexture, PropertyTextureProperty, PropertyTexturePropertyView, PropertyTextureView,
    Schema, SchemaEnum, resolve_schema_uri,
};

pub use property::{
    IntoF64, MetadataConvert, MetadataValue, PropertyArrayCopy, PropertyArrayIter,
    PropertyArrayView, PropertyComponentType, PropertyElement, PropertyMat2, PropertyMat3,
    PropertyMat4, PropertyType, PropertyViewError, TransformProperty, VariablePropertyArrayView,
};

pub use sampler::{FilterMode, WrapMode};

pub use scene::{
    NodeTransform, SceneError, SceneGraph, SceneNode, SceneNodeIterator, SceneRootIterator,
    Transform, TransformCache, TransformSOA,
};

pub use semantics::{InstanceAttribute, VertexAttribute};

pub use texture::{
    FeatureIdTextureView, TextureTransform, TextureView, TextureViewError, TextureViewOptions,
};

pub use io::GltfParseError;

pub use skirt::SkirtMeshMetadata;

// Top-level re-exports for convenience.
pub use reader::{GltfError, GltfReader, GltfReaderOptions};
pub use writer::{GltfWriter, GltfWriterOptions};

//! [`GltfModelBuilder`] - ergonomic constructor for [`GltfModel`].
//!
//! Handles the bookkeeping of buffers, buffer views, and accessors so callers
//! can focus on the data rather than index wiring.
//!
//! # Example: Draco i3S -> glTF
//! ```ignore
//! use moderu::{AccessorType, GltfModelBuilder};
//!
//! let mut b = GltfModelBuilder::new();
//!
//! let pos  = b.add_accessor(&positions_vec3);
//! let norm = b.add_accessor(&normals_vec3);
//! let idxs = b.add_indices(&indices_u32);
//!
//! let prim = b.primitive()
//!     .indices(idxs)
//!     .attribute("POSITION", pos)
//!     .attribute("NORMAL",   norm)
//!     .build();
//!
//! b.mesh().primitive(prim).build();
//! let model = b.finish();
//! ```

use std::collections::HashMap;

use crate::{
    Accessor, AccessorComponentType, AccessorType, Asset, Buffer, BufferView, GltfModel, Material,
    MaterialPbrMetallicRoughness, Mesh, MeshPrimitive, Node, PrimitiveMode, Scene,
};

mod private {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
    impl Sealed for u16 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for i8 {}
    impl Sealed for [f32; 2] {}
    impl Sealed for [f32; 3] {}
    impl Sealed for [f32; 4] {}
    impl Sealed for [f32; 9] {}
    impl Sealed for [f32; 16] {}
    impl Sealed for [u8; 3] {}
    impl Sealed for [u8; 4] {}
    impl Sealed for [u16; 4] {}
    impl Sealed for [u32; 4] {}
    impl Sealed for [i8; 4] {}
    impl Sealed for [i16; 4] {}

    /// Sub-seal for integer types valid as glTF index data.
    pub trait SealedIndex: Sealed {}
    impl SealedIndex for u8 {}
    impl SealedIndex for u16 {}
    impl SealedIndex for u32 {}
}

/// Marker trait that associates a Rust type with its glTF accessor component type
/// and accessor shape (SCALAR, VEC2, VEC3, …).
///
/// Sealed - cannot be implemented outside this crate.
pub trait GltfData: bytemuck::Pod + private::Sealed {
    const COMPONENT_TYPE: AccessorComponentType;
    const ACCESSOR_TYPE: AccessorType;
}

/// Marker trait for types valid as glTF index data: `u8`, `u16`, or `u32`.
///
/// Restricts [`GltfModelBuilder::push_indices`] to index-appropriate types,
/// making it a compile error to accidentally pass vertex attribute data.
///
/// Sealed - cannot be implemented outside this crate.
pub trait IndexData: GltfData + private::SealedIndex {}
impl IndexData for u8 {}
impl IndexData for u16 {}
impl IndexData for u32 {}

/// Marker trait for array glTF data types that can be constructed from a flat
/// slice of their scalar [`Element`](FlatData::Element) type.
///
/// Used by [`GltfModelBuilder::add_flat`] to reinterpret a flat buffer (e.g.
/// `&[f32]`) as a typed slice (e.g. `&[[f32; 3]]`) without copying.
///
/// Sealed - cannot be implemented outside this crate.
pub trait FlatData: GltfData {
    /// The scalar element type (e.g. `f32` for `[f32; 3]`).
    type Element: bytemuck::NoUninit;
}

impl FlatData for [f32; 2] {
    type Element = f32;
}
impl FlatData for [f32; 3] {
    type Element = f32;
}
impl FlatData for [f32; 4] {
    type Element = f32;
}
impl FlatData for [f32; 9] {
    type Element = f32;
}
impl FlatData for [f32; 16] {
    type Element = f32;
}
impl FlatData for [u8; 3] {
    type Element = u8;
}
impl FlatData for [u8; 4] {
    type Element = u8;
}
impl FlatData for [u16; 4] {
    type Element = u16;
}
impl FlatData for [u32; 4] {
    type Element = u32;
}
impl FlatData for [i8; 4] {
    type Element = i8;
}
impl FlatData for [i16; 4] {
    type Element = i16;
}

impl GltfData for f32 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}
impl GltfData for u32 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedInt;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}
impl GltfData for u16 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedShort;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}
impl GltfData for u8 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedByte;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}
impl GltfData for i16 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Short;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}
impl GltfData for i8 {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Byte;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Scalar;
}

impl GltfData for [f32; 2] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec2;
}

impl GltfData for [f32; 3] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec3;
}

impl GltfData for [f32; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}
impl GltfData for [u8; 3] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedByte;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec3;
}
impl GltfData for [u8; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedByte;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}
impl GltfData for [u16; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedShort;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}
impl GltfData for [u32; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::UnsignedInt;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}
impl GltfData for [i8; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Byte;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}
impl GltfData for [i16; 4] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Short;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Vec4;
}

impl GltfData for [f32; 9] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Mat3;
}
impl GltfData for [f32; 16] {
    const COMPONENT_TYPE: AccessorComponentType = AccessorComponentType::Float;
    const ACCESSOR_TYPE: AccessorType = AccessorType::Mat4;
}

/// Typed index into `GltfModel::accessors`.
///
/// Returned by [`GltfModelBuilder::push_accessor`] and [`GltfModelBuilder::push_indices`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct AccessorIndex(pub usize);

/// Typed index into `GltfModel::buffer_views`.
///
/// Returned by [`GltfModelBuilder::push_raw`] and [`GltfModelBuilder::push_raw_strided`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct BufferViewIndex(pub usize);

/// Typed index into `GltfModel::meshes`.
///
/// Returned by [`GltfModelBuilder::push_mesh`] and [`MeshBuilder::build`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MeshIndex(pub usize);

/// Typed index into `GltfModel::materials`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct MaterialIndex(pub usize);

/// Typed index into `GltfModel::nodes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct NodeIndex(pub usize);

macro_rules! impl_index_conversions {
    ($($T:ident),*) => { $(
        impl From<$T> for usize {
            fn from(idx: $T) -> Self { idx.0 }
        }
        impl From<usize> for $T {
            fn from(val: usize) -> Self { $T(val) }
        }
        impl std::fmt::Display for $T {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    )* };
}

impl_index_conversions!(
    AccessorIndex,
    BufferViewIndex,
    MeshIndex,
    MaterialIndex,
    NodeIndex
);

/// Up-axis convention encoded in the glTF `extras["gltfUpAxis"]` field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum UpAxis {
    X = 0,
    Y = 1,
    Z = 2,
}

/// Ergonomic builder for [`GltfModel`].
///
/// All binary data is accumulated into a single internal buffer.
/// Call [`finish`](GltfModelBuilder::finish) to get the completed model.
///
/// On [`finish`](GltfModelBuilder::finish), if no scene has been created but
/// meshes and nodes exist, a default scene referencing all root nodes is
/// generated automatically.
#[derive(Debug)]
pub struct GltfModelBuilder {
    model: GltfModel,
    /// Index of the shared buffer all data is appended to.
    buf_idx: usize,
}

impl GltfModelBuilder {
    /// Create a new builder with a single shared buffer and glTF 2.0 asset metadata.
    pub fn new() -> Self {
        let mut model = GltfModel {
            asset: Asset {
                version: "2.0".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        model.buffers.push(Buffer::default());
        Self { model, buf_idx: 0 }
    }

    /// Set the `asset.generator` string (e.g. `"MyApp v1.0"`).
    pub fn asset_generator(&mut self, generator: &str) -> &mut Self {
        self.model.asset.generator = Some(generator.into());
        self
    }

    /// Append `bytes` to the shared buffer and register a buffer view over them.
    /// Returns the buffer view index.
    pub fn add_buffer_view(&mut self, bytes: &[u8]) -> BufferViewIndex {
        let offset = self.model.buffers[self.buf_idx].data.len();
        self.model.buffers[self.buf_idx]
            .data
            .extend_from_slice(bytes);
        let bv_idx = self.model.buffer_views.len();
        self.model.buffer_views.push(BufferView {
            buffer: self.buf_idx,
            byte_offset: offset,
            byte_length: bytes.len(),
            ..Default::default()
        });
        BufferViewIndex(bv_idx)
    }

    /// Like [`push_raw`] but records an explicit `byteStride` for interleaved layouts.
    /// Only use this when multiple accessors share the same buffer view.
    pub fn add_buffer_view_strided(&mut self, bytes: &[u8], byte_stride: usize) -> BufferViewIndex {
        let offset = self.model.buffers[self.buf_idx].data.len();
        self.model.buffers[self.buf_idx]
            .data
            .extend_from_slice(bytes);
        let bv_idx = self.model.buffer_views.len();
        self.model.buffer_views.push(BufferView {
            buffer: self.buf_idx,
            byte_offset: offset,
            byte_length: bytes.len(),
            byte_stride: Some(byte_stride),
            ..Default::default()
        });
        BufferViewIndex(bv_idx)
    }

    /// Push a typed attribute array and create an accessor for it.
    ///
    /// Both the glTF component type and accessor shape are derived from `T`
    /// via [`GltfData`] - no extra parameters needed.
    ///
    /// ```ignore
    /// let pos = b.add_accessor(&positions_vec3);
    /// let uvs = b.add_accessor(&uvs_vec2);
    /// ```
    pub fn add_accessor<T: GltfData>(&mut self, data: &[T]) -> AccessorIndex {
        let bytes = bytemuck::cast_slice(data);
        let bv = self.add_buffer_view(bytes);
        let acc_idx = self.model.accessors.len();
        self.model.accessors.push(Accessor {
            buffer_view: Some(bv.0),
            component_type: T::COMPONENT_TYPE,
            count: data.len(),
            r#type: T::ACCESSOR_TYPE,
            ..Default::default()
        });
        AccessorIndex(acc_idx)
    }

    /// Push a triangle index buffer and create a `Scalar` accessor for it.
    ///
    /// `T` must be an index-compatible type: `u8`, `u16`, or `u32`.
    /// Passing a vertex attribute type (e.g. `[f32; 3]`) is a compile error.
    ///
    /// ```ignore
    /// let idxs = b.add_indices(&indices_u32);
    /// ```
    pub fn add_indices<T: IndexData>(&mut self, indices: &[T]) -> AccessorIndex {
        let bytes = bytemuck::cast_slice(indices);
        let bv = self.add_buffer_view(bytes);
        let acc_idx = self.model.accessors.len();
        self.model.accessors.push(Accessor {
            buffer_view: Some(bv.0),
            component_type: T::COMPONENT_TYPE,
            count: indices.len(),
            r#type: AccessorType::Scalar,
            ..Default::default()
        });
        AccessorIndex(acc_idx)
    }

    /// Push a `u32` triangle index buffer, automatically **downgrading to `u16`**
    /// if every index value fits within `u16::MAX` (65 535).
    ///
    /// This saves 50 % of index buffer memory for meshes with <= 65 535 vertices
    /// without requiring the caller to branch and allocate a second collection.
    ///
    /// ```ignore
    /// // Works for both small and large meshes - no manual u16/u32 branch needed.
    /// let idx = b.add_indices_compact(&indices_u32);
    /// ```
    pub fn add_indices_compact(&mut self, indices: &[u32]) -> AccessorIndex {
        if indices.iter().all(|&i| i <= u16::MAX as u32) {
            let idx16: Vec<u16> = indices.iter().map(|&i| i as u16).collect();
            self.add_indices(idx16.as_slice())
        } else {
            self.add_indices(indices)
        }
    }

    /// Push a flat scalar slice as a typed vertex attribute by reinterpreting
    /// it as a slice of the target array type.
    ///
    /// `T` must implement [`FlatData`], associating the array type with its
    /// element type.  No allocation is performed - the data is cast in place.
    ///
    /// ```ignore
    /// // Type-safe replacement for:
    /// //   b.add_accessor(bytemuck::cast_slice::<f32, [f32; 3]>(&flat_f32s))
    /// let acc = b.add_flat::<[f32; 3]>(&flat_f32s);
    /// ```
    pub fn add_flat<T: FlatData>(&mut self, data: &[T::Element]) -> AccessorIndex {
        self.add_accessor(bytemuck::cast_slice::<T::Element, T>(data))
    }

    /// Push a flat `f32` slice as a vertex attribute, grouping values by
    /// `components` at runtime.
    ///
    /// Prefer the type-safe [`add_flat`](Self::add_flat) when the component
    /// count is known at compile time.  This method exists for callers that
    /// read the component count from data (e.g. I3S geometry descriptors).
    ///
    /// | `components` | accessor type  |
    /// |---|---|
    /// | 1 | `SCALAR f32`    |
    /// | 2 | `VEC2  [f32;2]` |
    /// | 3 | `VEC3  [f32;3]` |
    /// | 4 | `VEC4  [f32;4]` |
    ///
    /// Panics if `components` is 0 or greater than 4.
    pub fn add_floats_as_attribute(&mut self, data: &[f32], components: usize) -> AccessorIndex {
        match components {
            1 => self.add_accessor(data),
            2 => self.add_flat::<[f32; 2]>(data),
            3 => self.add_flat::<[f32; 3]>(data),
            4 => self.add_flat::<[f32; 4]>(data),
            n => panic!("add_floats_as_attribute: unsupported component count {n}"),
        }
    }

    /// Push a [`Material`] and return its index.
    pub fn add_material(&mut self, material: Material) -> MaterialIndex {
        let idx = self.model.materials.len();
        self.model.materials.push(material);
        MaterialIndex(idx)
    }

    /// Push a default PBR material with the given RGBA base color (each in 0..1).
    ///
    /// Metallic factor is 0 and roughness is 1 (fully dielectric/rough),
    pub fn add_default_material(&mut self, base_color: [f64; 4]) -> MaterialIndex {
        self.add_material(Material {
            pbr_metallic_roughness: Some(MaterialPbrMetallicRoughness {
                base_color_factor: base_color.to_vec(),
                metallic_factor: 0.0,
                roughness_factor: 1.0,
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    /// Start building a node. Call `.build()` to push it into the model.
    ///
    /// ```ignore
    /// let node = b.node().mesh(mesh).build();
    /// let node = b.node().mesh(mesh).matrix(&mat).build();
    /// let node = b.node().mesh(mesh).name("terrain").children(&[child]).build();
    /// ```
    pub fn node(&mut self) -> NodeBuilder<'_> {
        NodeBuilder {
            builder: self,
            mesh: None,
            name: None,
            matrix: None,
            translation: None,
            rotation: None,
            scale: None,
            children: Vec::new(),
        }
    }

    /// Set the `gltfUpAxis` extras value.
    ///
    /// Use [`UpAxis::X`], [`UpAxis::Y`], or [`UpAxis::Z`] instead of raw integers.
    pub fn up_axis(&mut self, axis: UpAxis) -> &mut Self {
        let extras = self
            .model
            .extras
            .get_or_insert_with(|| serde_json::json!({}));
        extras["gltfUpAxis"] = serde_json::json!(axis as u8);
        self
    }

    /// Start building a [`MeshPrimitive`].
    pub fn primitive(&self) -> PrimitiveBuilder {
        PrimitiveBuilder::new()
    }

    /// Start building a [`Mesh`] that is pushed into the model on
    /// [`MeshBuilder::build`], which returns the mesh index.
    pub fn mesh(&mut self) -> MeshBuilder<'_> {
        MeshBuilder {
            builder: self,
            name: None,
            primitives: Vec::new(),
        }
    }

    /// Push a pre-built [`MeshPrimitive`] as a single-primitive mesh.
    /// Returns the mesh index.
    pub fn add_mesh(&mut self, primitive: MeshPrimitive) -> MeshIndex {
        let mesh_idx = self.model.meshes.len();
        self.model.meshes.push(Mesh {
            primitives: vec![primitive],
            ..Default::default()
        });
        MeshIndex(mesh_idx)
    }

    /// Finalise and return the [`GltfModel`].
    ///
    /// If meshes exist but no scene has been created, a default scene
    /// referencing all **root** nodes (nodes not referenced as any other
    /// node's child) is generated automatically. If meshes exist but no
    /// nodes, one node per mesh is created first.
    #[must_use]
    pub fn finish(mut self) -> GltfModel {
        let len = self.model.buffers[self.buf_idx].data.len();
        self.model.buffers[self.buf_idx].byte_length = len;

        // Auto-create nodes for meshes that have no node yet.
        if !self.model.meshes.is_empty() && self.model.nodes.is_empty() {
            for i in 0..self.model.meshes.len() {
                self.model.nodes.push(Node {
                    mesh: Some(i),
                    ..Default::default()
                });
            }
        }

        // Auto-create a default scene if none exists.
        if self.model.scenes.is_empty() && !self.model.nodes.is_empty() {
            // Collect all node indices that appear as children of another node.
            let mut child_indices: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for node in &self.model.nodes {
                if let Some(children) = &node.children {
                    for &c in children {
                        child_indices.insert(c);
                    }
                }
            }
            // Scene roots = nodes that are not anyone's child.
            let root_indices: Vec<usize> = (0..self.model.nodes.len())
                .filter(|i| !child_indices.contains(i))
                .collect();
            self.model.scenes.push(Scene {
                nodes: Some(root_indices),
                ..Default::default()
            });
            self.model.scene = Some(0);
        }

        self.model
    }

    /// Borrow the model being built (e.g. to inspect indices mid-build).
    pub fn model(&self) -> &GltfModel {
        &self.model
    }

    /// Mutably borrow the model being built (e.g. to set extras on meshes).
    pub fn model_mut(&mut self) -> &mut GltfModel {
        &mut self.model
    }
}

impl Default for GltfModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for a single [`MeshPrimitive`].
pub struct PrimitiveBuilder {
    indices: Option<usize>,
    attributes: HashMap<String, usize>,
    mode: PrimitiveMode,
    material: Option<usize>,
    extras: Option<serde_json::Value>,
}

impl PrimitiveBuilder {
    fn new() -> Self {
        Self {
            indices: None,
            attributes: HashMap::new(),
            mode: PrimitiveMode::Triangles,
            material: None,
            extras: None,
        }
    }

    /// Set the index accessor.
    pub fn indices(mut self, acc: AccessorIndex) -> Self {
        self.indices = Some(acc.0);
        self
    }

    /// Add a vertex attribute (e.g. `"POSITION"`, `"NORMAL"`, `"TEXCOORD_0"`).
    pub fn attribute(mut self, semantic: impl Into<String>, acc: AccessorIndex) -> Self {
        self.attributes.insert(semantic.into(), acc.0);
        self
    }

    /// Set the primitive topology (default: `Triangles`).
    pub fn mode(mut self, mode: PrimitiveMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the material index.
    pub fn material(mut self, mat_idx: MaterialIndex) -> Self {
        self.material = Some(mat_idx.0);
        self
    }

    /// Set application-specific extras on the primitive.
    pub fn extras(mut self, extras: serde_json::Value) -> Self {
        self.extras = Some(extras);
        self
    }

    /// Consume the builder and produce a [`MeshPrimitive`].
    pub fn build(self) -> MeshPrimitive {
        MeshPrimitive {
            indices: self.indices,
            attributes: self.attributes,
            mode: self.mode,
            material: self.material,
            extras: self.extras,
            ..Default::default()
        }
    }
}

/// Builder for a [`Mesh`] that is pushed into the model on [`build`](Self::build).
pub struct MeshBuilder<'a> {
    builder: &'a mut GltfModelBuilder,
    name: Option<String>,
    primitives: Vec<MeshPrimitive>,
}

impl<'a> MeshBuilder<'a> {
    /// Add a primitive.
    pub fn primitive(mut self, prim: MeshPrimitive) -> Self {
        self.primitives.push(prim);
        self
    }

    /// Set the mesh name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Push the mesh into the model and return its index.
    pub fn build(self) -> MeshIndex {
        let mesh_idx = self.builder.model.meshes.len();
        self.builder.model.meshes.push(Mesh {
            primitives: self.primitives,
            name: self.name,
            ..Default::default()
        });
        MeshIndex(mesh_idx)
    }
}

/// Builder for a glTF [`Node`].
///
/// Obtained from [`GltfModelBuilder::node`]. Call [`build`](NodeBuilder::build) to push
/// the node into the model and get its index back.
pub struct NodeBuilder<'a> {
    builder: &'a mut GltfModelBuilder,
    mesh: Option<MeshIndex>,
    name: Option<String>,
    matrix: Option<[f64; 16]>,
    translation: Option<[f64; 3]>,
    rotation: Option<[f64; 4]>,
    scale: Option<[f64; 3]>,
    children: Vec<NodeIndex>,
}

impl<'a> NodeBuilder<'a> {
    /// Attach a mesh to this node.
    pub fn mesh(mut self, mesh: MeshIndex) -> Self {
        self.mesh = Some(mesh);
        self
    }

    /// Set the node name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a column-major 4x4 world-from-local transform matrix.
    /// Clears any previously set TRS values.
    pub fn matrix(mut self, m: &[f64; 16]) -> Self {
        self.matrix = Some(*m);
        self.translation = None;
        self.rotation = None;
        self.scale = None;
        self
    }

    /// Set the transform from a [`glam::DMat4`], stored as a column-major
    /// `[f64; 16]` array. Clears any previously set TRS values.
    ///
    /// Equivalent to `.matrix(&m.to_cols_array())`.
    pub fn matrix_dmat4(self, m: glam::DMat4) -> Self {
        self.matrix(&m.to_cols_array())
    }

    /// Set a translation vector `[x, y, z]`. Clears any matrix.
    pub fn translation(mut self, t: [f64; 3]) -> Self {
        self.translation = Some(t);
        self.matrix = None;
        self
    }

    /// Set a rotation quaternion `[x, y, z, w]`. Clears any matrix.
    pub fn rotation(mut self, r: [f64; 4]) -> Self {
        self.rotation = Some(r);
        self.matrix = None;
        self
    }

    /// Set a scale vector `[x, y, z]`. Clears any matrix.
    pub fn scale(mut self, s: [f64; 3]) -> Self {
        self.scale = Some(s);
        self.matrix = None;
        self
    }

    /// Add child node indices.
    pub fn children(mut self, children: &[NodeIndex]) -> Self {
        self.children.extend_from_slice(children);
        self
    }

    /// Push the node into the model and return its index.
    pub fn build(self) -> NodeIndex {
        let idx = self.builder.model.nodes.len();
        let children = if self.children.is_empty() {
            None
        } else {
            Some(self.children.into_iter().map(|n| n.0).collect())
        };
        self.builder.model.nodes.push(Node {
            mesh: self.mesh.map(|m| m.0),
            name: self.name,
            matrix: self.matrix.map(|m| m.to_vec()).unwrap_or_default(),
            translation: self.translation.map(|t| t.to_vec()).unwrap_or_default(),
            rotation: self.rotation.map(|r| r.to_vec()).unwrap_or_default(),
            scale: self.scale.map(|s| s.to_vec()).unwrap_or_default(),
            children,
            ..Default::default()
        });
        NodeIndex(idx)
    }
}

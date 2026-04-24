//! SoA tile tree - the tile store for `kiban`.
//!
//! [`TileStore`] uses Structure-of-Arrays layout so the selection traversal
//! touches only the hot arrays (bounding volumes + geometric errors) while
//! rarely-accessed fields (content keys, world transforms, globe rectangles)
//! live in separate allocations and stay out of cache during traversal.
//!
//! [`TileId`] is a `NonZeroU32`-backed index.  The value stored inside is
//! `slot + 1`, so slot 0 has `TileId(1)`, slot 1 has `TileId(2)`, etc.
//! `Option<TileId>` is therefore the same width as `u32` thanks to the
//! niche optimisation.
//!
//! # Children storage
//!
//! Children for all nodes share a single flat `children_buf`.  Each tile holds
//! a `(start, len)` pair indexing into that buffer.  For leaf nodes `len == 0`.
//! This keeps the child list cache-local and avoids per-tile heap allocations.

use std::num::NonZeroU32;

use glam::DMat4;
use sovra::OverlayHierarchy;
use terra::GlobeRectangle;
use zukei::SpatialBounds;

/// A dense index into [`TileStore`]'s parallel arrays.
///
/// Backed by `NonZeroU32` so `Option<NodeIndex>` is 4 bytes.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TileId(pub NonZeroU32);

impl TileId {
    /// Construct from a 0-based slot.
    #[inline]
    pub fn from_slot(slot: u32) -> Self {
        // Safety: slot + 1 is always nonzero.
        Self(unsafe { NonZeroU32::new_unchecked(slot + 1) })
    }

    /// Return the 0-based slot (direct `Vec` index).
    #[inline]
    pub fn slot(self) -> usize {
        (self.0.get() - 1) as usize
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RefinementMode {
    /// Children add to the parent's geometry (ADD).
    Add,
    /// Children replace the parent's geometry (REPLACE).
    Replace,
}

/// Open struct describing what kind of content a tile carries.
///
/// Using a struct with boolean fields instead of an enum makes it possible to
/// add new tile categories (e.g. point clouds, voxels, proprietary formats)
/// without modifying kiban: just define a new constant or set the relevant
/// fields directly.
///
/// # Built-in constants
///
/// | Constant             | `has_content` | `is_external` | Meaning                                 |
/// |----------------------|:-------------:|:-------------:|----------------------------------------|
/// | `TileKind::CONTENT`  |     `true`    |    `false`    | Tile has renderable geometry.           |
/// | `TileKind::EMPTY`    |    `false`    |    `false`    | Placeholder tile with no content.       |
/// | `TileKind::EXTERNAL` |    `false`    |     `true`    | Points to an external layer to be resolved.   |
///
/// # Extension
///
/// ```rust
/// use kiban::TileKind;
/// // A tile kind that has both renderable content AND references an external subtree:
/// pub const HYBRID: TileKind = TileKind { has_content: true, is_external: true };
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct TileKind {
    /// The tile carries renderable content that can be loaded and displayed.
    pub has_content: bool,
    /// The tile points to an external `tileset.json` subtree that must be
    /// fetched before children can be known.
    pub is_external: bool,
}

impl TileKind {
    /// Tile with geometry content.
    pub const CONTENT: Self = Self {
        has_content: true,
        is_external: false,
    };
    /// Empty placeholder tile - no content, only structural children.
    pub const EMPTY: Self = Self {
        has_content: false,
        is_external: false,
    };
    /// Tile that points to an external layer to be resolved.
    pub const EXTERNAL: Self = Self {
        has_content: false,
        is_external: true,
    };
}

bitflags::bitflags! {
    #[derive(Clone, Copy, Debug, Default)]
    pub struct TileFlags: u8 {
        /// The selection algorithm should call `expand` if the tile has no
        /// recorded children - they might be discoverable (implicit tiling,
        /// external tileset).
        const MIGHT_HAVE_LATENT_CHILDREN = 0b0000_0001;
        /// Always refine; never render this tile itself.
        const UNCONDITIONALLY_REFINED    = 0b0000_0010;
        /// All direct children's bounding volumes are fully contained within
        /// this tile's bounding volume (conservative sphere-in-sphere test).
        ///
        /// When set, the traversal can skip individual child frustum tests:
        /// if the parent is visible the children are guaranteed visible too.
        ///
        /// Matches CesiumJS `_optimChildrenWithinParent` hint.
        const CHILDREN_WITHIN_PARENT     = 0b0000_0100;
    }
}

/// Identifies which `Box<dyn ContentLoader>` in `ContentManager::loaders`
/// services a given tile.
///
/// Most nodes inherit the loader from their parent.  Only subtree roots
/// (implicit tiling) and external tileset roots get an explicit `LoaderIndex`.
/// Analogous to cesium-native's per-`Tile` `TilesetContentLoader*`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct LoaderIndex(pub u32);

impl LoaderIndex {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Identifies tile content: either a network URL, an implicit tile address,
/// or a user-defined addressing scheme.
///
/// # Extension
///
/// The `Custom` variant provides an escape hatch for addressing schemes not
/// covered by the built-in variants (e.g. CDB tile coordinates,
/// proprietary stream keys).  Downcast the `Arc<dyn Any + Send + Sync>` to
/// your concrete key type inside your [`ContentLoader`] implementation.
///
/// Existing loaders' `_ => TileLoadResult::failed()` fallbacks handle
/// `Custom` keys transparently - they simply decline to serve them and let
/// your loader take over via `LoaderIndex`.
///
/// [`ContentLoader`]: crate::ContentLoader
#[derive(Clone, Debug)]
pub enum ContentKey {
    Uri(String),
    /// Implicit quadtree tile: (level, x, y).
    Quadtree(u32, u32, u32),
    /// Implicit octree tile: (level, x, y, z).
    Octree(u32, u32, u32, u32),
    /// S2 implicit tile — raw 64-bit S2 cell ID.
    S2(u64),
    /// Escape hatch for user-defined tile addressing schemes.
    ///
    /// Downcast with `Arc::downcast` to your concrete key type inside your
    /// `ContentLoader::load_tile` implementation.
    Custom(std::sync::Arc<dyn std::any::Any + Send + Sync>),
}

/// Data needed to insert a new tile into [`TileStore`].
///
/// Used by loaders when expanding the tree (initial construction or latent
/// child expansion).
#[derive(Clone, Debug)]
pub struct TileDescriptor {
    /// Bounding volume in world space.
    pub bounds: SpatialBounds,
    /// Geometric error in metres (3D Tiles `geometricError`).
    pub geometric_error: f64,
    /// Refinement mode inherited or specified in the tileset.
    pub refinement: RefinementMode,
    pub kind: TileKind,
    pub flags: TileFlags,
    /// Content keys.  Empty for pure interior nodes.
    pub content_keys: Vec<ContentKey>,
    /// World-space transform accumulated from all ancestor transforms.
    pub world_transform: DMat4,
    /// Child descriptors inlined for bulk construction.
    ///
    /// For latent-expansion nodes the vec is empty; children are inserted
    /// later when the subtree file arrives.
    pub children: Vec<TileDescriptor>,
    /// Optional tighter bounding volume for the content mesh only.
    pub content_bounds: Option<SpatialBounds>,
    /// Traversal skip: only enter this tile when a camera is inside this
    /// volume (3D Tiles `viewerRequestVolume`).
    pub viewer_request_volume: Option<SpatialBounds>,
    /// Geographic footprint (lon/lat radians) for overlay resolution.
    pub globe_rectangle: Option<GlobeRectangle>,
    /// Maximum age before re-fetching the content.
    pub content_max_age: Option<std::time::Duration>,
    /// Loader that owns this tile.  `None` means "inherit from ancestor".
    ///
    /// Set on subtree root tiles (implicit tiling) and external tileset roots.
    /// All other nodes leave this `None` - the `ContentManager` walks up the
    /// parent chain to find the effective loader.
    pub loader_index: Option<LoaderIndex>,
}

impl TileDescriptor {
    /// Convenience: a leaf tile with no children.
    pub fn leaf(bounds: SpatialBounds, geometric_error: f64, content_key: ContentKey) -> Self {
        Self {
            bounds,
            geometric_error,
            refinement: RefinementMode::Replace,
            kind: TileKind::CONTENT,
            flags: TileFlags::empty(),
            content_keys: vec![content_key],
            world_transform: DMat4::IDENTITY,
            children: Vec::new(),
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: None,
            content_max_age: None,
            loader_index: None,
        }
    }

    /// Convenience: an interior tile with no content.
    pub fn interior(bounds: SpatialBounds, geometric_error: f64, children: Vec<Self>) -> Self {
        Self {
            bounds,
            geometric_error,
            refinement: RefinementMode::Replace,
            kind: TileKind::EMPTY,
            flags: TileFlags::empty(),
            content_keys: Vec::new(),
            world_transform: DMat4::IDENTITY,
            children,
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: None,
            content_max_age: None,
            loader_index: None,
        }
    }

    /// Convenience: a child tile produced during implicit-tiling subtree
    /// population.  Sets all the invariant `None`/empty fields in one place.
    pub(crate) fn implicit_child(
        bounds: SpatialBounds,
        geometric_error: f64,
        refinement: RefinementMode,
        kind: TileKind,
        flags: TileFlags,
        content_keys: Vec<ContentKey>,
        world_transform: DMat4,
        globe_rectangle: Option<GlobeRectangle>,
    ) -> Self {
        Self {
            bounds,
            geometric_error,
            refinement,
            kind,
            flags,
            content_keys,
            world_transform,
            children: Vec::new(),
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle,
            content_max_age: None,
            loader_index: None,
        }
    }
}

/// Approximate bounding-sphere radius from any [`SpatialBounds`].
///
/// Used for load priority scoring and the cull-while-moving heuristic.
pub(crate) fn bounding_sphere_radius(bounds: &SpatialBounds) -> f64 {
    match bounds {
        SpatialBounds::Obb(o) => {
            let cx = o.half_axes.x_axis.length();
            let cy = o.half_axes.y_axis.length();
            let cz = o.half_axes.z_axis.length();
            (cx * cx + cy * cy + cz * cz).sqrt()
        }
        SpatialBounds::Sphere(s) => s.radius,
        SpatialBounds::Aabb(a) => (a.max - a.min).length() / 2.0,
        _ => 0.0,
    }
}

/// The SoA tile store.
///
/// # Layout rationale
///
/// The selection traversal only needs `bounds` and `geometric_error` for most
/// nodes (culling + SSE check).  Those two arrays are grouped here as the
/// "hot" tier.  The warm and cold tiers live in separate arrays that stay out
/// of cache during traversal.
pub struct TileStore {
    /// Bounding volumes in world space.  Indexed by `NodeIndex::slot()`.
    pub(crate) bounds: Vec<SpatialBounds>,
    /// Geometric error in metres (3D Tiles `geometricError`).
    pub(crate) geometric_errors: Vec<f64>,

    /// Per-tile refinement mode.
    pub(crate) refinement: Vec<RefinementMode>,
    pub(crate) kind: Vec<TileKind>,
    pub(crate) flags: Vec<TileFlags>,
    /// Children: (start, len) into `children_buf`.
    pub(crate) child_ranges: Vec<(u32, u16)>,
    /// Flat children arena.
    pub(crate) children_buf: Vec<TileId>,

    /// Optional per-tile content keys.
    pub(crate) content_keys: Vec<Vec<ContentKey>>,
    /// World-space transform.
    pub(crate) world_transforms: Vec<DMat4>,
    /// Optional geographic rectangle for overlay resolution.
    pub(crate) globe_rectangles: Vec<Option<GlobeRectangle>>,
    /// Optional tighter bounding volume for the content mesh.
    pub(crate) content_bounds: Vec<Option<SpatialBounds>>,
    /// Optional viewer-request-volume for traversal gating.
    pub(crate) viewer_request_volumes: Vec<Option<SpatialBounds>>,
    /// Parent link (None for the root).
    pub(crate) parents: Vec<Option<TileId>>,
    /// Explicit loader assignment for subtree roots and external tileset roots.
    ///
    /// Most nodes have `None` - they inherit the loader from their ancestor.
    /// Analogous to the per-`Tile` `TilesetContentLoader*` in cesium-native.
    pub(crate) loader_indices: Vec<Option<LoaderIndex>>,
    /// Maximum age for cached content before a re-fetch is triggered.
    /// `None` means the content never expires.  Mirrors 3D Tiles `tile.expire`.
    pub(crate) content_max_ages: Vec<Option<std::time::Duration>>,

    root: TileId,
    len: u32,
}

impl TileStore {
    /// Build a `TileStore` from a root [`NodeDescriptor`] tree.
    pub fn from_descriptor(root: TileDescriptor) -> Self {
        let mut store = Self {
            bounds: Vec::new(),
            geometric_errors: Vec::new(),
            refinement: Vec::new(),
            kind: Vec::new(),
            flags: Vec::new(),
            child_ranges: Vec::new(),
            children_buf: Vec::new(),
            content_keys: Vec::new(),
            world_transforms: Vec::new(),
            globe_rectangles: Vec::new(),
            content_bounds: Vec::new(),
            viewer_request_volumes: Vec::new(),
            parents: Vec::new(),
            loader_indices: Vec::new(),
            content_max_ages: Vec::new(),
            root: TileId::from_slot(0), // placeholder, set below
            len: 0,
        };
        let root_idx = store.insert_recursive(&root, None);
        store.root = root_idx;
        store
    }

    fn alloc_slot(&mut self) -> TileId {
        let slot = self.len;
        self.len += 1;
        // Extend all parallel arrays.
        self.bounds.push(SpatialBounds::Empty);
        self.geometric_errors.push(0.0);
        self.refinement.push(RefinementMode::Replace);
        self.kind.push(TileKind::EMPTY);
        self.flags.push(TileFlags::empty());
        self.child_ranges.push((0, 0));
        self.content_keys.push(Vec::new());
        self.world_transforms.push(DMat4::IDENTITY);
        self.globe_rectangles.push(None);
        self.content_bounds.push(None);
        self.viewer_request_volumes.push(None);
        self.parents.push(None);
        self.loader_indices.push(None);
        self.content_max_ages.push(None);
        TileId::from_slot(slot)
    }

    fn insert_recursive(&mut self, desc: &TileDescriptor, parent: Option<TileId>) -> TileId {
        let idx = self.alloc_slot();
        let s = idx.slot();
        self.bounds[s] = desc.bounds.clone();
        self.geometric_errors[s] = desc.geometric_error;
        self.refinement[s] = desc.refinement;
        self.kind[s] = desc.kind;
        self.flags[s] = desc.flags;
        self.content_keys[s] = desc.content_keys.clone();
        self.world_transforms[s] = desc.world_transform;
        self.globe_rectangles[s] = desc.globe_rectangle;
        self.content_bounds[s] = desc.content_bounds.clone();
        self.viewer_request_volumes[s] = desc.viewer_request_volume.clone();
        self.parents[s] = parent;
        self.loader_indices[s] = desc.loader_index;
        self.content_max_ages[s] = desc.content_max_age;

        if !desc.children.is_empty() {
            let child_len = desc.children.len() as u16;
            // Recurse first so all descendants are added to children_buf before
            // we append this tile's own direct-children list.
            let child_indices: Vec<TileId> = desc
                .children
                .iter()
                .map(|child_desc| self.insert_recursive(child_desc, Some(idx)))
                .collect();
            // Append direct children contiguously (post-order layout).
            let child_start = self.children_buf.len() as u32;
            self.children_buf.extend_from_slice(&child_indices);
            self.child_ranges[s] = (child_start, child_len);

            // Set CHILDREN_WITHIN_PARENT if all children's bounds are
            // conservatively contained within this tile's bounds.  Used by the
            // traversal to skip per-child frustum tests when the parent is visible.
            // Mirrors CesiumJS `_optimChildrenWithinParent`.
            let parent_bounds = &desc.bounds;
            let all_within = desc
                .children
                .iter()
                .all(|c| Self::bounds_contained_in(parent_bounds, &c.bounds));
            if all_within {
                self.flags[s].insert(TileFlags::CHILDREN_WITHIN_PARENT);
            }
        }

        idx
    }

    /// Total number of nodes in the store.
    #[inline]
    pub fn len(&self) -> usize {
        self.len as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Root tile.
    #[inline]
    pub fn root(&self) -> TileId {
        self.root
    }

    #[inline]
    pub fn bounds(&self, tile: TileId) -> &SpatialBounds {
        &self.bounds[tile.slot()]
    }

    #[inline]
    pub fn geometric_error(&self, tile: TileId) -> f64 {
        self.geometric_errors[tile.slot()]
    }

    #[inline]
    pub fn refinement(&self, tile: TileId) -> RefinementMode {
        self.refinement[tile.slot()]
    }

    #[inline]
    pub fn kind(&self, tile: TileId) -> TileKind {
        self.kind[tile.slot()]
    }

    #[inline]
    pub fn flags(&self, tile: TileId) -> TileFlags {
        self.flags[tile.slot()]
    }

    #[inline]
    pub fn might_have_latent_children(&self, tile: TileId) -> bool {
        self.flags[tile.slot()].contains(TileFlags::MIGHT_HAVE_LATENT_CHILDREN)
    }

    #[inline]
    pub fn unconditionally_refined(&self, tile: TileId) -> bool {
        self.flags[tile.slot()].contains(TileFlags::UNCONDITIONALLY_REFINED)
    }

    /// Children of `tile` as a slice of [`NodeIndex`].
    #[inline]
    pub fn children(&self, tile: TileId) -> &[TileId] {
        let (start, len) = self.child_ranges[tile.slot()];
        &self.children_buf[start as usize..start as usize + len as usize]
    }

    #[inline]
    pub fn content_keys(&self, tile: TileId) -> &[ContentKey] {
        &self.content_keys[tile.slot()]
    }

    #[inline]
    pub fn world_transform(&self, tile: TileId) -> DMat4 {
        self.world_transforms[tile.slot()]
    }

    #[inline]
    pub fn globe_rectangle(&self, tile: TileId) -> Option<GlobeRectangle> {
        self.globe_rectangles[tile.slot()]
    }

    #[inline]
    pub fn content_bounds(&self, tile: TileId) -> Option<&SpatialBounds> {
        self.content_bounds[tile.slot()].as_ref()
    }

    #[inline]
    pub fn viewer_request_volume(&self, tile: TileId) -> Option<&SpatialBounds> {
        self.viewer_request_volumes[tile.slot()].as_ref()
    }

    #[inline]
    pub fn parent(&self, tile: TileId) -> Option<TileId> {
        self.parents[tile.slot()]
    }

    /// Explicit loader index for this tile, if one was assigned.
    ///
    /// `None` means the tile inherits the loader from its closest ancestor
    /// that has one set.  Use [`TileStore::effective_loader_index`] to walk
    /// up to the first set value.
    #[inline]
    pub fn loader_index(&self, tile: TileId) -> Option<LoaderIndex> {
        self.loader_indices[tile.slot()]
    }
    /// Maximum age for this tile's content before a re-fetch is triggered.
    /// `None` means the content never expires.
    #[inline]
    pub fn content_max_age(&self, tile: TileId) -> Option<std::time::Duration> {
        self.content_max_ages.get(tile.slot()).copied().flatten()
    }
    /// Walk up the parent chain until a tile with an explicit [`LoaderIndex`]
    /// is found.  Returns `None` only if no ancestor (including `tile` itself)
    /// has a loader assigned, which should not happen in a well-formed store.
    pub fn effective_loader_index(&self, mut tile: TileId) -> Option<LoaderIndex> {
        loop {
            if let Some(li) = self.loader_indices[tile.slot()] {
                return Some(li);
            }
            match self.parents[tile.slot()] {
                Some(parent) => tile = parent,
                None => return None,
            }
        }
    }

    /// Update the bounding volume for a tile (e.g. after a load refines it).
    pub fn set_bounds(&mut self, tile: TileId, bounds: SpatialBounds) {
        self.bounds[tile.slot()] = bounds;
    }

    /// Assign or replace the loader for `tile`.
    ///
    /// Call this when an implicit subtree root or external tileset root is
    /// inserted, passing the index of the new loader in `ContentManager::loaders`.
    pub fn set_loader_index(&mut self, tile: TileId, loader: LoaderIndex) {
        self.loader_indices[tile.slot()] = Some(loader);
    }

    /// Append children to a tile that was previously a latent-expansion leaf.
    ///
    /// The tile's `MIGHT_HAVE_LATENT_CHILDREN` flag is cleared after children
    /// are inserted - it may have more (e.g. implicit tiling subtrees), in
    /// which case the caller should re-set it.
    pub fn insert_children(&mut self, parent: TileId, descs: &[TileDescriptor]) {
        assert_eq!(
            self.child_ranges[parent.slot()].1,
            0,
            "insert_children called on a tile that already has children"
        );
        let child_start = self.children_buf.len() as u32;
        let child_len = descs.len() as u16;

        let start = child_start;
        // Recurse first (fills children_buf with descendants), then record
        // the starting offset for direct children.
        let children: Vec<TileId> = descs
            .iter()
            .map(|d| self.insert_recursive(d, Some(parent)))
            .collect();
        self.children_buf.extend_from_slice(&children);
        self.child_ranges[parent.slot()] = (start, child_len);
        // Clear the latent flag - caller re-sets it if more subtrees remain.
        self.flags[parent.slot()].remove(TileFlags::MIGHT_HAVE_LATENT_CHILDREN);

        // Update CHILDREN_WITHIN_PARENT hint now that children are known.
        let parent_bounds = self.bounds[parent.slot()].clone();
        let all_within = descs
            .iter()
            .all(|d| Self::bounds_contained_in(&parent_bounds, &d.bounds));
        if all_within {
            self.flags[parent.slot()].insert(TileFlags::CHILDREN_WITHIN_PARENT);
        } else {
            self.flags[parent.slot()].remove(TileFlags::CHILDREN_WITHIN_PARENT);
        }
    }

    /// Update the globe rectangle (e.g. computed from content bounds
    /// after a tile loads).
    pub fn set_globe_rectangle(&mut self, tile: TileId, rect: Option<GlobeRectangle>) {
        self.globe_rectangles[tile.slot()] = rect;
    }

    /// Update the world transform (e.g. from a tile initializer callback).
    pub fn set_world_transform(&mut self, tile: TileId, transform: DMat4) {
        self.world_transforms[tile.slot()] = transform;
    }

    /// Conservative sphere-in-sphere containment test.
    ///
    /// Returns `true` when `child`'s circumsphere is fully inside `parent`'s
    /// circumsphere.  Uses the actual diagonal circumsphere for OBBs / AABBs
    /// so the test is tight enough to capture typical 3D-Tiles spatial
    /// subdivisions without producing false positives.
    ///
    /// Mismatched bound types fall back to `false` (conservative).
    fn bounds_contained_in(parent: &SpatialBounds, child: &SpatialBounds) -> bool {
        let Some((pc, pr)) = Self::circumsphere(parent) else {
            return false;
        };
        let Some((cc, cr)) = Self::circumsphere(child) else {
            return false;
        };
        // Child circumsphere fully inside parent circumsphere:
        //   distance(centers) + child_radius <= parent_radius
        let dist = (pc - cc).length();
        dist + cr <= pr
    }

    /// Returns `(center, circumsphere_radius)` for supported bound types.
    /// The circumsphere for an OBB is the sphere that encloses all 8 corners;
    /// its radius is the Euclidean half-diagonal √(|a|^2+|b|^2+|c|^2).
    fn circumsphere(bounds: &SpatialBounds) -> Option<(glam::DVec3, f64)> {
        match bounds {
            SpatialBounds::Sphere(s) => Some((s.center, s.radius)),
            SpatialBounds::Aabb(a) => {
                let center = a.center();
                let radius = center.distance(a.max);
                Some((center, radius))
            }
            SpatialBounds::Obb(o) => {
                // Half-diagonal of OBB = sqrt(|col0|^2 + |col1|^2 + |col2|^2)
                let radius = (o.half_axes.x_axis.length_squared()
                    + o.half_axes.y_axis.length_squared()
                    + o.half_axes.z_axis.length_squared())
                .sqrt();
                Some((o.center, radius))
            }
            _ => None, // Empty / Rectangle / Polygon - skip
        }
    }
}

impl OverlayHierarchy<TileId> for TileStore {
    fn parent(&self, tile: TileId) -> Option<TileId> {
        self.parent(tile)
    }

    fn globe_rectangle(&self, tile: TileId) -> Option<GlobeRectangle> {
        self.globe_rectangle(tile)
    }
}

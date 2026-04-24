//! [`EllipsoidTilesetLoader`] - generates an in-memory quadtree of globe-patch
//! tiles whose content is a tessellated ellipsoid surface mesh.
//!
//! Mirrors `Cesium3DTilesSelection::EllipsoidTilesetLoader`.  Useful as a
//! terrain fallback (smooth globe with no elevation features) and for testing
//! the tile-streaming pipeline without a server.
//!
//! # Key differences from C++
//!
//! * Mesh geometry is generated in `load_tile` (no separate `createModel`
//!   method - same result, simpler path via [`GltfModelBuilder`]).
//! * The root descriptor tree is built by [`create_tileset`] and handed to
//!   `ContentManager::from_descriptor`; no heap-allocated `Tile` AoS.
//! * `HeightSampler` is implemented directly on the loader struct: it always
//!   returns `0.0` (ellipsoid surface = zero height).

use std::f64::consts::PI;

use glam::DMat4;
use moderu::{GltfModelBuilder, UpAxis};
use orkester::Task;
use terra::{
    Cartographic, Ellipsoid, GlobeRectangle, aabb_for_region, calc_quadtree_max_geometric_error,
};
use zukei::{QuadtreeTileID, QuadtreeTilingScheme, SpatialBounds};

use crate::loader::{
    ContentLoader, HeightSampler, TileChildrenResult, TileLoadInput, TileLoadResult,
};
use crate::tile_store::{
    ContentKey, RefinementMode, TileDescriptor, TileFlags, TileId, TileKind, TileStore,
};

/// Tessellation resolution along each axis per tile patch.  Matches C++: 24.
const RESOLUTION: usize = 24;

/// Generates procedural globe-surface tile content analytically.
///
/// Mirrors `Cesium3DTilesSelection::EllipsoidTilesetLoader`.
pub struct EllipsoidTilesetLoader {
    ellipsoid: Ellipsoid,
    tiling_scheme: QuadtreeTilingScheme,
}

impl EllipsoidTilesetLoader {
    /// Create a loader for the given ellipsoid using a geographic 2x1
    /// root tiling scheme (matches cesium-native's constructor).
    pub fn new(ellipsoid: Ellipsoid) -> Self {
        let tiling_scheme = QuadtreeTilingScheme::geographic();
        Self {
            ellipsoid,
            tiling_scheme,
        }
    }

    /// Build the initial [`NodeDescriptor`] tree (root + level-0 children).
    ///
    /// The synthetic root is UNCONDITIONALLY_REFINED.  Each level-0 child
    /// has MIGHT_HAVE_LATENT_CHILDREN so the selection algorithm calls
    /// `create_children` lazily.
    pub fn create_tileset(&self) -> TileDescriptor {
        let root_error = self.tile_geometric_error(QuadtreeTileID::new(0, 0, 0));

        // Level-0 tiles: 2 col x 1 row.
        let mut level0_children = Vec::new();
        for x in 0..self.tiling_scheme.root_tiles_x() {
            let id = QuadtreeTileID::new(0, x, 0);
            if let Some(desc) = self.build_node(id, root_error, DMat4::IDENTITY) {
                level0_children.push(desc);
            }
        }

        // Synthetic root: UNCONDITIONALLY_REFINED, no content.
        TileDescriptor {
            bounds: SpatialBounds::Empty, // traversal doesn't cull the root
            geometric_error: f64::MAX,
            refinement: RefinementMode::Replace,
            kind: TileKind::EMPTY,
            flags: TileFlags::UNCONDITIONALLY_REFINED,
            content_keys: Vec::new(),
            world_transform: DMat4::IDENTITY,
            children: level0_children,
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: None,
            content_max_age: None,
            loader_index: None,
        }
    }

    /// Build a single `NodeDescriptor` for `tile_id`, or `None` if the tile
    /// falls outside the tiling scheme.
    fn build_node(
        &self,
        id: QuadtreeTileID,
        _parent_geometric_error: f64,
        parent_transform: DMat4,
    ) -> Option<TileDescriptor> {
        let rect_proj = self.tiling_scheme.tile_to_rectangle(id)?;

        let lon_west = rect_proj.minimum_x;
        let lon_east = rect_proj.maximum_x;
        let lat_south = rect_proj.minimum_y;
        let lat_north = rect_proj.maximum_y;

        let globe_rect = GlobeRectangle::new(lon_west, lat_south, lon_east, lat_north);

        // Northwest corner - used as the tile's local origin (matching C++).
        let origin = self
            .ellipsoid
            .cartographic_to_ecef(Cartographic::new(lon_west, lat_north, 0.0));

        // Tile transform = translation to NW ECEF corner, relative to parent origin.
        // C++: child.setTransform(glm::translate(glm::dmat4x4(1.0), northwest_ecef))
        // parent_transform already incorporates ancestor translations; we apply the
        // full absolute transform so world_transform = translate(origin).
        let world_transform = parent_transform * DMat4::from_translation(origin);
        let _ = world_transform; // suppress unused - we store absolute below

        // C++ stores absolute ECEF transforms accumulated from parent.
        // For the root children parent_transform == IDENTITY so world = translate(origin).
        let absolute_transform = DMat4::from_translation(origin);

        let geometric_error = self.tile_geometric_error(id);

        // Bounding volume: AxisAlignedBox in geographic space for globe_rectangle
        // extraction; the traversal distance is approximate for flat-Earth but
        // acceptable for an ellipsoid fallback.
        let bounds = aabb_for_region(&self.ellipsoid, lon_west, lat_south, lon_east, lat_north);

        // Content key: URI encodes the geographic patch so `load_tile` can
        // regenerate the mesh without any network fetch.
        let uri = format!("{lon_west},{lat_south},{lon_east},{lat_north}");

        Some(TileDescriptor {
            bounds,
            geometric_error,
            refinement: RefinementMode::Replace,
            kind: TileKind::CONTENT,
            flags: if id.level < 30 {
                TileFlags::MIGHT_HAVE_LATENT_CHILDREN
            } else {
                TileFlags::empty()
            },
            content_keys: vec![ContentKey::Uri(uri)],
            world_transform: absolute_transform,
            children: Vec::new(),
            content_bounds: None,
            viewer_request_volume: None,
            globe_rectangle: Some(globe_rect),
            content_max_age: None,
            loader_index: None,
        })
    }

    /// Geometric error for one tile - matches `calcQuadtreeMaxGeometricError *
    /// tile_angular_width` from C++.
    fn tile_geometric_error(&self, id: QuadtreeTileID) -> f64 {
        let max_err = calc_quadtree_max_geometric_error(&self.ellipsoid);
        let nx = self.tiling_scheme.tiles_x_at_level(id.level) as f64;
        // angular width of one tile at this level
        let angular_width = (2.0 * PI) / nx;
        8.0 * max_err * angular_width
    }

    /// Tessellate a geographic patch into a glTF mesh.
    ///
    /// Mirrors `EllipsoidTilesetLoader::createGeometry` + `createModel`.
    /// Uses Z-up (`gltfUpAxis = 2`) to match C++.
    fn create_model(
        &self,
        lon_west: f64,
        lat_south: f64,
        lon_east: f64,
        lat_north: f64,
    ) -> moderu::GltfModel {
        const N: usize = RESOLUTION;

        // Origin = NW corner ECEF (same as the tile transform).
        let origin = self
            .ellipsoid
            .cartographic_to_ecef(Cartographic::new(lon_west, lat_north, 0.0));

        let lon_step = (lon_east - lon_west) / (N - 1) as f64;
        // latitude goes north -> south (y index 0 = north)
        let lat_step = (lat_south - lat_north) / (N - 1) as f64;

        let mut vertices: Vec<[f32; 3]> = Vec::with_capacity(N * N);
        let mut normals: Vec<[f32; 3]> = Vec::with_capacity(N * N);
        let mut texcoords: Vec<[f32; 2]> = Vec::with_capacity(N * N);
        let mut indices: Vec<u16> = Vec::with_capacity(6 * (N - 1) * (N - 1));

        for x in 0..N {
            let lon = lon_west + lon_step * x as f64;
            for y in 0..N {
                let lat = lat_north + lat_step * y as f64;
                let carto = Cartographic::new(lon, lat, 0.0);
                let ecef = self.ellipsoid.cartographic_to_ecef(carto);
                // Vertex relative to tile origin (as f32, matching C++ vec3).
                let rel = ecef - origin;
                vertices.push([rel.x as f32, rel.y as f32, rel.z as f32]);

                let n = self.ellipsoid.geodetic_surface_normal_at(carto);
                normals.push([n.x as f32, n.y as f32, n.z as f32]);

                // TEXCOORD_0: u = west->east, v = north->south (glTF top-down).
                texcoords.push([x as f32 / (N - 1) as f32, y as f32 / (N - 1) as f32]);

                // Build two triangles per quad, same winding as C++.
                if x < N - 1 && y < N - 1 {
                    let idx = (N * x + y) as u16;
                    let a = idx + 1;
                    let b = idx + N as u16;
                    let c = b + 1;
                    // C++: {b, index, a, b, a, c}
                    indices.extend_from_slice(&[b, idx, a, b, a, c]);
                }
            }
        }

        let mut b = GltfModelBuilder::new();
        // Z-up - matches C++ `model.extras["gltfUpAxis"] = Axis::Z`
        b.up_axis(UpAxis::Z);

        let pos_acc = b.add_accessor(&vertices);
        let norm_acc = b.add_accessor(&normals);
        let tc_acc = b.add_accessor(&texcoords);
        let idx_acc = b.add_indices(&indices);
        let mat = b.add_default_material([1.0, 1.0, 1.0, 1.0]);

        let prim = b
            .primitive()
            .indices(idx_acc)
            .attribute("POSITION", pos_acc)
            .attribute("NORMAL", norm_acc)
            .attribute("TEXCOORD_0", tc_acc)
            .material(mat)
            .build();

        b.add_mesh(prim);
        b.finish()
    }
}

impl ContentLoader for EllipsoidTilesetLoader {
    fn load_tile(&self, input: TileLoadInput) -> Task<TileLoadResult> {
        // Content key is the URI string "lon_west,lat_south,lon_east,lat_north".
        let uri = match input.content_keys.first() {
            Some(ContentKey::Uri(u)) => u.clone(),
            _ => return orkester::resolved(TileLoadResult::failed()),
        };

        let coords: Vec<f64> = uri.split(',').filter_map(|s| s.parse().ok()).collect();

        if coords.len() != 4 {
            return orkester::resolved(TileLoadResult::failed());
        }

        let model = self.create_model(coords[0], coords[1], coords[2], coords[3]);
        let mut result = TileLoadResult::gltf(model);
        // Ellipsoid geometry is Z-up.
        result.gltf_up_axis = zukei::Axis::Z;
        orkester::resolved(result)
    }

    fn create_children(
        &self,
        tile: TileId,
        store: &TileStore,
        _ellipsoid: &Ellipsoid,
    ) -> TileChildrenResult {
        // Decode this tile's tile ID from its content key URI.
        let uri = match store.content_keys(tile).first() {
            Some(ContentKey::Uri(u)) => u.clone(),
            _ => return TileChildrenResult::None,
        };

        let coords: Vec<f64> = uri.split(',').filter_map(|s| s.parse().ok()).collect();

        if coords.len() != 4 {
            return TileChildrenResult::None;
        }

        let (lon_west, lat_south, lon_east, lat_north) =
            (coords[0], coords[1], coords[2], coords[3]);

        // Infer the QuadtreeTileID from the geographic rectangle.
        // We reconstruct the level+x+y from the tiling scheme.
        let tile_id = match self
            .tiling_scheme
            .tile_for_angular_width(lon_west, lat_south, lon_east, lat_north)
        {
            Some(id) => id,
            None => return TileChildrenResult::None,
        };

        // C++: only subdivide through level 30 (u32 safety limit).
        if tile_id.level >= 30 {
            return TileChildrenResult::None;
        }

        let parent_error = store.geometric_error(tile);
        let parent_transform = store.world_transform(tile);

        let child_ids = tile_id.children();
        let mut children = Vec::with_capacity(4);
        for child_id in child_ids {
            if let Some(desc) = self.build_node(child_id, parent_error, parent_transform) {
                children.push(desc);
            }
        }

        if children.is_empty() {
            TileChildrenResult::None
        } else {
            TileChildrenResult::Children(children)
        }
    }

    fn height_sampler(&self) -> Option<&dyn HeightSampler> {
        Some(self)
    }
}

impl HeightSampler for EllipsoidTilesetLoader {
    /// Ellipsoid surface is always at height 0.
    fn sample_height(
        &self,
        _longitude: f64,
        _latitude: f64,
        _ellipsoid: &Ellipsoid,
    ) -> Option<f64> {
        Some(0.0)
    }
}

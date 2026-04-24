//! `arazi` — 3D terrain generation and selection library.

pub mod heightmap;
pub mod quantized_mesh;

pub use heightmap::{
    ChildFlags, HeightmapError, HeightmapTile, MASK_CELL_SIZE, MASK_SIZE, TILE_CELL_SIZE,
    TILE_SIZE, WaterMask, decode_heightmap, encode_heightmap,
};

pub use quantized_mesh::{
    HwmIndex, QuadtreeTileRectangularRange, QuantizedMeshError, QuantizedMeshHeader,
    QuantizedMeshInput, QuantizedMeshResult, decode_quantized_mesh, encode_quantized_mesh,
    high_watermark_decode,
};

mod provider;

pub use provider::{ArcGISTerrainProvider, CesiumTerrainProvider, TerrainProvider};

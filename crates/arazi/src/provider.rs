pub trait TerrainProvider {}

pub struct CesiumTerrainProvider {}

impl TerrainProvider for CesiumTerrainProvider {}

pub struct ArcGISTerrainProvider {}

impl TerrainProvider for ArcGISTerrainProvider {}

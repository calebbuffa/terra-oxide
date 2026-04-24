//! `vekuta` — GeoJSON parsing and vector data rasterization.

pub mod document;
pub mod error;
pub mod iter;
pub mod style;
pub mod types;

#[cfg(not(target_arch = "wasm32"))]
pub mod rasterizer;

#[cfg(not(target_arch = "wasm32"))]
pub use rasterizer::VectorRasterizer;

pub use document::{Attribution, GeoJsonDocument};
pub use error::LoadError;
pub use iter::{FeatureIter, LineIter, PointIter, PolygonIter};
pub use style::{
    Color, ColorMode, ColorStyle, LineStyle, LineWidthMode, PolygonStyle, VectorStyle,
};
pub use types::{
    Bbox, Feature, FeatureCollection, FeatureId, GeoJsonObject, Geometry, GeometryCollection,
    LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Position,
    bbox_to_globe_rectangle,
};

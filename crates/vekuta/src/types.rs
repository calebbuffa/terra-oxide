//! GeoJSON object types (RFC 7946).

use serde::Deserialize;
use serde_json::{Map, Value};
use terra::GlobeRectangle;

use crate::style::VectorStyle;

/// A GeoJSON position: `[longitude_deg, latitude_deg, height_m]`.
/// Height defaults to `0.0` when absent. All values are degrees/metres (RFC 7946).
pub type Position = [f64; 3];

/// A GeoJSON `bbox`: `[west_deg, south_deg, east_deg, north_deg]`.
pub type Bbox = [f64; 4];

/// Convert a [`Bbox`] (degrees) to a [`GlobeRectangle`] (radians).
pub fn bbox_to_globe_rectangle(bbox: Bbox) -> GlobeRectangle {
    GlobeRectangle::from_degrees(bbox[0], bbox[1], bbox[2], bbox[3])
}

/// The optional `id` field of a GeoJSON Feature (string or integer).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum FeatureId {
    String(String),
    Number(i64),
}

macro_rules! geojson_geometry {
    ($(#[$meta:meta])* $name:ident { $coords_field:ident: $coords_ty:ty }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Deserialize)]
        pub struct $name {
            #[serde(rename = "coordinates")]
            pub $coords_field: $coords_ty,
            pub bbox: Option<Bbox>,
            #[serde(skip)]
            pub style: Option<VectorStyle>,
            #[serde(flatten)]
            pub foreign_members: Map<String, Value>,
        }

        impl $name {
            pub fn new($coords_field: $coords_ty) -> Self {
                Self { $coords_field, bbox: None, style: None, foreign_members: Map::new() }
            }
        }
    };
}

geojson_geometry! {
    /// A single geographic position.
    Point { coordinates: Position }
}

geojson_geometry! {
    /// An array of positions.
    MultiPoint { coordinates: Vec<Position> }
}

geojson_geometry! {
    /// A sequence of two or more positions.
    LineString { coordinates: Vec<Position> }
}

geojson_geometry! {
    /// An array of LineStrings.
    MultiLineString { coordinates: Vec<Vec<Position>> }
}

geojson_geometry! {
    /// Polygon rings (first = exterior, rest = holes). Each ring is closed (first == last).
    Polygon { coordinates: Vec<Vec<Position>> }
}

geojson_geometry! {
    /// An array of Polygons.
    MultiPolygon { coordinates: Vec<Vec<Vec<Position>>> }
}

/// A heterogeneous collection of geometry objects (no Features).
#[derive(Debug, Clone, Deserialize)]
pub struct GeometryCollection {
    #[serde(rename = "geometries")]
    pub geometries: Vec<Geometry>,
    pub bbox: Option<Bbox>,
    #[serde(skip)]
    pub style: Option<VectorStyle>,
    #[serde(flatten)]
    pub foreign_members: Map<String, Value>,
}

impl GeometryCollection {
    pub fn new(geometries: Vec<Geometry>) -> Self {
        Self {
            geometries,
            bbox: None,
            style: None,
            foreign_members: Map::new(),
        }
    }
}

/// Any GeoJSON geometry type (no Feature or FeatureCollection).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum Geometry {
    Point(Point),
    MultiPoint(MultiPoint),
    LineString(LineString),
    MultiLineString(MultiLineString),
    Polygon(Polygon),
    MultiPolygon(MultiPolygon),
    GeometryCollection(GeometryCollection),
}

impl Geometry {
    pub fn bbox(&self) -> Option<Bbox> {
        match self {
            Geometry::Point(g) => g.bbox,
            Geometry::MultiPoint(g) => g.bbox,
            Geometry::LineString(g) => g.bbox,
            Geometry::MultiLineString(g) => g.bbox,
            Geometry::Polygon(g) => g.bbox,
            Geometry::MultiPolygon(g) => g.bbox,
            Geometry::GeometryCollection(g) => g.bbox,
        }
    }

    pub fn style(&self) -> Option<&VectorStyle> {
        match self {
            Geometry::Point(g) => g.style.as_ref(),
            Geometry::MultiPoint(g) => g.style.as_ref(),
            Geometry::LineString(g) => g.style.as_ref(),
            Geometry::MultiLineString(g) => g.style.as_ref(),
            Geometry::Polygon(g) => g.style.as_ref(),
            Geometry::MultiPolygon(g) => g.style.as_ref(),
            Geometry::GeometryCollection(g) => g.style.as_ref(),
        }
    }
}

/// A GeoJSON Feature.
#[derive(Debug, Clone, Deserialize)]
pub struct Feature {
    pub id: Option<FeatureId>,
    pub geometry: Option<Box<Geometry>>,
    /// May be `null` or an object per RFC 7946.
    pub properties: Option<Map<String, Value>>,
    pub bbox: Option<Bbox>,
    #[serde(skip)]
    pub style: Option<VectorStyle>,
    #[serde(flatten)]
    pub foreign_members: Map<String, Value>,
}

/// A GeoJSON FeatureCollection.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureCollection {
    pub features: Vec<GeoJsonObject>,
    pub bbox: Option<Bbox>,
    #[serde(skip)]
    pub style: Option<VectorStyle>,
    #[serde(flatten)]
    pub foreign_members: Map<String, Value>,
}

impl FeatureCollection {
    pub fn new(features: Vec<GeoJsonObject>) -> Self {
        Self {
            features,
            bbox: None,
            style: None,
            foreign_members: Map::new(),
        }
    }
}

/// Any GeoJSON object.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum GeoJsonObject {
    Point(Point),
    MultiPoint(MultiPoint),
    LineString(LineString),
    MultiLineString(MultiLineString),
    Polygon(Polygon),
    MultiPolygon(MultiPolygon),
    GeometryCollection(GeometryCollection),
    Feature(Feature),
    FeatureCollection(FeatureCollection),
}

impl GeoJsonObject {
    pub fn bbox(&self) -> Option<Bbox> {
        match self {
            GeoJsonObject::Point(g) => g.bbox,
            GeoJsonObject::MultiPoint(g) => g.bbox,
            GeoJsonObject::LineString(g) => g.bbox,
            GeoJsonObject::MultiLineString(g) => g.bbox,
            GeoJsonObject::Polygon(g) => g.bbox,
            GeoJsonObject::MultiPolygon(g) => g.bbox,
            GeoJsonObject::GeometryCollection(g) => g.bbox,
            GeoJsonObject::Feature(f) => f.bbox,
            GeoJsonObject::FeatureCollection(fc) => fc.bbox,
        }
    }

    pub fn style(&self) -> Option<&VectorStyle> {
        match self {
            GeoJsonObject::Point(g) => g.style.as_ref(),
            GeoJsonObject::MultiPoint(g) => g.style.as_ref(),
            GeoJsonObject::LineString(g) => g.style.as_ref(),
            GeoJsonObject::MultiLineString(g) => g.style.as_ref(),
            GeoJsonObject::Polygon(g) => g.style.as_ref(),
            GeoJsonObject::MultiPolygon(g) => g.style.as_ref(),
            GeoJsonObject::GeometryCollection(g) => g.style.as_ref(),
            GeoJsonObject::Feature(f) => f.style.as_ref(),
            GeoJsonObject::FeatureCollection(fc) => fc.style.as_ref(),
        }
    }

    pub fn foreign_members(&self) -> &Map<String, Value> {
        match self {
            GeoJsonObject::Point(g) => &g.foreign_members,
            GeoJsonObject::MultiPoint(g) => &g.foreign_members,
            GeoJsonObject::LineString(g) => &g.foreign_members,
            GeoJsonObject::MultiLineString(g) => &g.foreign_members,
            GeoJsonObject::Polygon(g) => &g.foreign_members,
            GeoJsonObject::MultiPolygon(g) => &g.foreign_members,
            GeoJsonObject::GeometryCollection(g) => &g.foreign_members,
            GeoJsonObject::Feature(f) => &f.foreign_members,
            GeoJsonObject::FeatureCollection(fc) => &fc.foreign_members,
        }
    }

    pub fn points(&self) -> crate::iter::PointIter<'_> {
        crate::iter::PointIter::new(self)
    }
    pub fn lines(&self) -> crate::iter::LineIter<'_> {
        crate::iter::LineIter::new(self)
    }
    pub fn polygons(&self) -> crate::iter::PolygonIter<'_> {
        crate::iter::PolygonIter::new(self)
    }
    pub fn features(&self) -> crate::iter::FeatureIter<'_> {
        crate::iter::FeatureIter::new(self)
    }
}

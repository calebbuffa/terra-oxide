use serde_json::Value;
use terra::GlobeRectangle;

use crate::error::LoadError;
use crate::types::{GeoJsonObject, Geometry, Position, bbox_to_globe_rectangle};

/// Attribution HTML for a loaded GeoJSON document.
#[derive(Debug, Clone)]
pub struct Attribution {
    pub html: String,
}

/// A parsed GeoJSON document.
#[derive(Debug, Clone)]
pub struct GeoJsonDocument {
    pub root: GeoJsonObject,
    pub attributions: Vec<Attribution>,
}

impl GeoJsonDocument {
    /// Parse a GeoJSON document from raw bytes.
    ///
    /// Non-fatal issues (e.g. unclosed polygon rings) are emitted as `log::warn!`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        let root: GeoJsonObject = serde_json::from_slice(bytes)?;
        let mut root = root;
        validate_and_fix(&mut root);
        Ok(Self {
            root,
            attributions: vec![],
        })
    }

    /// Parse from a pre-parsed [`serde_json::Value`].
    pub fn from_json(json: &Value) -> Result<Self, LoadError> {
        let root: GeoJsonObject = serde_json::from_value(json.clone())?;
        let mut root = root;
        validate_and_fix(&mut root);
        Ok(Self {
            root,
            attributions: vec![],
        })
    }

    pub fn points(&self) -> crate::iter::PointIter<'_> {
        self.root.points()
    }
    pub fn lines(&self) -> crate::iter::LineIter<'_> {
        self.root.lines()
    }
    pub fn polygons(&self) -> crate::iter::PolygonIter<'_> {
        self.root.polygons()
    }
    pub fn features(&self) -> crate::iter::FeatureIter<'_> {
        self.root.features()
    }

    /// Bounding rectangle (radians): uses explicit `bbox` if present, otherwise
    /// derived from all coordinate positions.
    pub fn bounds(&self) -> Option<GlobeRectangle> {
        if let Some(bbox) = self.root.bbox() {
            return Some(bbox_to_globe_rectangle(bbox));
        }
        let mut west = f64::MAX;
        let mut south = f64::MAX;
        let mut east = f64::MIN;
        let mut north = f64::MIN;
        let mut any = false;
        let mut update = |pos: &Position| {
            west = west.min(pos[0]);
            south = south.min(pos[1]);
            east = east.max(pos[0]);
            north = north.max(pos[1]);
            any = true;
        };
        for pos in self.points() {
            update(pos);
        }
        for line in self.lines() {
            for pos in line {
                update(pos);
            }
        }
        for rings in self.polygons() {
            for ring in rings {
                for pos in ring {
                    update(pos);
                }
            }
        }
        if any {
            Some(GlobeRectangle::from_degrees(west, south, east, north))
        } else {
            None
        }
    }
}

/// Recursively validate and fix polygon ring closure, emitting warnings for issues found.
fn validate_and_fix(obj: &mut GeoJsonObject) {
    match obj {
        GeoJsonObject::Polygon(p) => fix_polygon_rings(&mut p.coordinates),
        GeoJsonObject::MultiPolygon(mp) => {
            for rings in &mut mp.coordinates {
                fix_polygon_rings(rings);
            }
        }
        GeoJsonObject::GeometryCollection(gc) => {
            for geom in &mut gc.geometries {
                validate_geometry(geom);
            }
        }
        GeoJsonObject::Feature(f) => {
            if let Some(geom) = &mut f.geometry {
                validate_geometry(geom);
            }
        }
        GeoJsonObject::FeatureCollection(fc) => {
            for child in &mut fc.features {
                validate_and_fix(child);
            }
        }
        _ => {}
    }
}

fn validate_geometry(geom: &mut Geometry) {
    match geom {
        Geometry::Polygon(p) => fix_polygon_rings(&mut p.coordinates),
        Geometry::MultiPolygon(mp) => {
            for rings in &mut mp.coordinates {
                fix_polygon_rings(rings);
            }
        }
        Geometry::GeometryCollection(gc) => {
            for child in &mut gc.geometries {
                validate_geometry(child);
            }
        }
        _ => {}
    }
}

fn fix_polygon_rings(rings: &mut Vec<Vec<Position>>) {
    for ring in rings.iter_mut() {
        if ring.len() < 4 {
            log::warn!(
                "polygon ring has {} positions (minimum 4); ring may be invalid",
                ring.len()
            );
            continue;
        }
        let first = ring[0];
        let last = *ring.last().unwrap();
        if first[0] != last[0] || first[1] != last[1] {
            log::warn!("polygon ring is not closed; auto-closing");
            ring.push(first);
        }
    }
}

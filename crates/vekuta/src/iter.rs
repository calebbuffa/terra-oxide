//! Lazy recursive iterators over GeoJSON object trees.

use crate::types::{Feature, GeoJsonObject, Geometry, Position};

/// A stack node that can be either a top-level object or a bare geometry.
enum Node<'a> {
    Obj(&'a GeoJsonObject),
    Geom(&'a Geometry),
}

/// Push a geometry from a Feature's geometry field (type-erased as `&Geometry`).
#[inline]
fn push_geom_children<'a>(stack: &mut Vec<Node<'a>>, g: &'a Geometry) {
    if let Geometry::GeometryCollection(gc) = g {
        for child in gc.geometries.iter().rev() {
            stack.push(Node::Geom(child));
        }
    }
}

/// Iterator yielding every `&Position` reachable from a [`GeoJsonObject`] tree.
pub struct PointIter<'a> {
    stack: Vec<Node<'a>>,
    multi_buf: std::slice::Iter<'a, Position>,
}

impl<'a> PointIter<'a> {
    pub(crate) fn new(root: &'a GeoJsonObject) -> Self {
        Self {
            stack: vec![Node::Obj(root)],
            multi_buf: [].iter(),
        }
    }
}

impl<'a> Iterator for PointIter<'a> {
    type Item = &'a Position;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(pos) = self.multi_buf.next() {
            return Some(pos);
        }
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Obj(obj) => match obj {
                    GeoJsonObject::Point(p) => return Some(&p.coordinates),
                    GeoJsonObject::MultiPoint(mp) => {
                        self.multi_buf = mp.coordinates.iter();
                        if let Some(pos) = self.multi_buf.next() {
                            return Some(pos);
                        }
                    }
                    GeoJsonObject::GeometryCollection(gc) => {
                        for child in gc.geometries.iter().rev() {
                            self.stack.push(Node::Geom(child));
                        }
                    }
                    GeoJsonObject::Feature(f) => {
                        if let Some(g) = &f.geometry {
                            self.stack.push(Node::Geom(g.as_ref()));
                        }
                    }
                    GeoJsonObject::FeatureCollection(fc) => {
                        for child in fc.features.iter().rev() {
                            self.stack.push(Node::Obj(child));
                        }
                    }
                    _ => {} // LineString, MultiLineString, Polygon, MultiPolygon have no points
                },
                Node::Geom(g) => match g {
                    Geometry::Point(p) => return Some(&p.coordinates),
                    Geometry::MultiPoint(mp) => {
                        self.multi_buf = mp.coordinates.iter();
                        if let Some(pos) = self.multi_buf.next() {
                            return Some(pos);
                        }
                    }
                    _ => push_geom_children(&mut self.stack, g),
                },
            }
        }
        None
    }
}

/// Iterator yielding every `&Vec<Position>` (LineString coordinates) in the tree.
pub struct LineIter<'a> {
    stack: Vec<Node<'a>>,
    multi_buf: std::slice::Iter<'a, Vec<Position>>,
}

impl<'a> LineIter<'a> {
    pub(crate) fn new(root: &'a GeoJsonObject) -> Self {
        Self {
            stack: vec![Node::Obj(root)],
            multi_buf: [].iter(),
        }
    }
}

impl<'a> Iterator for LineIter<'a> {
    type Item = &'a Vec<Position>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(line) = self.multi_buf.next() {
            return Some(line);
        }
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Obj(obj) => match obj {
                    GeoJsonObject::LineString(ls) => return Some(&ls.coordinates),
                    GeoJsonObject::MultiLineString(mls) => {
                        self.multi_buf = mls.coordinates.iter();
                        if let Some(line) = self.multi_buf.next() {
                            return Some(line);
                        }
                    }
                    GeoJsonObject::GeometryCollection(gc) => {
                        for child in gc.geometries.iter().rev() {
                            self.stack.push(Node::Geom(child));
                        }
                    }
                    GeoJsonObject::Feature(f) => {
                        if let Some(g) = &f.geometry {
                            self.stack.push(Node::Geom(g.as_ref()));
                        }
                    }
                    GeoJsonObject::FeatureCollection(fc) => {
                        for child in fc.features.iter().rev() {
                            self.stack.push(Node::Obj(child));
                        }
                    }
                    _ => {}
                },
                Node::Geom(g) => match g {
                    Geometry::LineString(ls) => return Some(&ls.coordinates),
                    Geometry::MultiLineString(mls) => {
                        self.multi_buf = mls.coordinates.iter();
                        if let Some(line) = self.multi_buf.next() {
                            return Some(line);
                        }
                    }
                    _ => push_geom_children(&mut self.stack, g),
                },
            }
        }
        None
    }
}

/// Iterator yielding every `&Vec<Vec<Position>>` (polygon rings) in the tree.
pub struct PolygonIter<'a> {
    stack: Vec<Node<'a>>,
    multi_buf: std::slice::Iter<'a, Vec<Vec<Position>>>,
}

impl<'a> PolygonIter<'a> {
    pub(crate) fn new(root: &'a GeoJsonObject) -> Self {
        Self {
            stack: vec![Node::Obj(root)],
            multi_buf: [].iter(),
        }
    }
}

impl<'a> Iterator for PolygonIter<'a> {
    type Item = &'a Vec<Vec<Position>>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(rings) = self.multi_buf.next() {
            return Some(rings);
        }
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Obj(obj) => match obj {
                    GeoJsonObject::Polygon(p) => return Some(&p.coordinates),
                    GeoJsonObject::MultiPolygon(mp) => {
                        self.multi_buf = mp.coordinates.iter();
                        if let Some(rings) = self.multi_buf.next() {
                            return Some(rings);
                        }
                    }
                    GeoJsonObject::GeometryCollection(gc) => {
                        for child in gc.geometries.iter().rev() {
                            self.stack.push(Node::Geom(child));
                        }
                    }
                    GeoJsonObject::Feature(f) => {
                        if let Some(g) = &f.geometry {
                            self.stack.push(Node::Geom(g.as_ref()));
                        }
                    }
                    GeoJsonObject::FeatureCollection(fc) => {
                        for child in fc.features.iter().rev() {
                            self.stack.push(Node::Obj(child));
                        }
                    }
                    _ => {}
                },
                Node::Geom(g) => match g {
                    Geometry::Polygon(p) => return Some(&p.coordinates),
                    Geometry::MultiPolygon(mp) => {
                        self.multi_buf = mp.coordinates.iter();
                        if let Some(rings) = self.multi_buf.next() {
                            return Some(rings);
                        }
                    }
                    _ => push_geom_children(&mut self.stack, g),
                },
            }
        }
        None
    }
}

/// Iterator yielding every [`Feature`] in the tree.
pub struct FeatureIter<'a> {
    stack: Vec<Node<'a>>,
}

impl<'a> FeatureIter<'a> {
    pub(crate) fn new(root: &'a GeoJsonObject) -> Self {
        Self {
            stack: vec![Node::Obj(root)],
        }
    }
}

impl<'a> Iterator for FeatureIter<'a> {
    type Item = &'a Feature;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(node) = self.stack.pop() {
            match node {
                Node::Obj(GeoJsonObject::Feature(f)) => return Some(f),
                Node::Obj(GeoJsonObject::FeatureCollection(fc)) => {
                    for child in fc.features.iter().rev() {
                        self.stack.push(Node::Obj(child));
                    }
                }
                _ => {}
            }
        }
        None
    }
}

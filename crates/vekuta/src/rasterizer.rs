//! Vector rasterization using [`tiny_skia`].
//!
//! Rasterizes GeoJSON geometry objects into an RGBA pixel buffer sized to a
//! specific tile rectangle.

use tiny_skia::{
    Color as SkColor, FillRule, LineCap, LineJoin, Paint, PathBuilder, Pixmap, Stroke, Transform,
};

use terra::GlobeRectangle;

use crate::style::{Color, LineStyle, PolygonStyle, VectorStyle};
use crate::types::{GeoJsonObject, Geometry, Position};

/// Convert a geographic `[lon_deg, lat_deg, ...]` position to pixel coordinates
/// within the tile, given the tile's bounds in **radians**.
///
/// - x increases left->right (west->east)
/// - y increases top->bottom (north->south) — standard image convention
#[inline]
fn pos_to_pixel(pos: &Position, bounds: &GlobeRectangle, width: f32, height: f32) -> (f32, f32) {
    use std::f64::consts::PI;
    let lon_rad = pos[0].to_radians();
    let lat_rad = pos[1].to_radians();

    let west = bounds.west;
    let east = bounds.east;
    let south = bounds.south;
    let north = bounds.north;

    let bw = if east > west {
        east - west
    } else {
        east - west + 2.0 * PI
    };
    let bh = north - south;

    let x = ((lon_rad - west) / bw * width as f64) as f32;
    let y = ((north - lat_rad) / bh * height as f64) as f32;
    (x, y)
}

fn sk_color(c: Color) -> SkColor {
    SkColor::from_rgba(c.r, c.g, c.b, c.a).unwrap_or(SkColor::WHITE)
}

/// Rasterizes GeoJSON vector data into an RGBA tile image.
pub struct VectorRasterizer {
    bounds: GlobeRectangle,
    pixmap: Pixmap,
}

impl VectorRasterizer {
    /// Create a new rasterizer for the given tile bounds (radians) and pixel dimensions.
    pub fn new(bounds: GlobeRectangle, width: u32, height: u32) -> Self {
        let pixmap = Pixmap::new(width, height).expect("invalid pixmap dimensions");
        Self { bounds, pixmap }
    }

    /// Rasterize all renderable geometry in `object` using `default_style` as
    /// a fallback when the object carries no style of its own.
    pub fn draw_object(&mut self, object: &GeoJsonObject, default_style: &VectorStyle) {
        self.draw_object_with_style(object, default_style, default_style);
    }

    /// Consume the rasterizer and return the RGBA pixel buffer (row-major,
    /// top-left origin, 4 bytes per pixel).
    pub fn finish(self) -> Vec<u8> {
        self.pixmap.data().to_vec()
    }

    pub fn draw_line_string(&mut self, coords: &[Position], style: &LineStyle) {
        if coords.len() < 2 {
            return;
        }
        let (w, h) = (self.pixmap.width() as f32, self.pixmap.height() as f32);

        let mut pb = PathBuilder::new();
        let (x, y) = pos_to_pixel(&coords[0], &self.bounds, w, h);
        pb.move_to(x, y);
        for pos in &coords[1..] {
            let (x, y) = pos_to_pixel(pos, &self.bounds, w, h);
            pb.line_to(x, y);
        }
        let path = match pb.finish() {
            Some(p) => p,
            None => return,
        };

        let color = style.resolve_color(0);
        let mut paint = Paint::default();
        paint.set_color(sk_color(color));
        paint.anti_alias = true;

        let stroke = Stroke {
            width: style.width as f32,
            line_cap: LineCap::Round,
            line_join: LineJoin::Round,
            ..Stroke::default()
        };

        self.pixmap
            .stroke_path(&path, &paint, &stroke, Transform::identity(), None);
    }

    pub fn draw_polygon(&mut self, rings: &[Vec<Position>], style: &PolygonStyle) {
        if rings.is_empty() {
            return;
        }
        let (w, h) = (self.pixmap.width() as f32, self.pixmap.height() as f32);

        let path = {
            let mut pb = PathBuilder::new();
            for ring in rings {
                if ring.is_empty() {
                    continue;
                }
                let (x, y) = pos_to_pixel(&ring[0], &self.bounds, w, h);
                pb.move_to(x, y);
                for pos in &ring[1..] {
                    let (x, y) = pos_to_pixel(pos, &self.bounds, w, h);
                    pb.line_to(x, y);
                }
                pb.close();
            }
            match pb.finish() {
                Some(p) => p,
                None => return,
            }
        };

        // Fill
        if let Some(fill) = &style.fill {
            let color = fill.resolve(0);
            let mut paint = Paint::default();
            paint.set_color(sk_color(color));
            paint.anti_alias = true;
            self.pixmap.fill_path(
                &path,
                &paint,
                FillRule::EvenOdd,
                Transform::identity(),
                None,
            );
        }

        // Outline
        if let Some(outline) = &style.outline {
            let color = outline.resolve_color(0);
            let mut paint = Paint::default();
            paint.set_color(sk_color(color));
            paint.anti_alias = true;

            let stroke = Stroke {
                width: outline.width as f32,
                line_cap: LineCap::Round,
                line_join: LineJoin::Round,
                ..Stroke::default()
            };

            self.pixmap
                .stroke_path(&path, &paint, &stroke, Transform::identity(), None);
        }
    }

    fn draw_object_with_style(
        &mut self,
        object: &GeoJsonObject,
        parent_style: &VectorStyle,
        default_style: &VectorStyle,
    ) {
        let effective = object.style().unwrap_or(parent_style);
        match object {
            GeoJsonObject::Point(_) | GeoJsonObject::MultiPoint(_) => {}
            GeoJsonObject::LineString(ls) => {
                self.draw_line_string(&ls.coordinates, &effective.line);
            }
            GeoJsonObject::MultiLineString(mls) => {
                for line in &mls.coordinates {
                    self.draw_line_string(line, &effective.line);
                }
            }
            GeoJsonObject::Polygon(p) => {
                self.draw_polygon(&p.coordinates, &effective.polygon);
            }
            GeoJsonObject::MultiPolygon(mp) => {
                for rings in &mp.coordinates {
                    self.draw_polygon(rings, &effective.polygon);
                }
            }
            GeoJsonObject::GeometryCollection(gc) => {
                let gc_style = gc.style.as_ref().unwrap_or(effective);
                for child in &gc.geometries {
                    self.draw_geometry(child, gc_style);
                }
            }
            GeoJsonObject::Feature(f) => {
                let feat_style = f.style.as_ref().unwrap_or(effective);
                if let Some(geom) = &f.geometry {
                    let geom_style = match geom.as_ref() {
                        Geometry::Point(p) => p.style.as_ref().unwrap_or(feat_style),
                        Geometry::MultiPoint(mp) => mp.style.as_ref().unwrap_or(feat_style),
                        Geometry::LineString(ls) => ls.style.as_ref().unwrap_or(feat_style),
                        Geometry::MultiLineString(mls) => mls.style.as_ref().unwrap_or(feat_style),
                        Geometry::Polygon(p) => p.style.as_ref().unwrap_or(feat_style),
                        Geometry::MultiPolygon(mp) => mp.style.as_ref().unwrap_or(feat_style),
                        Geometry::GeometryCollection(gc) => gc.style.as_ref().unwrap_or(feat_style),
                    };
                    self.draw_geometry(geom, geom_style);
                }
            }
            GeoJsonObject::FeatureCollection(fc) => {
                let fc_style = fc.style.as_ref().unwrap_or(effective);
                for child in &fc.features {
                    self.draw_object_with_style(child, fc_style, default_style);
                }
            }
        }
    }

    fn draw_geometry(&mut self, g: &Geometry, style: &VectorStyle) {
        match g {
            Geometry::LineString(ls) => self.draw_line_string(&ls.coordinates, &style.line),
            Geometry::MultiLineString(mls) => {
                for line in &mls.coordinates {
                    self.draw_line_string(line, &style.line);
                }
            }
            Geometry::Polygon(p) => self.draw_polygon(&p.coordinates, &style.polygon),
            Geometry::MultiPolygon(mp) => {
                for rings in &mp.coordinates {
                    self.draw_polygon(rings, &style.polygon);
                }
            }
            Geometry::GeometryCollection(gc) => {
                let gc_style = gc.style.as_ref().unwrap_or(style);
                for child in &gc.geometries {
                    self.draw_geometry(child, gc_style);
                }
            }
            // Points and MultiPoints are not rasterized (no area/line to draw).
            Geometry::Point(_) | Geometry::MultiPoint(_) => {}
        }
    }
}

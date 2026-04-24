//! Vector styling types.

/// An RGBA color with components in `[0.0, 1.0]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const WHITE: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 1.0,
        a: 1.0,
    };
    pub const BLACK: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    };
    pub const TRANSPARENT: Self = Self {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 0.0,
    };
    pub const YELLOW: Self = Self {
        r: 1.0,
        g: 1.0,
        b: 0.0,
        a: 1.0,
    };

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Pack to `[r, g, b, a]` bytes in `[0, 255]`.
    pub fn rgba_u8(self) -> [u8; 4] {
        [
            (self.r.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.g.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.b.clamp(0.0, 1.0) * 255.0).round() as u8,
            (self.a.clamp(0.0, 1.0) * 255.0).round() as u8,
        ]
    }

    /// Deterministic "random" color from a seed, for `ColorMode::Random`.
    pub fn from_seed(seed: u64) -> Self {
        // Simple hash: mix seed bits and map to hue.
        let h = ((seed.wrapping_mul(2654435761) >> 32) as f32) / (u32::MAX as f32);
        hsl_to_rgb(h, 0.7, 0.55)
    }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> Color {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h * 6.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match (h * 6.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    Color::new(r + m, g + m, b + m, 1.0)
}

/// Whether the color is fixed or derived from a random seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMode {
    #[default]
    Normal,
    Random,
}

/// A color with an optional random-seed mode.
#[derive(Debug, Clone)]
pub struct ColorStyle {
    pub color: Color,
    pub mode: ColorMode,
}

impl ColorStyle {
    pub fn solid(color: Color) -> Self {
        Self {
            color,
            mode: ColorMode::Normal,
        }
    }

    /// Resolve the effective color, using `random_seed` when `mode == Random`.
    pub fn resolve(&self, random_seed: u64) -> Color {
        match self.mode {
            ColorMode::Normal => self.color,
            ColorMode::Random => Color::from_seed(random_seed),
        }
    }
}

impl Default for ColorStyle {
    fn default() -> Self {
        Self::solid(Color::WHITE)
    }
}

/// Unit for line width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LineWidthMode {
    #[default]
    Pixels,
    Meters,
}

/// Style for line/polyline geometry.
#[derive(Debug, Clone)]
pub struct LineStyle {
    pub color: Color,
    pub mode: ColorMode,
    pub width: f64,
    pub width_mode: LineWidthMode,
}

impl LineStyle {
    pub fn resolve_color(&self, random_seed: u64) -> Color {
        match self.mode {
            ColorMode::Normal => self.color,
            ColorMode::Random => Color::from_seed(random_seed),
        }
    }
}

impl Default for LineStyle {
    fn default() -> Self {
        Self {
            color: Color::YELLOW,
            mode: ColorMode::Normal,
            width: 2.0,
            width_mode: LineWidthMode::Pixels,
        }
    }
}

/// Style for polygon geometry (fill and optional outline).
#[derive(Debug, Clone, Default)]
pub struct PolygonStyle {
    /// `None` means no fill (transparent interior).
    pub fill: Option<ColorStyle>,
    /// `None` means no outline.
    pub outline: Option<LineStyle>,
}

impl PolygonStyle {
    pub fn filled(fill: ColorStyle) -> Self {
        Self {
            fill: Some(fill),
            outline: None,
        }
    }

    pub fn outlined(outline: LineStyle) -> Self {
        Self {
            fill: None,
            outline: Some(outline),
        }
    }
}

/// Combined style applied to all geometry types in a document or object.
#[derive(Debug, Clone)]
pub struct VectorStyle {
    pub line: LineStyle,
    pub polygon: PolygonStyle,
}

impl Default for VectorStyle {
    fn default() -> Self {
        Self {
            line: LineStyle::default(),
            polygon: PolygonStyle {
                fill: Some(ColorStyle::solid(Color::new(1.0, 1.0, 0.0, 0.4))),
                outline: Some(LineStyle::default()),
            },
        }
    }
}

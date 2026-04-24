//! Raster overlay types and collection management.

use std::sync::Arc;
use thiserror::Error;

use courtier::AssetAccessor;
use orkester::Task;

use crate::credit::Credit;

/// Coordinate system used by an overlay tile provider.
///
/// `GeographicProjection` (equirectangular, EPSG:4326) and
/// `WebMercatorProjection` (EPSG:3857, used by OSM, Google Maps, Bing Maps).
///
/// The projection determines how image pixel rows map to latitude.  For
/// Geographic overlays the mapping is linear; for WebMercator it follows
/// `y = atanh(sin(lat))` (= `ln(tan(\pi/4 + lat/2))`), so a naïve linear
/// interpolation in latitude space would produce visible distortion at
/// latitudes above ~30 degree and severe distortion above ~60 degree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OverlayProjection {
    /// Equirectangular / geographic - pixel rows are linear in latitude.
    /// Used by TMS, most WMS services, and EPSG:4326-based providers.
    #[default]
    Geographic,
    /// Web Mercator (EPSG:3857) - pixel rows are linear in Mercator Y.
    /// Used by OSM slippy tiles, Google Maps, Bing Maps, WMTS with EPSG:3857.
    WebMercator,
}

/// A single raster overlay tile - pixel data plus its geographic extent.
#[derive(Clone, Debug)]
pub struct RasterOverlayTile {
    /// RGBA pixel data, row-major from top-left.
    pub pixels: Arc<[u8]>,
    pub width: u32,
    pub height: u32,
    /// Geographic rectangle this overlay tile covers (always in geodetic
    /// radians regardless of [`projection`]).
    pub rectangle: terra::GlobeRectangle,
    /// Projection used to encode pixel rows within [`rectangle`].
    pub projection: OverlayProjection,
}

/// Failure modes for fetching or decoding a single raster overlay tile.
///
/// Provider implementations return these in the task payload rather than
/// panicking so that a bad URL or corrupt image on a single tile never
/// tears down the background worker thread.
#[derive(Debug, Error)]
pub enum TileFetchError {
    #[error("tile fetch error: {0}")]
    Fetch(#[from] courtier::FetchError),
    #[error("tile image decode failed: {0}")]
    Decode(Arc<str>),
}

/// Produces individual overlay tiles on demand.
///
/// Implementors fetch tiles from a URL template, WMS endpoint, etc.
pub trait RasterOverlayTileProvider: Send + Sync {
    /// Fetch the tile at the given tile coordinates.
    ///
    /// Returns a task yielding either the decoded tile or a [`TileFetchError`]
    /// describing why the fetch/decode failed. The overlay engine treats
    /// errors as "tile not available" and skips them during compositing.
    fn get_tile(
        &self,
        x: u32,
        y: u32,
        level: u32,
    ) -> Task<Result<RasterOverlayTile, TileFetchError>>;
    /// Geographic coverage of this provider.
    fn bounds(&self) -> terra::GlobeRectangle;
    /// Maximum available tile zoom level.
    fn maximum_level(&self) -> u32;
    /// Minimum available tile zoom level.
    fn minimum_level(&self) -> u32 {
        0
    }
    /// Projection used by this provider's tiles.
    ///
    /// Defaults to [`OverlayProjection::Geographic`].  Override for
    /// WebMercator tile sources (OSM, Bing, Google Maps, WMTS EPSG:3857).
    fn projection(&self) -> OverlayProjection {
        OverlayProjection::Geographic
    }

    /// Credits to display for this overlay's data source.
    /// Implementors should return appropriate attribution.
    fn credits(&self) -> Vec<Credit> {
        vec![]
    }

    /// Find all tile coordinates whose pixel coverage of `extent` most closely
    /// matches `target_screen_pixels` (X = projected-X span size, Y = projected-Y).
    ///
    /// Returns `(x, y, level)` tuples. Returning an empty `Vec` means no tile
    /// covers the given extent (e.g., the extent is outside the provider's
    /// [`bounds()`](Self::bounds)).
    ///
    /// The default implementation uses a power-of-two Web Mercator–style
    /// scheme and is suitable for providers whose levels double in resolution.
    /// Use [`get_tiles_for_extent`] in your implementation body.
    fn tiles_for_extent(
        &self,
        extent: terra::GlobeRectangle,
        target_screen_pixels: glam::DVec2,
    ) -> Vec<(u32, u32, u32)>;
}

/// An overlay data source that can create a tile provider.
///
/// Implement this for each overlay type (URL template, Cesium ion, WMS, …).
pub trait RasterOverlay: Send + Sync {
    /// Asynchronously construct the tile provider.
    ///
    /// Called once when the overlay is added to a `Stratum`. The provider
    /// is then used for the lifetime of the overlay.
    fn create_tile_provider(
        &self,
        context: &orkester::Context,
        accessor: &Arc<dyn AssetAccessor>,
    ) -> Task<Arc<dyn RasterOverlayTileProvider>>;
}

/// Opaque handle to an overlay added to an [`OverlayCollection`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OverlayId(pub u32);

/// Manages a set of raster overlays attached to a `Stratum`.
///
/// Overlays are added at runtime; tile providers are created asynchronously.
/// The collection drives attach/detach calls to the `OverlayablePreparer`.
#[derive(Default)]
pub struct OverlayCollection {
    overlays: Vec<OverlayEntry>,
    next_id: u32,
}

struct OverlayEntry {
    id: OverlayId,
    overlay: Box<dyn RasterOverlay>,
}

impl OverlayCollection {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an overlay. Returns an opaque `OverlayId` for later removal.
    pub fn add(&mut self, overlay: impl RasterOverlay + 'static) -> OverlayId {
        let id = OverlayId(self.next_id);
        self.next_id += 1;
        self.overlays.push(OverlayEntry {
            id,
            overlay: Box::new(overlay),
        });
        id
    }

    /// Remove an overlay by its id.
    pub fn remove(&mut self, id: OverlayId) {
        self.overlays.retain(|e| e.id != id);
    }

    /// Number of active overlays.
    pub fn len(&self) -> usize {
        self.overlays.len()
    }

    pub fn is_empty(&self) -> bool {
        self.overlays.is_empty()
    }

    /// Iterate over (id, overlay) pairs. Used by `OverlayStratum` to
    /// initialize tile providers when overlays are added.
    pub fn iter(&self) -> impl Iterator<Item = (OverlayId, &dyn RasterOverlay)> {
        self.overlays.iter().map(|e| (e.id, e.overlay.as_ref()))
    }
}

// Helper: geographic latitude -> Web Mercator Y angle, via terra's projection.
// This is `atanh(sin(lat))` clamped to +-MAXIMUM_LATITUDE.
#[inline]
fn lat_to_merc(lat: f64) -> f64 {
    terra::WebMercatorProjection::geodetic_latitude_to_mercator_angle(lat)
}

// Helper: Web Mercator Y angle -> geographic latitude.
#[inline]
fn merc_to_lat(m: f64) -> f64 {
    terra::WebMercatorProjection::mercator_angle_to_geodetic_latitude(m)
}

/// Default implementation of [`RasterOverlayTileProvider::tiles_for_extent`].
///
/// Mirrors `CesiumRasterOverlays::QuadtreeRasterOverlayTileProvider::computeLevelFromTargetScreenPixels`:
///
/// ```text
///   target_tile_dimensions = extent_dimensions / (target_screen_pixels / 256)
///   total_tile_dimensions  = provider_bounds_dimensions / 1   (root tiles = 1)
///   level = round(log2(total_tile_dimensions / target_tile_dimensions))
/// ```
///
/// Then iterates the resulting `level` tile grid intersecting the clamped
/// extent. Handles both Geographic (equirectangular) and WebMercator providers;
/// for WebMercator the Y axis is computed in Mercator angle space so the grid
/// matches OSM / slippy-map servers exactly.
pub fn get_tiles_for_extent(
    provider: &dyn RasterOverlayTileProvider,
    extent: terra::GlobeRectangle,
    target_screen_pixels: glam::DVec2,
) -> Vec<(u32, u32, u32)> {
    let provider_bounds = provider.bounds();
    let projection = provider.projection();

    // Clamp the query extent to the provider's coverage.
    let west = extent.west.max(provider_bounds.west);
    let east = extent.east.min(provider_bounds.east);
    let south = extent.south.max(provider_bounds.south);
    let north = extent.north.min(provider_bounds.north);
    if west >= east || south >= north {
        return vec![];
    }

    // Provider span in projected coordinates (X = lon, Y = native).
    let full_lon = (provider_bounds.east - provider_bounds.west)
        .abs()
        .max(f64::EPSILON);
    let (full_y_span, south_y, north_y) = match projection {
        OverlayProjection::Geographic => {
            let span = (provider_bounds.north - provider_bounds.south)
                .abs()
                .max(f64::EPSILON);
            (
                span,
                south - provider_bounds.south,
                north - provider_bounds.south,
            )
        }
        OverlayProjection::WebMercator => {
            let merc_pb_south = lat_to_merc(provider_bounds.south);
            let merc_pb_north = lat_to_merc(provider_bounds.north);
            let span = (merc_pb_north - merc_pb_south).abs().max(f64::EPSILON);
            (
                span,
                lat_to_merc(south) - merc_pb_south,
                lat_to_merc(north) - merc_pb_south,
            )
        }
    };

    // Extent dimensions in projected space.
    let extent_lon = (east - west).abs().max(f64::EPSILON);
    let extent_y = (north_y - south_y).abs().max(f64::EPSILON);

    // tile_pixels = 256, root_tiles = 1.
    //   raster_tiles            = target_screen_pixels / 256
    //   target_tile_dimensions  = (extent_lon, extent_y) / raster_tiles
    //   total_tile_dimensions   = (full_lon, full_y_span)
    //   level = round(log2(total_tile_dimensions / target_tile_dimensions))
    const TILE_PIXELS: f64 = 256.0;
    let raster_tiles_x = (target_screen_pixels.x / TILE_PIXELS).max(f64::EPSILON);
    let raster_tiles_y = (target_screen_pixels.y / TILE_PIXELS).max(f64::EPSILON);
    let target_tile_dim_x = extent_lon / raster_tiles_x;
    let target_tile_dim_y = extent_y / raster_tiles_y;
    let two_to_level_x = full_lon / target_tile_dim_x;
    let two_to_level_y = full_y_span / target_tile_dim_y;
    let level_x = two_to_level_x.max(1.0).log2();
    let level_y = two_to_level_y.max(1.0).log2();
    let level_f = level_x.max(level_y).round().max(0.0);

    let min_level = provider.minimum_level();
    let max_level = provider.maximum_level();
    let chosen_level = (level_f as u32).clamp(min_level, max_level);

    let x_tiles = 1u32.checked_shl(chosen_level).unwrap_or(u32::MAX);
    let y_tiles = 1u32.checked_shl(chosen_level).unwrap_or(u32::MAX);

    let tile_lon = full_lon / x_tiles as f64;
    let tile_y = full_y_span / y_tiles as f64;

    let x0 = ((west - provider_bounds.west) / tile_lon).floor() as u32;
    let x1 = ((east - provider_bounds.west) / tile_lon).ceil() as u32;
    let y0 = (south_y / tile_y).floor() as u32;
    let y1 = (north_y / tile_y).ceil() as u32;

    let x0 = x0.min(x_tiles - 1);
    let x1 = x1.min(x_tiles);
    let y0 = y0.min(y_tiles - 1);
    let y1 = y1.min(y_tiles);

    let mut out = Vec::with_capacity(((x1 - x0) * (y1 - y0)) as usize);
    for y in y0..y1 {
        for x in x0..x1 {
            out.push((x, y, chosen_level));
        }
    }
    out
}

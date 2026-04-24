//! `banin` — Cesium terrain tile builder.
//!
//! Converts a GeoTIFF DEM (any CRS with an EPSG code) into a Cesium-compatible
//! terrain tileset on disk, ready to be served as a `CesiumTerrainProvider`.
//!
//! Pure Rust — no GDAL or other C dependencies required.
//!
//! # Output layout
//!
//! ```text
//! <output>/
//!   layer.json          # tileset metadata consumed by CesiumJS
//!   <zoom>/
//!     <x>/
//!       <y>.terrain     # gzip-compressed tile bytes
//! ```
//!
//! # Serving
//!
//! Tiles are gzip-compressed.  Serve with `Content-Encoding: gzip` and
//! `Content-Type: application/octet-stream`.  Any static file server that
//! transparently decompresses `.gz` files (nginx, Caddy, terriajs-server)
//! will work out of the box.
//!
//! # Example
//!
//! ```text
//! banin --input dem.tif --output ./tiles --format quantized-mesh
//! banin --input dem.tif --output ./tiles --format heightmap --zoom-max 8
//! ```

use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use arazi::{
    ChildFlags, HeightmapTile, QuantizedMeshInput, TILE_CELL_SIZE, TILE_SIZE, WaterMask,
    encode_heightmap, encode_quantized_mesh,
};
use banin::tiler::{TerrainTiler, TileData, elevation_to_u16};
use clap::{Parser, ValueEnum};
use flate2::{Compression, write::GzEncoder};
use terra::Ellipsoid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum Format {
    /// Cesium `heightmap-1.0` (65x65 u16 heights, gzip-compressed)
    Heightmap,
    /// Cesium `quantized-mesh-1.0` (triangulated mesh, gzip-compressed)
    #[value(name = "quantized-mesh")]
    QuantizedMesh,
}

#[derive(Parser, Debug)]
#[command(
    name    = "banin",
    version,
    about   = "Convert a GeoTIFF DEM to a Cesium terrain tileset",
    long_about = None,
)]
struct Args {
    /// Input GeoTIFF DEM (must have an embedded EPSG code)
    #[arg(short, long, value_name = "FILE")]
    input: PathBuf,

    /// Output directory for tiles and layer.json
    #[arg(short, long, value_name = "DIR", default_value = "tiles")]
    output: PathBuf,

    /// Terrain format to generate
    #[arg(short, long, default_value = "quantized-mesh")]
    format: Format,

    /// Minimum zoom level [default: 0]
    #[arg(long, default_value_t = 0)]
    zoom_min: u32,

    /// Maximum zoom level [default: auto-detected from DEM resolution]
    #[arg(long)]
    zoom_max: Option<u32>,

    /// Height samples per tile side; heightmap format always uses 65
    #[arg(long, default_value_t = 65)]
    grid_size: usize,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // heightmap is always 65x65
    let grid_size = match args.format {
        Format::Heightmap => {
            if args.grid_size != 65 {
                eprintln!(
                    "warning: heightmap format always uses 65x65; ignoring --grid-size {}",
                    args.grid_size
                );
            }
            TILE_SIZE
        }
        Format::QuantizedMesh => args.grid_size,
    };

    let tiler = if grid_size != banin::tiler::DEFAULT_GRID_SIZE {
        TerrainTiler::open_with_grid_size(&args.input, grid_size)?
    } else {
        TerrainTiler::open(&args.input)?
    };

    let zoom_max = args.zoom_max.unwrap_or(tiler.max_zoom());
    if args.zoom_min > zoom_max {
        return Err(format!(
            "--zoom-min {} > --zoom-max {}; nothing to do",
            args.zoom_min, zoom_max
        )
        .into());
    }

    let ellipsoid = Ellipsoid::wgs84();
    let format_name = match args.format {
        Format::Heightmap => "heightmap-1.0",
        Format::QuantizedMesh => "quantized-mesh-1.0",
    };

    fs::create_dir_all(&args.output)?;

    println!("Input  : {}", args.input.display());
    println!("Output : {}", args.output.display());
    println!("Format : {format_name}");
    println!("Zoom   : {}..={zoom_max}", args.zoom_min);
    println!();

    let mut tiles_written: u64 = 0;
    let mut errors: u64 = 0;
    let stdout = io::stdout();
    // Track tile coords per zoom for the `available` field in layer.json.
    let mut available: std::collections::BTreeMap<u32, (u64, u64, u64, u64)> =
        std::collections::BTreeMap::new();

    for result in tiler.tiles(args.zoom_min..=zoom_max) {
        match result {
            Err(e) => {
                eprintln!("\nerror: {e}");
                errors += 1;
            }
            Ok(tile) => {
                let dir = args
                    .output
                    .join(tile.id.level.to_string())
                    .join(tile.id.x.to_string());
                fs::create_dir_all(&dir)?;

                let path = dir.join(format!("{}.terrain", tile.id.y));
                let raw = encode_tile(&tile, args.format, zoom_max, &ellipsoid);
                let compressed = gzip(&raw)?;
                fs::write(&path, &compressed)?;

                // Expand the bounding box for this zoom level.
                let entry = available.entry(tile.id.level).or_insert((
                    tile.id.x as u64,
                    tile.id.x as u64,
                    tile.id.y as u64,
                    tile.id.y as u64,
                ));
                entry.0 = entry.0.min(tile.id.x as u64);
                entry.1 = entry.1.max(tile.id.x as u64);
                entry.2 = entry.2.min(tile.id.y as u64);
                entry.3 = entry.3.max(tile.id.y as u64);

                tiles_written += 1;
                {
                    let mut out = stdout.lock();
                    write!(
                        out,
                        "\r  zoom {}/{zoom_max}  tile {}/{}/{}  ({tiles_written} written)",
                        tile.id.level, tile.id.level, tile.id.x, tile.id.y
                    )?;
                    out.flush()?;
                }
            }
        }
    }

    println!("\n\nDone: {tiles_written} tiles written, {errors} errors.");
    write_layer_json(&args, zoom_max, &tiler, format_name, &available)?;
    println!("layer.json → {}", args.output.join("layer.json").display());

    if errors > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn encode_tile(tile: &TileData, format: Format, zoom_max: u32, ellipsoid: &Ellipsoid) -> Vec<u8> {
    match format {
        Format::QuantizedMesh => encode_quantized_mesh(&QuantizedMeshInput {
            heights: &tile.heights,
            grid_size: tile.grid_size,
            west: tile.bounds.west,
            south: tile.bounds.south,
            east: tile.bounds.east,
            north: tile.bounds.north,
            ellipsoid,
        }),
        Format::Heightmap => {
            let mut heights = [0u16; TILE_CELL_SIZE];
            for (i, &h) in tile.heights.iter().enumerate().take(TILE_CELL_SIZE) {
                heights[i] = elevation_to_u16(h);
            }
            let children = if tile.id.level < zoom_max {
                ChildFlags::ALL
            } else {
                ChildFlags::NONE
            };
            encode_heightmap(&HeightmapTile {
                heights,
                children,
                water_mask: WaterMask::Land,
            })
        }
    }
}

fn gzip(data: &[u8]) -> io::Result<Vec<u8>> {
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(data)?;
    enc.finish()
}

fn write_layer_json(
    args: &Args,
    zoom_max: u32,
    tiler: &TerrainTiler,
    format_name: &str,
    available: &std::collections::BTreeMap<u32, (u64, u64, u64, u64)>,
) -> Result<(), Box<dyn std::error::Error>> {
    let (west, south, east, north) = tiler.bounds_deg();

    // Build the `available` JSON array, one entry per zoom level.
    // Each entry is a list of tile ranges: [{startX, endX, startY, endY}].
    let available_str = if available.is_empty() {
        "[]".to_string()
    } else {
        let levels: Vec<String> = (args.zoom_min..=zoom_max)
            .map(|z| {
                if let Some(&(min_x, max_x, min_y, max_y)) = available.get(&z) {
                    format!(
                        r#"[{{"startX":{min_x},"endX":{max_x},"startY":{min_y},"endY":{max_y}}}]"#
                    )
                } else {
                    "[]".to_string()
                }
            })
            .collect();
        format!("[\n    {}\n  ]", levels.join(",\n    "))
    };

    let json = format!(
        r#"{{
  "tilejson": "2.1.0",
  "format": "{format_name}",
  "version": "1.0.0",
  "scheme": "tms",
  "tiles": ["{{z}}/{{x}}/{{y}}.terrain"],
  "projection": "EPSG:4326",
  "bounds": [{west:.6}, {south:.6}, {east:.6}, {north:.6}],
  "minzoom": {minzoom},
  "maxzoom": {maxzoom},
  "available": {available_str}
}}"#,
        minzoom = args.zoom_min,
        maxzoom = zoom_max,
    );
    let path = args.output.join("layer.json");
    fs::write(&path, json)?;
    Ok(())
}

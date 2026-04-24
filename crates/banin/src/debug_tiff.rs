use oxigdal_core::io::FileDataSource;
use oxigdal_geotiff::CogReader;

fn main() {
    let source = FileDataSource::open("N45E006.tif").unwrap();
    let reader = CogReader::open(source).unwrap();

    let info = reader.primary_info();
    println!("size: {}x{}", info.width, info.height);
    println!("data_type: {:?}", info.data_type());
    println!("compression: {:?}", info.compression);
    println!("predictor: {:?}", info.predictor);
    println!("tile: {:?}x{:?}", info.tile_width, info.tile_height);
    println!("bps: {:?}", info.bits_per_sample);
    println!("spp: {}", info.samples_per_pixel);

    // Read tile 0,0 and print first 8 bytes and first few f32 values
    let raw = reader.read_tile(0, 0, 0).unwrap();
    println!("decompressed tile[0,0]: {} bytes", raw.len());

    // Print first 8 floats as both raw bytes and f32
    for i in 0..8 {
        let offset = i * 4;
        let bytes = &raw[offset..offset + 4];
        let v = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        println!(
            "  pixel[{}]: bytes={:02x}{:02x}{:02x}{:02x}  f32_le={}",
            i, bytes[0], bytes[1], bytes[2], bytes[3], v
        );
    }

    // Also check nodata
    println!("nodata: {:?}", reader.nodata());
    println!("epsg: {:?}", reader.epsg_code());
}

use ndarray_vision::core::{Gray, Image};
use std::fs::File;

pub fn load_lena() -> Image<u8, Gray> {
    let dec = png::Decoder::new(File::open("data/lena.png").unwrap());
    let (info, mut reader) = dec.read_info().unwrap();
    let mut buf = vec![0; info.buffer_size()];
    reader.next_frame(&mut buf).unwrap();

    assert_eq!(
        info.bit_depth,
        png::BitDepth::Eight,
        "Bit depth doesn't match"
    );
    assert_eq!(
        info.color_type,
        png::ColorType::Grayscale,
        "Colour format is wrong"
    );

    Image::<u8, Gray>::from_shape_data(info.height as usize, info.width as usize, buf)
}

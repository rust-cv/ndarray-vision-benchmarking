use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray_vision::core::{Image, Gray};
use ndarray_vision::processing::*;
use std::fs::File;
use image::open;

pub fn ndarray_vision_bench(c: &mut Criterion) {
    // gaussian sigma in imageproc is default 1.4 in a 7x7 kernel
    // runs on u8 grayscale image

    let canny_params = CannyBuilder::<f32>::new()
        .lower_threshold(1.0/255.0)
        .upper_threshold(30.0/255.0)
        .blur((7, 7), [1.4, 1.4])
        .build();

    let dec = png::Decoder::new(File::open("data/lena.png").unwrap());
    let (info, mut reader) = dec.read_info().unwrap();
    let mut buf = vec![0; info.buffer_size()];
    reader.next_frame(&mut buf).unwrap();

    assert_eq!(info.bit_depth, png::BitDepth::Eight, "Bit depth doesn't match");
    assert_eq!(info.color_type, png::ColorType::Grayscale, "Colour format is wrong");

    let image = Image::<u8, Gray>::from_shape_data(info.height as usize, info.width as usize, buf);
    let image: Image::<f32, Gray> = image.into_type();
    c.bench_function("canny", |b| b.iter(|| image.canny_edge_detector(black_box(canny_params.clone()))));

}

pub fn imageproc_bench(c: &mut Criterion) {
    let image = open("data/lena.png").unwrap().to_luma();
    c.bench_function("canny", |b| b.iter(|| imageproc::edges::canny(black_box(&image), black_box(1.0), black_box(30.0))));
}

criterion_group!(benches, ndarray_vision_bench, imageproc_bench);
criterion_main!(benches);

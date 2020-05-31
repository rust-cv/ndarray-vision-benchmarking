use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::open;
use ndarray_vision::core::{Gray, Image};
use ndarray_vision::processing::*;
use ndarray_vision_benchmarking::load_lena;

pub fn ndarray_vision_bench(c: &mut Criterion) {
    // gaussian sigma in imageproc is default 1.4 in a 7x7 kernel
    // runs on u8 grayscale image

    let canny_params = CannyBuilder::<f32>::new()
        .lower_threshold(1.0 / 255.0)
        .upper_threshold(30.0 / 255.0)
        .blur((7, 7), [1.4, 1.4])
        .build();

    let image = load_lena();

    let image: Image<f32, Gray> = image.into_type();
    c.bench_function("canny_ndarray_vision", |b| {
        b.iter(|| image.canny_edge_detector(black_box(canny_params.clone())))
    });
}

pub fn imageproc_bench(c: &mut Criterion) {
    let image = open("data/lena.png").unwrap().to_luma();
    c.bench_function("canny_imageproc", |b| {
        b.iter(|| imageproc::edges::canny(black_box(&image), black_box(1.0), black_box(30.0)))
    });
}

criterion_group!(benches, imageproc_bench, ndarray_vision_bench);
criterion_main!(benches);

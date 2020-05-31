use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::open;
use ndarray_vision::core::{Gray, Image};
use ndarray_vision::processing::*;
use ndarray_vision_benchmarking::load_lena;

pub fn ndarray_vision_bench(c: &mut Criterion) {
    let image: Image<f32, Gray> = load_lena().into_type();

    c.bench_function("sobel_ndarray_vision", |b| b.iter(|| image.apply_sobel()));
}

pub fn imageproc_bench(c: &mut Criterion) {
    let image = open("data/lena.png").unwrap().to_luma();
    c.bench_function("sobel_imageproc", |b| {
        b.iter(|| imageproc::gradients::sobel_gradients(black_box(&image)))
    });
}

criterion_group!(benches, imageproc_bench, ndarray_vision_bench);
criterion_main!(benches);

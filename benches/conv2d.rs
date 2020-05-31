use criterion::{black_box, criterion_group, criterion_main, Criterion};
use image::open;
use ndarray_vision::core::{Gray, Image, ZeroPadding};
use ndarray_vision_benchmarking::load_lena;
use ndarray_vision::processing::*;
use std::fs::File;
use ndarray::arr3;

pub fn ndarray_vision_bench(c: &mut Criterion) {
    let image = load_lena();

    let kernel = arr3(
        &[
            [[1], [0], [1]],
            [[0], [1], [0]],
            [[1], [0], [1]]
        ]);

    let padding = ZeroPadding{};

    c.bench_function("conv2d_3x3_ndarray_vision", |b| {
        b.iter(|| image.conv2d_with_padding(black_box(kernel.view()), black_box(&padding)))
    });
}

pub fn imageproc_bench(c: &mut Criterion) {
    let image = open("data/lena.png").unwrap().to_luma();
    let weights: Vec<u8> = vec![1,0,1,0,1,0,1,0,1];
    c.bench_function("conv2d+3x3_imageproc", |b| {
        b.iter(|| imageproc::filter::filter3x3::<_, u8, u8>(black_box(&image), black_box(&weights)))
    });
}

criterion_group!(benches, imageproc_bench, ndarray_vision_bench);
criterion_main!(benches);


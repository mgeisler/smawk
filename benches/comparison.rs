#![cfg(feature = "ndarray")]

use divan::Bencher;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[path = "../tests/random_monge/mod.rs"]
mod random_monge;
use random_monge::random_monge_matrix;

fn random_matrix(size: usize) -> Array2<i32> {
    let mut rng = StdRng::seed_from_u64(0);
    random_monge_matrix(size, size, &mut rng)
}

const SIZES: [usize; 5] = [25, 50, 100, 200, 400];

// Brute Force

#[divan::bench(name = "brute_force_row_minima", consts = SIZES)]
fn brute_force_row_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    bencher.bench_local(|| smawk::brute_force::row_minima(&matrix));
}

#[divan::bench(name = "brute_force_column_minima", consts = SIZES)]
fn brute_force_column_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    let transposed = matrix.reversed_axes();
    bencher.bench_local(|| smawk::brute_force::column_minima(&transposed));
}

// Recursive

#[divan::bench(name = "recursive_row_minima", consts = SIZES)]
fn recursive_row_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    bencher.bench_local(|| smawk::recursive::row_minima(&matrix));
}

#[divan::bench(name = "recursive_column_minima", consts = SIZES)]
fn recursive_column_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    let transposed = matrix.reversed_axes();
    bencher.bench_local(|| smawk::recursive::column_minima(&transposed));
}

// SMAWK

#[divan::bench(name = "smawk_row_minima", consts = SIZES)]
fn smawk_row_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    bencher.bench_local(|| smawk::row_minima(&matrix));
}

#[divan::bench(name = "smawk_column_minima", consts = SIZES)]
fn smawk_column_minima<const N: usize>(bencher: Bencher) {
    let matrix = random_matrix(N);
    let transposed = matrix.reversed_axes();
    bencher.bench_local(|| smawk::column_minima(&transposed));
}

fn main() {
    divan::main();
}

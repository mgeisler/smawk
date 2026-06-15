#![cfg(feature = "ndarray")]

use divan::{AllocProfiler, Bencher};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[global_allocator]
static ALLOC: AllocProfiler = AllocProfiler::system();

#[path = "../tests/random_monge/mod.rs"]
mod random_monge;
use random_monge::random_monge_matrix;

fn random_matrix(size: usize) -> Array2<i32> {
    let mut rng = StdRng::seed_from_u64(0);
    random_monge_matrix(size, size, &mut rng)
}

const SIZES: [usize; 5] = [50, 100, 200, 400, 800];

#[divan::bench(name = "smawk_online_column_minima", consts = SIZES)]
fn smawk_online_column_minima<const N: usize>(bencher: Bencher) {
    let mut matrix = random_matrix(N);
    let max = *matrix.iter().max().unwrap_or(&0);
    for idx in 0..N {
        matrix
            .slice_mut(ndarray::s![idx..idx + 1, ..idx + 1])
            .fill(max);
    }
    let initial = 42;
    matrix.slice_mut(ndarray::s![0.., ..1]).fill(initial);
    bencher.bench_local(|| smawk::online_column_minima(initial, N, |_, i, j| matrix[[i, j]]));
}

fn main() {
    divan::main();
}

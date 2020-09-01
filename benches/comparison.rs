#![cfg(feature = "ndarray")]
#![feature(test)]

extern crate test;

use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use test::Bencher;

#[path = "../tests/random_monge/mod.rs"]
mod random_monge;
use random_monge::random_monge_matrix;

macro_rules! repeat {
    ([ $( ($row_bench:ident, $column_bench:ident, $size:expr) $(,)* )* ],
     $row_func:path, $column_func:path) => {
        $(
            #[bench]
            fn $row_bench(b: &mut Bencher) {
                let mut rng = ChaCha20Rng::seed_from_u64(0);
                let matrix: Array2<i32> = random_monge_matrix($size, $size, &mut rng);
                b.iter(|| $row_func(&matrix));
            }

            #[bench]
            fn $column_bench(b: &mut Bencher) {
                let mut rng = ChaCha20Rng::seed_from_u64(0);
                let matrix: Array2<i32> = random_monge_matrix($size, $size, &mut rng).reversed_axes();
                b.iter(|| $column_func(&matrix));
            }
        )*
    };
}

repeat!(
    [
        (row_brute_force_025, column_brute_force_025, 25),
        (row_brute_force_050, column_brute_force_050, 50),
        (row_brute_force_100, column_brute_force_100, 100),
        (row_brute_force_200, column_brute_force_200, 200),
        (row_brute_force_400, column_brute_force_400, 400)
    ],
    smawk::brute_force::row_minima,
    smawk::brute_force::column_minima
);

repeat!(
    [
        (row_recursive_025, column_recursive_025, 25),
        (row_recursive_050, column_recursive_050, 50),
        (row_recursive_100, column_recursive_100, 100),
        (row_recursive_200, column_recursive_200, 200),
        (row_recursive_400, column_recursive_400, 400)
    ],
    smawk::recursive::row_minima,
    smawk::recursive::column_minima
);

repeat!(
    [
        (row_smawk_025, column_smawk_025, 25),
        (row_smawk_050, column_smawk_050, 50),
        (row_smawk_100, column_smawk_100, 100),
        (row_smawk_200, column_smawk_200, 200),
        (row_smawk_400, column_smawk_400, 400)
    ],
    smawk::smawk_row_minima,
    smawk::smawk_column_minima
);

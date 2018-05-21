#![feature(test)]

extern crate ndarray;
extern crate rand;
extern crate smawk;
extern crate test;

use rand::XorShiftRng;
use test::Bencher;

macro_rules! repeat {
    ([ $( ($row_bench:ident, $column_bench:ident, $size:expr) $(,)* )* ],
     $row_func:path, $column_func:path) => {
        $(
            #[bench]
            fn $row_bench(b: &mut Bencher) {
                let mut rng = XorShiftRng::new_unseeded();
                let matrix = smawk::random_monge_matrix($size, $size, &mut rng);
                b.iter(|| $row_func(&matrix));
            }

            #[bench]
            fn $column_bench(b: &mut Bencher) {
                let mut rng = XorShiftRng::new_unseeded();
                let matrix = smawk::random_monge_matrix($size, $size, &mut rng).reversed_axes();
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
    smawk::brute_force_row_minima,
    smawk::brute_force_column_minima
);

repeat!(
    [
        (row_recursive_025, column_recursive_025, 25),
        (row_recursive_050, column_recursive_050, 50),
        (row_recursive_100, column_recursive_100, 100),
        (row_recursive_200, column_recursive_200, 200),
        (row_recursive_400, column_recursive_400, 400)
    ],
    smawk::recursive_row_minima,
    smawk::recursive_column_minima
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

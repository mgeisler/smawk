#![feature(test)]

extern crate ndarray;
extern crate rand;
extern crate smawk;
extern crate test;

use rand::XorShiftRng;
use test::Bencher;

macro_rules! repeat {
    ([ $( ($func:ident, $size:expr) $(,)* )* ], $row_func:path) => {
        $(
            #[bench]
            fn $func(b: &mut Bencher) {
                let mut rng = XorShiftRng::new_unseeded();
                let matrix = smawk::random_monge_matrix($size, $size, &mut rng);
                b.iter(|| $row_func(&matrix));
            }
        )*
    };
}

repeat!([(brute_force_025, 25),
         (brute_force_050, 50),
         (brute_force_100, 100),
         (brute_force_200, 200),
         (brute_force_400, 400)],
        smawk::brute_force_row_minima);

repeat!([(recursive_0025, 25),
         (recursive_0050, 50),
         (recursive_0100, 100),
         (recursive_0200, 200),
         (recursive_0400, 400),
         (recursive_0800, 800)],
        smawk::recursive_row_minima);

repeat!([(smawk_025, 25),
         (smawk_050, 50),
         (smawk_100, 100),
         (smawk_200, 200),
         (smawk_400, 400),
         (smawk_800, 800)],
        smawk::smawk_row_minima);

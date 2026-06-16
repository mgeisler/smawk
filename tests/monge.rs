#![cfg(feature = "ndarray")]

use ndarray::{arr2, Array, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rstest::rstest;
use smawk::monge::is_monge;

mod random_monge;
use random_monge::{random_monge_matrix, MongePrim};

#[test]
fn random_monge() {
    let mut rng = StdRng::seed_from_u64(0);
    let matrix: Array2<u8> = random_monge_matrix(5, 5, &mut rng);

    assert!(is_monge(&matrix));
    assert_eq!(
        matrix,
        arr2(&[
            [3, 6, 5, 6, 8],
            [3, 6, 5, 5, 7],
            [4, 7, 6, 6, 8],
            [3, 5, 4, 4, 6],
            [2, 4, 2, 2, 4],
        ])
    );
}

#[test]
fn monge_constant_rows() {
    let mut rng = StdRng::seed_from_u64(0);
    let matrix: Array2<u8> = MongePrim::ConstantRows.to_matrix(5, 4, &mut rng);
    assert!(is_monge(&matrix));
    for row in matrix.rows() {
        let elem = row[0];
        assert_eq!(row, Array::from_elem(matrix.ncols(), elem));
    }
}

#[test]
fn monge_constant_cols() {
    let mut rng = StdRng::seed_from_u64(0);
    let matrix: Array2<u8> = MongePrim::ConstantCols.to_matrix(5, 4, &mut rng);
    assert!(is_monge(&matrix));
    for column in matrix.columns() {
        let elem = column[0];
        assert_eq!(column, Array::from_elem(matrix.nrows(), elem));
    }
}

#[test]
fn monge_upper_right_ones() {
    let mut rng = StdRng::seed_from_u64(0);
    let matrix: Array2<u8> = MongePrim::UpperRightOnes.to_matrix(5, 4, &mut rng);
    assert!(is_monge(&matrix));
    assert_eq!(
        matrix,
        arr2(&[
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
        ])
    );
}

#[test]
fn monge_lower_left_ones() {
    let mut rng = StdRng::seed_from_u64(0);
    let matrix: Array2<u8> = MongePrim::LowerLeftOnes.to_matrix(5, 4, &mut rng);
    assert!(is_monge(&matrix));
    assert_eq!(
        matrix,
        arr2(&[
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ])
    );
}

#[rstest]
fn many_random_monge_matrix_is_monge(
    // Generate 100 different seeds.
    #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] seed_a: u64,
    #[values(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)] seed_b: u64,
) {
    let seed = seed_a * 10 + seed_b;
    let mut rng = StdRng::seed_from_u64(seed);
    let m = rng.random_range(1..50);
    let n = rng.random_range(1..50);
    let matrix: Array2<i32> = random_monge_matrix(m, n, &mut rng);
    assert!(is_monge(&matrix), "m={m}, n={n}");
}

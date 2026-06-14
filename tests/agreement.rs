#![cfg(feature = "ndarray")]

use ndarray::{s, Array2};
use rand::rngs::StdRng;
use rand::SeedableRng;
use smawk::{brute_force, online_column_minima, recursive};

mod random_monge;
use random_monge::random_monge_matrix;

/// Check that the brute force, recursive, and SMAWK functions
/// give identical results on a large number of randomly generated
/// Monge matrices.
#[test]
fn column_minima_agree() {
    let sizes = [1, 2, 3, 4, 5, 10, 15, 20, 30];
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..4 {
        for &m in &sizes {
            for &n in &sizes {
                let matrix: Array2<i32> = random_monge_matrix(m, n, &mut rng);

                // Compute and test row minima.
                let brute_force = brute_force::row_minima(&matrix);
                let recursive = recursive::row_minima(&matrix);
                let smawk = smawk::row_minima(&matrix);
                assert_eq!(
                    brute_force, recursive,
                    "recursive and brute force differs on:\n{:?}",
                    matrix
                );
                assert_eq!(
                    brute_force, smawk,
                    "SMAWK and brute force differs on:\n{:?}",
                    matrix
                );

                // Do the same for the column minima.
                let brute_force = brute_force::column_minima(&matrix);
                let recursive = recursive::column_minima(&matrix);
                let smawk = smawk::column_minima(&matrix);
                assert_eq!(
                    brute_force, recursive,
                    "recursive and brute force differs on:\n{:?}",
                    matrix
                );
                assert_eq!(
                    brute_force, smawk,
                    "SMAWK and brute force differs on:\n{:?}",
                    matrix
                );
            }
        }
    }
}

/// Check that the brute force and online SMAWK functions give
/// identical results on a large number of randomly generated
/// Monge matrices.
#[test]
fn online_agree() {
    let sizes = [1, 2, 3, 4, 5, 10, 15, 20, 30, 50];
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..5 {
        for &size in &sizes {
            // Random totally monotone square matrix of the
            // desired size.
            let mut matrix: Array2<i32> = random_monge_matrix(size, size, &mut rng);

            // Adjust matrix so the column minima are above the
            // diagonal. The brute_force::column_minima will still
            // work just fine on such a mangled Monge matrix.
            let max = *matrix.iter().max().unwrap_or(&0);
            for idx in 0..(size as isize) {
                // Using the maximum value of the matrix instead
                // of i32::max_value() makes for prettier matrices
                // in case we want to print them.
                matrix.slice_mut(s![idx..idx + 1, ..idx + 1]).fill(max);
            }

            // The online algorithm always returns the initial
            // value for the left-most column -- without
            // inspecting the column at all. So we fill the
            // left-most column with this value to have the brute
            // force algorithm do the same.
            let initial = 42;
            matrix.slice_mut(s![0.., ..1]).fill(initial);

            // Brute-force computation of column minima, returned
            // in the same form as online_column_minima.
            let brute_force = brute_force::column_minima(&matrix)
                .iter()
                .enumerate()
                .map(|(j, &i)| (i, matrix[[i, j]]))
                .collect::<Vec<_>>();
            let online = online_column_minima(initial, size, |_, i, j| matrix[[i, j]]);
            assert_eq!(
                brute_force, online,
                "brute force and online differ on:\n{:3?}",
                matrix
            );
        }
    }
}

/// Check that the algorithms agree on extremely unbalanced
/// rectangular matrices.
#[test]
fn unbalanced_matrices_agree() {
    let mut rng = StdRng::seed_from_u64(42);
    let cases = [(1, 100), (100, 1), (2, 50), (50, 2), (5, 80), (80, 5)];
    for &(m, n) in &cases {
        let matrix: Array2<i32> = random_monge_matrix(m, n, &mut rng);

        // Compute and test row minima.
        let brute_force = brute_force::row_minima(&matrix);
        let recursive = recursive::row_minima(&matrix);
        let smawk = smawk::row_minima(&matrix);
        assert_eq!(brute_force, recursive);
        assert_eq!(brute_force, smawk);

        // Do the same for the column minima.
        let brute_force = brute_force::column_minima(&matrix);
        let recursive = recursive::column_minima(&matrix);
        let smawk = smawk::column_minima(&matrix);
        assert_eq!(brute_force, recursive);
        assert_eq!(brute_force, smawk);
    }
}

/// Check that all three algorithms agree on handling empty 0x0
/// matrices correctly.
#[test]
fn empty_matrices_agree() {
    let matrix: Array2<i32> = Array2::from_elem((0, 0), 0);
    let brute_force_row = brute_force::row_minima(&matrix);
    let recursive_row = recursive::row_minima(&matrix);
    let smawk_row = smawk::row_minima(&matrix);
    assert_eq!(brute_force_row, recursive_row);
    assert_eq!(brute_force_row, smawk_row);
    assert_eq!(brute_force_row, vec![]);

    let brute_force_col = brute_force::column_minima(&matrix);
    let recursive_col = recursive::column_minima(&matrix);
    let smawk_col = smawk::column_minima(&matrix);
    assert_eq!(brute_force_col, recursive_col);
    assert_eq!(brute_force_col, smawk_col);
    assert_eq!(brute_force_col, vec![]);
}

/// Check that smawk::row_minima panics when ncols is 0.
#[test]
#[should_panic]
fn smawk_row_minima_empty_cols_panics() {
    let matrix: Array2<i32> = Array2::from_elem((5, 0), 0);
    smawk::row_minima(&matrix);
}

/// Check that brute_force::row_minima panics when ncols is 0.
#[test]
#[should_panic]
fn brute_force_row_minima_empty_cols_panics() {
    let matrix: Array2<i32> = Array2::from_elem((5, 0), 0);
    brute_force::row_minima(&matrix);
}

/// Check that recursive::row_minima panics when ncols is 0.
#[test]
#[should_panic]
fn recursive_row_minima_empty_cols_panics() {
    let matrix: Array2<i32> = Array2::from_elem((5, 0), 0);
    recursive::row_minima(&matrix);
}

/// Check that smawk::column_minima panics when nrows is 0.
#[test]
#[should_panic]
fn smawk_column_minima_empty_rows_panics() {
    let matrix: Array2<i32> = Array2::from_elem((0, 5), 0);
    smawk::column_minima(&matrix);
}

/// Check that brute_force::column_minima panics when nrows is 0.
#[test]
#[should_panic]
fn brute_force_column_minima_empty_rows_panics() {
    let matrix: Array2<i32> = Array2::from_elem((0, 5), 0);
    brute_force::column_minima(&matrix);
}

/// Check that recursive::column_minima panics when nrows is 0.
#[test]
#[should_panic]
fn recursive_column_minima_empty_rows_panics() {
    let matrix: Array2<i32> = Array2::from_elem((0, 5), 0);
    recursive::column_minima(&matrix);
}

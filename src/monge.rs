//! Functions for generating and checking Monge arrays.
//!
//! The functions here are mostly meant to be used for testing
//! correctness of the SMAWK implementation.
//!
//! **Note: this module is only available if you enable the
//! `num-traits` Cargo feature.**

use crate::Matrix;
use num_traits::ops::overflowing::OverflowingAdd;
use num_traits::PrimInt;

/// Verify that a matrix is a Monge matrix.
///
/// A [Monge matrix] \(or array) is a matrix where the following
/// inequality holds:
///
/// ```text
/// M[i, j] + M[i', j'] <= M[i, j'] + M[i', j]  for all i < i', j < j'
/// ```
///
/// The inequality says that the sum of the main diagonal is less than
/// the sum of the antidiagonal. Checking this condition is done by
/// checking *n* × *m* submatrices, so the running time is O(*mn*).
///
/// [Monge matrix]: https://en.wikipedia.org/wiki/Monge_array
pub fn is_monge<T: PrimInt + OverflowingAdd, M: Matrix<T>>(matrix: &M) -> bool {
    (0..matrix.nrows() - 1)
        .flat_map(|row| (0..matrix.ncols() - 1).map(move |col| (row, col)))
        .all(|(row, col)| {
            let top_left = matrix.index(row, col);
            let top_right = matrix.index(row, col + 1);
            let bot_left = matrix.index(row + 1, col);
            let bot_right = matrix.index(row + 1, col + 1);

            let (main_sum, main_overflow) = top_left.overflowing_add(&bot_right);
            let (anti_sum, anti_overflow) = top_right.overflowing_add(&bot_left);
            match (main_overflow, anti_overflow) {
                (false, false) => main_sum <= anti_sum, // No overflow.
                (true, true) => main_sum <= anti_sum,   // Double overflow.
                (false, true) => true,                  // Anti-diagonal overflow.
                (true, false) => false,                 // Main diagonal overflow.
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_monge_handles_overflow() {
        // The x + y <= z + w computations will overflow for an u8
        // matrix unless is_monge is careful.
        let matrix: Vec<Vec<u8>> = vec![
            vec![200, 200, 200, 200],
            vec![200, 200, 200, 200],
            vec![200, 200, 200, 200],
        ];
        assert!(is_monge(&matrix));
    }

    #[test]
    fn monge_constant_rows() {
        let matrix = vec![
            vec![42, 42, 42, 42],
            vec![0, 0, 0, 0],
            vec![100, 100, 100, 100],
            vec![1000, 1000, 1000, 1000],
        ];
        assert!(is_monge(&matrix));
    }

    #[test]
    fn monge_constant_cols() {
        let matrix = vec![
            vec![42, 0, 100, 1000],
            vec![42, 0, 100, 1000],
            vec![42, 0, 100, 1000],
            vec![42, 0, 100, 1000],
        ];
        assert!(is_monge(&matrix));
    }

    #[test]
    fn monge_upper_right() {
        let matrix = vec![
            vec![10, 10, 42, 42, 42],
            vec![10, 10, 42, 42, 42],
            vec![10, 10, 10, 10, 10],
            vec![10, 10, 10, 10, 10],
        ];
        assert!(is_monge(&matrix));
    }

    #[test]
    fn monge_lower_left() {
        let matrix = vec![
            vec![10, 10, 10, 10, 10],
            vec![10, 10, 10, 10, 10],
            vec![42, 42, 42, 10, 10],
            vec![42, 42, 42, 10, 10],
        ];
        assert!(is_monge(&matrix));
    }
}

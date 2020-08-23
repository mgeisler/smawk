//! Functions for generating and checking Monge arrays.
//!
//! The functions here are mostly meant to be used for testing
//! correctness of the SMAWK implementation.
//!
//! **Note: this module is only available if you enable the `ndarray`
//! Cargo feature.**

use crate::Matrix;
use ndarray::{s, Array2};
use num_traits::PrimInt;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use std::num::Wrapping;
use std::ops::Add;

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
/// checking *n* ✕ *m* submatrices, so the running time is O(*mn*).
///
/// [Monge matrix]: https://en.wikipedia.org/wiki/Monge_array
pub fn is_monge<T: Ord + Copy, M: Matrix<T>>(matrix: &M) -> bool
where
    Wrapping<T>: Add<Output = Wrapping<T>>,
{
    /// Returns `Ok(a + b)` if the computation can be done without
    /// overflow, otherwise `Err(a + b - T::MAX - 1)` is returned.
    fn checked_add<T: Ord + Copy>(a: Wrapping<T>, b: Wrapping<T>) -> Result<T, T>
    where
        Wrapping<T>: Add<Output = Wrapping<T>>,
    {
        let sum = a + b;
        if sum < a {
            Err(sum.0)
        } else {
            Ok(sum.0)
        }
    }

    (0..matrix.nrows() - 1)
        .flat_map(|row| (0..matrix.ncols() - 1).map(move |col| (row, col)))
        .all(|(row, col)| {
            let top_left = Wrapping(matrix.index(row, col));
            let top_right = Wrapping(matrix.index(row, col + 1));
            let bot_left = Wrapping(matrix.index(row + 1, col));
            let bot_right = Wrapping(matrix.index(row + 1, col + 1));

            match (
                checked_add(top_left, bot_right),
                checked_add(bot_left, top_right),
            ) {
                (Ok(a), Ok(b)) => a <= b,   // No overflow.
                (Err(a), Err(b)) => a <= b, // Double overflow.
                (Ok(_), Err(_)) => true,    // Antidiagonal overflow.
                (Err(_), Ok(_)) => false,   // Main diagonal overflow.
            }
        })
}

/// A Monge matrix can be decomposed into one of these primitive
/// building blocks.
#[derive(Copy, Clone)]
enum MongePrim {
    ConstantRows,
    ConstantCols,
    UpperRightOnes,
    LowerLeftOnes,
}

impl MongePrim {
    /// Generate a Monge matrix from a primitive.
    fn to_matrix<T: PrimInt, R: Rng>(&self, m: usize, n: usize, rng: &mut R) -> Array2<T>
    where
        Standard: Distribution<T>,
    {
        let mut matrix = Array2::from_elem((m, n), T::zero());
        // Avoid panic in UpperRightOnes and LowerLeftOnes below.
        if m == 0 || n == 0 {
            return matrix;
        }

        match *self {
            MongePrim::ConstantRows => {
                for mut row in matrix.genrows_mut() {
                    if rng.gen::<bool>() {
                        row.fill(T::one())
                    }
                }
            }
            MongePrim::ConstantCols => {
                for mut col in matrix.gencolumns_mut() {
                    if rng.gen::<bool>() {
                        col.fill(T::one())
                    }
                }
            }
            MongePrim::UpperRightOnes => {
                let i = rng.gen_range(0, (m + 1) as isize);
                let j = rng.gen_range(0, (n + 1) as isize);
                matrix.slice_mut(s![..i, -j..]).fill(T::one());
            }
            MongePrim::LowerLeftOnes => {
                let i = rng.gen_range(0, (m + 1) as isize);
                let j = rng.gen_range(0, (n + 1) as isize);
                matrix.slice_mut(s![-i.., ..j]).fill(T::one());
            }
        }

        matrix
    }
}

/// Generate a random Monge matrix.
pub fn random_monge_matrix<R: Rng, T: PrimInt>(m: usize, n: usize, rng: &mut R) -> Array2<T>
where
    Standard: Distribution<T>,
{
    let monge_primitives = [
        MongePrim::ConstantRows,
        MongePrim::ConstantCols,
        MongePrim::LowerLeftOnes,
        MongePrim::UpperRightOnes,
    ];
    let mut matrix = Array2::from_elem((m, n), T::zero());
    for _ in 0..(m + n) {
        let monge = monge_primitives[rng.gen_range(0, monge_primitives.len())];
        matrix = matrix + monge.to_matrix(m, n, rng);
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn is_monge_handles_overflow() {
        // The x + y <= z + w computations will overflow for an u8
        // matrix unless is_monge is careful.
        let matrix: Array2<u8> = arr2(&[
            [200, 200, 200, 200],
            [200, 200, 200, 200],
            [200, 200, 200, 200],
        ]);
        assert!(is_monge(&matrix));
    }

    #[test]
    fn monge_constant_rows() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let matrix: Array2<u8> = MongePrim::ConstantRows.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        for row in matrix.genrows() {
            let elem = row[0];
            assert_eq!(row, Array::from_elem(matrix.ncols(), elem));
        }
    }

    #[test]
    fn monge_constant_cols() {
        let mut rng = ChaCha20Rng::seed_from_u64(0);
        let matrix: Array2<u8> = MongePrim::ConstantCols.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        for column in matrix.gencolumns() {
            let elem = column[0];
            assert_eq!(column, Array::from_elem(matrix.nrows(), elem));
        }
    }

    #[test]
    fn monge_upper_right_ones() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let matrix: Array2<u8> = MongePrim::UpperRightOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(
            matrix,
            arr2(&[
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ])
        );
    }

    #[test]
    fn monge_lower_left_ones() {
        let mut rng = ChaCha20Rng::seed_from_u64(1);
        let matrix: Array2<u8> = MongePrim::LowerLeftOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(
            matrix,
            arr2(&[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0]
            ])
        );
    }
}

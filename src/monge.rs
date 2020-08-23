//! Functions for generating and checking Monge arrays.
//!
//! The functions here are mostly meant to be used for testing
//! correctness of the SMAWK implementation.
//!
//! **Note: this module is only available if you enable the `ndarray`
//! Cargo feature.**

use ndarray::{s, Array2};
use num_traits::{PrimInt, WrappingAdd};
use rand::distributions::{Distribution, Standard};
use rand::Rng;

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
pub fn is_monge<T: PrimInt + WrappingAdd>(matrix: &Array2<T>) -> bool {
    matrix.windows([2, 2]).into_iter().all(|sub| {
        let (x, y) = (sub[[0, 0]], sub[[1, 1]]);
        let (z, w) = (sub[[0, 1]], sub[[1, 0]]);
        match (x.checked_add(&y), z.checked_add(&w)) {
            (Some(a), Some(b)) => a <= b,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => x.wrapping_add(&y) <= z.wrapping_add(&w),
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

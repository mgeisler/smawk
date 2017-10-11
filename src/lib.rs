#[macro_use(s)]
extern crate ndarray;
extern crate num_traits;
extern crate rand;
#[macro_use]
extern crate rand_derive;

use ndarray::{Array2, ArrayView1, ArrayView2};
use num_traits::{NumOps, PrimInt};
use rand::{Rand, Rng};

/// Compute row minimum by brute force.
///
/// This does a simple scan through the row.
#[inline]
fn row_minimum(row: ArrayView1<i32>) -> usize {
    row.iter()
        .enumerate()
        .min_by_key(|&(idx, elem)| (elem, idx))
        .map(|(idx, _)| idx)
        .expect("empty row in matrix")
}

/// Compute row minima by brute force.
///
/// Running time on an *m* ✕ *n* matrix: O(*mn*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn brute_force_row_minima(matrix: &ArrayView2<i32>) -> Vec<usize> {
    matrix.genrows().into_iter().map(row_minimum).collect()
}

/// Verify that a matrix is a Monge matrix.
///
/// A [Monge matrix] \(or array) is a matrix where the following
/// inequality holds:
///
/// ```text
/// M[i, j] + M[i', j'] <= M[i, j'] + M[i, j']  for all i < i', j < j'
/// ```
///
/// The inequality says that the sum of the main diagonal is less than
/// the sum of the antidiagonal. Checking this condition is done by
/// checking *n* ✕ *m* submatrices, so the running time is O(*mn*).
///
/// [Monge matrix]: https://en.wikipedia.org/wiki/Monge_array
pub fn is_monge<T: Copy + PartialOrd + NumOps>(matrix: &Array2<T>) -> bool {
    matrix
        .windows([2, 2])
        .into_iter()
        .all(|sub| sub[[0, 0]] + sub[[1, 1]] <= sub[[0, 1]] + sub[[1, 0]])
}

/// A Monge matrix can be decomposed into one of these primitive
/// building blocks.
#[derive(Rand)]
enum MongePrim {
    ConstantRows,
    ConstantCols,
    UpperRightOnes,
    LowerLeftOnes,
}

impl MongePrim {
    /// Generate a Monge matrix from a primitive.
    fn to_matrix<T: Rand + PrimInt, R: Rng>(&self, m: usize, n: usize, rng: &mut R) -> Array2<T> {
        let mut matrix = Array2::from_elem((m, n), T::zero());
        // Avoid panic in UpperRightOnes and LowerLeftOnes below.
        if m == 0 || n == 0 {
            return matrix;
        }

        match *self {
            MongePrim::ConstantRows => {
                for mut row in matrix.genrows_mut() {
                    row.fill(rng.gen());
                }
            }
            MongePrim::ConstantCols => {
                for mut col in matrix.gencolumns_mut() {
                    col.fill(rng.gen());
                }
            }
            MongePrim::UpperRightOnes => {
                let i = rng.gen_range(0, m as isize);
                let j = rng.gen_range(0, n as isize);
                matrix.slice_mut(s![..i, -j..]).fill(T::one());
            }
            MongePrim::LowerLeftOnes => {
                let i = rng.gen_range(0, m as isize);
                let j = rng.gen_range(0, n as isize);
                matrix.slice_mut(s![-i.., ..j]).fill(T::one());
            }
        }

        matrix
    }
}

/// Generate a random Monge matrix.
pub fn random_monge<R: Rng>(m: usize, n: usize, rng: &mut R) -> Array2<i32> {
    let mut matrix = Array2::from_elem((m, n), 0);
    for _ in 0..15 {
        let tmp = match rng.gen() {
            true => MongePrim::LowerLeftOnes,
            false => MongePrim::UpperRightOnes,
        }.to_matrix(m, n, rng);
        matrix = matrix + rng.gen_range(1, 15) * tmp;
    }
    matrix
}


#[cfg(test)]
mod tests {
    use super::*;
    use rand::XorShiftRng;
    use ndarray::arr2;

    #[test]
    fn monge_constant_rows() {
        let mut rng = XorShiftRng::new_unseeded();
        assert_eq!(MongePrim::ConstantRows.to_matrix(5, 4, &mut rng),
                   arr2(&[[15u8, 15, 15, 15],
                          [132, 132, 132, 132],
                          [11, 11, 11, 11],
                          [140, 140, 140, 140],
                          [67, 67, 67, 67]]));
    }

    #[test]
    fn monge_constant_cols() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::ConstantCols.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(matrix,
                   arr2(&[[15u8, 132, 11, 140],
                          [15, 132, 11, 140],
                          [15, 132, 11, 140],
                          [15, 132, 11, 140],
                          [15, 132, 11, 140]]));
    }

    #[test]
    fn monge_upper_right_ones() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::UpperRightOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(matrix,
                   arr2(&[[1, 1, 1, 1],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]));
    }

    #[test]
    fn monge_lower_left_ones() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::LowerLeftOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(matrix,
                   arr2(&[[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]]));
    }

    #[test]
    fn brute_force_1x1() {
        let matrix = arr2(&[[7]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![0]);
    }

    #[test]
    fn brute_force_2x1() {
        let matrix = arr2(&[[7], [3]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![0, 0]);
    }

    #[test]
    fn brute_force_1x2() {
        let matrix = arr2(&[[7, 3]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![1]);
    }

    #[test]
    fn brute_force_2x2() {
        let matrix = arr2(&[[7, 3], [3, 7]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![1, 0]);
    }

    #[test]
    fn brute_force_3x3() {
        let matrix = arr2(&[[7, 3, 5], [7, 5, 3], [0, 0, 3]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![1, 2, 0]);
    }

    #[test]
    fn brute_force_4x4() {
        let matrix = arr2(&[[0, 7, 3, 5], [0, 7, 5, 3], [0, 0, 3, -1], [7, 7, 5, 5]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![0, 0, 3, 2]);
    }

    #[test]
    fn brute_force_5x5() {
        let matrix = arr2(&[[7, 7, 3, 5, 0],
                            [7, 7, 5, 5, 9],
                            [0, 0, 3, 1, 3],
                            [3, 4, 5, 6, 7],
                            [7, 6, 5, 5, 6]]);
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![4, 2, 0, 0, 2]);
    }

}

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

/// Compute row minima in O(*m* + *n* log *m*) time.
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn recursive_row_minima(matrix: &Array2<i32>) -> Vec<usize> {
    fn inner(matrix: ArrayView2<i32>, x_offset: usize, minima: &mut [usize]) {
        if matrix.is_empty() {
            return;
        }

        let mid_row = matrix.rows() / 2;
        let min_idx = row_minimum(matrix.row(mid_row));
        minima[mid_row] = x_offset + min_idx;

        if mid_row == 0 {
            return; // Matrix has a single row, so we're done.
        }

        let top_left = matrix.slice(s![..mid_row as isize, ..(min_idx + 1) as isize]);
        inner(top_left, x_offset, &mut minima[..mid_row]);

        let bot_right = matrix.slice(s![(mid_row + 1) as isize.., min_idx as isize..]);
        inner(bot_right, x_offset + min_idx, &mut minima[mid_row + 1..]);

    }

    let mut minima = vec![0; matrix.rows()];
    inner(matrix.view(), 0, &mut minima);

    minima
}

/// Compute row-minima using the SMAWK algorithm.
///
/// Running time on an *m* ✕ *n* matrix: O(*m* + *n*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn smawk_row_minima(matrix: &Array2<i32>) -> Vec<usize> {
    fn inner(matrix: &ArrayView2<i32>, rows: &[usize], cols: &[usize], mut minima: &mut [usize]) {
        if rows.is_empty() {
            return;
        }

        let mut stack = vec![];
        for c in cols {
            while !stack.is_empty() &&
                  matrix[[rows[stack.len() - 1], stack[stack.len() - 1]]] >
                  matrix[[rows[stack.len() - 1], *c]] {
                stack.pop();
            }
            if stack.len() != rows.len() {
                stack.push(*c);
            }
        }
        let cols = &stack;

        let mut odd_rows = Vec::with_capacity(1 + rows.len() / 2);
        for (idx, r) in rows.iter().enumerate() {
            if idx % 2 == 1 {
                odd_rows.push(*r);
            }
        }

        inner(&matrix, &odd_rows, cols, &mut minima);

        let mut c = 0;
        for (r, &row) in rows.iter().enumerate() {
            if r % 2 == 1 {
                continue;
            }
            let mut col = cols[c];
            let last_col = if r == rows.len() - 1 {
                cols[cols.len() - 1]
            } else {
                minima[rows[r + 1]]
            };
            let mut pair = (matrix[[row, col]], col);
            while col != last_col {
                c += 1;
                col = cols[c];
                pair = std::cmp::min(pair, (matrix[[row, col]], col));
            }
            minima[row] = pair.1;
        }
    }

    let mut minima = vec![0; matrix.rows()];
    inner(&matrix.view(),
          &(0..matrix.rows()).collect::<Vec<_>>(),
          &(0..matrix.cols()).collect::<Vec<_>>(),
          &mut minima);

    minima
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
pub fn random_monge_matrix<R: Rng>(m: usize, n: usize, rng: &mut R) -> Array2<i32> {
    let mut matrix = Array2::from_elem((m, n), 0);
    for _ in 0..(m + n) {
        let tmp = match rng.gen() {
            true => MongePrim::LowerLeftOnes,
            false => MongePrim::UpperRightOnes,
        }.to_matrix(m, n, rng);
        matrix = matrix + tmp;
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
                   arr2(&[[0, 0, 0, 1],
                          [0, 0, 0, 1],
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
                          [1, 0, 0, 0],
                          [1, 0, 0, 0]]));
    }

    #[test]
    fn brute_force_1x1() {
        let matrix = arr2(&[[2]]);
        let minima = vec![0];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn brute_force_5x5() {
        let matrix = arr2(&[[3, 2, 4, 5, 6],
                            [2, 1, 3, 3, 4],
                            [2, 1, 3, 3, 4],
                            [3, 2, 4, 3, 4],
                            [4, 3, 2, 1, 1]]);
        let minima = vec![1, 1, 1, 1, 3];
        assert_eq!(brute_force_row_minima(&matrix.view()), minima);
    }

    #[test]
    fn recursive_1x1() {
        let matrix = arr2(&[[2]]);
        let minima = vec![0];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn recursive_5x5() {
        let matrix = arr2(&[[3, 2, 4, 5, 6],
                            [2, 1, 3, 3, 4],
                            [2, 1, 3, 3, 4],
                            [3, 2, 4, 3, 4],
                            [4, 3, 2, 1, 1]]);
        let minima = vec![1, 1, 1, 1, 3];
        assert_eq!(recursive_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_1x1() {
        let matrix = arr2(&[[2]]);
        let minima = vec![0];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    #[test]
    fn smawk_5x5() {
        let matrix = arr2(&[[3, 2, 4, 5, 6],
                            [2, 1, 3, 3, 4],
                            [2, 1, 3, 3, 4],
                            [3, 2, 4, 3, 4],
                            [4, 3, 2, 1, 1]]);
        let minima = vec![1, 1, 1, 1, 3];
        assert_eq!(smawk_row_minima(&matrix), minima);
    }

    /// Check that the brute_force_row_minima, recursive_row_minima,
    /// and smawk_row_minima functions give identical results on a
    /// large number of randomly generated Monge matrices.
    #[test]
    fn implementations_agree() {
        let sizes = vec![1, 2, 3, 4, 5, 10, 15, 20, 30];
        let mut rng = XorShiftRng::new_unseeded();
        for _ in 0..4 {
            for m in sizes.clone().iter() {
                for n in sizes.clone().iter() {
                    let matrix = random_monge_matrix(*m, *n, &mut rng);
                    let brute_force = brute_force_row_minima(&matrix.view());
                    let recursive = recursive_row_minima(&matrix);
                    let smawk = smawk_row_minima(&matrix);
                    assert_eq!(brute_force,
                               recursive,
                               "recursive and brute force differs on:\n{:?}",
                               matrix);
                    assert_eq!(brute_force,
                               smawk,
                               "SMAWK and brute force differs on:\n{:?}",
                               matrix);
                }
            }
        }
    }
}

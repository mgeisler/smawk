#![doc(html_root_url = "https://docs.rs/smawk/0.1.0")]

#[macro_use(s)]
extern crate ndarray;
extern crate num_traits;
extern crate rand;
#[macro_use]
extern crate rand_derive;

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Si};
use num_traits::{NumOps, PrimInt};
use rand::{Rand, Rng};

/// Compute lane minimum by brute force.
///
/// This does a simple scan through the lane (row or column).
#[inline]
fn lane_minimum(lane: ArrayView1<i32>) -> usize {
    lane.iter()
        .enumerate()
        .min_by_key(|&(idx, elem)| (elem, idx))
        .map(|(idx, _)| idx)
        .expect("empty lane in matrix")
}

/// Compute row minima by brute force.
///
/// Running time on an *m* ✕ *n* matrix: O(*mn*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn brute_force_row_minima(matrix: &Array2<i32>) -> Vec<usize> {
    matrix.genrows().into_iter().map(lane_minimum).collect()
}

/// Compute column minima by brute force.
///
/// Running time on an *m* ✕ *n* matrix: O(*mn*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn brute_force_column_minima(matrix: &Array2<i32>) -> Vec<usize> {
    matrix.gencolumns().into_iter().map(lane_minimum).collect()
}

/// Compute row minima in O(*m* + *n* log *m*) time.
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn recursive_row_minima(matrix: &Array2<i32>) -> Vec<usize> {
    let mut minima = vec![0; matrix.rows()];
    recursive_inner(matrix.view(), &|| Direction::Row, 0, &mut minima);
    minima
}

/// Compute column minima in O(*n* + *m* log *n*) time.
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn recursive_column_minima(matrix: &Array2<i32>) -> Vec<usize> {
    let mut minima = vec![0; matrix.cols()];
    recursive_inner(matrix.view(), &|| Direction::Column, 0, &mut minima);
    minima
}

/// The type of minima (row or column) we compute.
enum Direction {
    Row,
    Column,
}

/// Compute the minima along the given direction (`Direction::Row` for
/// row minima and `Direction::Column` for column minima).
///
/// The direction is given as a generic function argument to allow
/// monomorphization to kick in. The function calls will be inlined
/// and optimized away and the result is that the compiler generates
/// differnet code for finding row and column minima.
fn recursive_inner<F: Fn() -> Direction>(matrix: ArrayView2<i32>,
                                         dir: &F,
                                         offset: usize,
                                         minima: &mut [usize]) {
    if matrix.is_empty() {
        return;
    }

    let axis = match dir() {
        Direction::Row => Axis(0),
        Direction::Column => Axis(1),
    };
    let mid = matrix.len_of(axis) / 2;
    let min_idx = lane_minimum(matrix.subview(axis, mid));
    minima[mid] = offset + min_idx;

    if mid == 0 {
        return; // Matrix has a single row or column, so we're done.
    }

    let top_left = match dir() {
        Direction::Row => {
            [Si(0, Some(mid as isize), 1),
             Si(0, Some((min_idx + 1) as isize), 1)]
        }
        Direction::Column => {
            [Si(0, Some((min_idx + 1) as isize), 1),
             Si(0, Some(mid as isize), 1)]
        }
    };
    let bot_right = match dir() {
        Direction::Row => {
            [Si((mid + 1) as isize, None, 1),
             Si(min_idx as isize, None, 1)]
        }
        Direction::Column => {
            [Si(min_idx as isize, None, 1),
             Si((mid + 1) as isize, None, 1)]
        }
    };
    recursive_inner(matrix.slice(&top_left), dir, offset, &mut minima[..mid]);
    recursive_inner(matrix.slice(&bot_right),
                    dir,
                    offset + min_idx,
                    &mut minima[mid + 1..]);
}

/// Compute row-minima using the SMAWK algorithm.
///
/// Running time on an *m* ✕ *n* matrix: O(*m* + *n*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn smawk_row_minima(matrix: &Array2<i32>) -> Vec<usize> {
    let mut minima = vec![0; matrix.rows()];
    smawk_inner(&matrix.view(),
                &(0..matrix.rows()).collect::<Vec<_>>(),
                &(0..matrix.cols()).collect::<Vec<_>>(),
                &mut minima);
    minima
}

/// Compute column-minima using the SMAWK algorithm.
///
/// Running time on an *m* ✕ *n* matrix: O(*m* + *n*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn smawk_column_minima(matrix: &Array2<i32>) -> Vec<usize> {
    // Benchmarking shows that SMAWK performs roughly the same on row-
    // and column-major matrices.
    let mut minima = vec![0; matrix.cols()];
    smawk_inner(&matrix.t(),
                &(0..matrix.cols()).collect::<Vec<_>>(),
                &(0..matrix.rows()).collect::<Vec<_>>(),
                &mut minima);
    minima
}

/// Compute row minima in the given area of the matrix. The `minima`
/// slice is updated inplace.
fn smawk_inner(matrix: &ArrayView2<i32>, rows: &[usize], cols: &[usize], mut minima: &mut [usize]) {
    if rows.is_empty() {
        return;
    }

    let mut stack = Vec::with_capacity(rows.len());
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

    smawk_inner(&matrix, &odd_rows, cols, &mut minima);

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
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn brute_force_5x5() {
        let matrix = arr2(&[[3, 2, 4, 5, 6],
                            [2, 1, 3, 3, 4],
                            [2, 1, 3, 3, 4],
                            [3, 2, 4, 3, 4],
                            [4, 3, 2, 1, 1]]);
        let minima = vec![1, 1, 1, 1, 3];
        assert_eq!(brute_force_row_minima(&matrix), minima);
        assert_eq!(brute_force_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_1x1() {
        let matrix = arr2(&[[2]]);
        let minima = vec![0];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn recursive_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(recursive_row_minima(&matrix), minima);
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
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
        assert_eq!(recursive_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_1x1() {
        let matrix = arr2(&[[2]]);
        let minima = vec![0];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_2x1() {
        let matrix = arr2(&[[3], [2]]);
        let minima = vec![0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_1x2() {
        let matrix = arr2(&[[2, 1]]);
        let minima = vec![1];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_2x2() {
        let matrix = arr2(&[[3, 2], [2, 1]]);
        let minima = vec![1, 1];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_3x3() {
        let matrix = arr2(&[[3, 4, 4], [3, 4, 4], [2, 3, 3]]);
        let minima = vec![0, 0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    #[test]
    fn smawk_4x4() {
        let matrix = arr2(&[[4, 5, 5, 5], [2, 3, 3, 3], [2, 3, 3, 3], [2, 2, 2, 2]]);
        let minima = vec![0, 0, 0, 0];
        assert_eq!(smawk_row_minima(&matrix), minima);
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
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
        assert_eq!(smawk_column_minima(&matrix.reversed_axes()), minima);
    }

    /// Check that the brute force, recursive, and SMAWK functions
    /// give identical results on a large number of randomly generated
    /// Monge matrices.
    #[test]
    fn implementations_agree() {
        let sizes = vec![1, 2, 3, 4, 5, 10, 15, 20, 30];
        let mut rng = XorShiftRng::new_unseeded();
        for _ in 0..4 {
            for m in sizes.clone().iter() {
                for n in sizes.clone().iter() {
                    let matrix = random_monge_matrix(*m, *n, &mut rng);

                    // Compute and test row minima.
                    let brute_force = brute_force_row_minima(&matrix);
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

                    // Do the same for the column minima.
                    let brute_force = brute_force_column_minima(&matrix);
                    let recursive = recursive_column_minima(&matrix);
                    let smawk = smawk_column_minima(&matrix);
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

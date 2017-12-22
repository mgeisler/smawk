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
fn lane_minimum<T: Ord>(lane: ArrayView1<T>) -> usize {
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
pub fn brute_force_row_minima<T: Ord>(matrix: &Array2<T>) -> Vec<usize> {
    matrix.genrows().into_iter().map(lane_minimum).collect()
}

/// Compute column minima by brute force.
///
/// Running time on an *m* ✕ *n* matrix: O(*mn*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn brute_force_column_minima<T: Ord>(matrix: &Array2<T>) -> Vec<usize> {
    matrix.gencolumns().into_iter().map(lane_minimum).collect()
}

/// Compute row minima in O(*m* + *n* log *m*) time.
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn recursive_row_minima<T: Ord>(matrix: &Array2<T>) -> Vec<usize> {
    let mut minima = vec![0; matrix.rows()];
    recursive_inner(matrix.view(), &|| Direction::Row, 0, &mut minima);
    minima
}

/// Compute column minima in O(*n* + *m* log *n*) time.
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn recursive_column_minima<T: Ord>(matrix: &Array2<T>) -> Vec<usize> {
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
fn recursive_inner<T: Ord, F: Fn() -> Direction>(
    matrix: ArrayView2<T>,
    dir: &F,
    offset: usize,
    minima: &mut [usize],
) {
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
        Direction::Row => [
            Si(0, Some(mid as isize), 1),
            Si(0, Some((min_idx + 1) as isize), 1),
        ],
        Direction::Column => [
            Si(0, Some((min_idx + 1) as isize), 1),
            Si(0, Some(mid as isize), 1),
        ],
    };
    let bot_right = match dir() {
        Direction::Row => [
            Si((mid + 1) as isize, None, 1),
            Si(min_idx as isize, None, 1),
        ],
        Direction::Column => [
            Si(min_idx as isize, None, 1),
            Si((mid + 1) as isize, None, 1),
        ],
    };
    recursive_inner(matrix.slice(&top_left), dir, offset, &mut minima[..mid]);
    recursive_inner(
        matrix.slice(&bot_right),
        dir,
        offset + min_idx,
        &mut minima[mid + 1..],
    );
}

/// Compute row minima in O(*m* + *n*) time.
///
/// This implements the SMAWK algorithm for finding row minima in a
/// totally monotone matrix.
///
/// The SMAWK algorithm is from Agarwal, Klawe, Moran, Shor, and
/// Wilbur, *Geometric applications of a matrix searching algorithm*,
/// Algorithmica 2, pp. 195-208 (1987) and the code here is a
/// translation [David Eppstein's Python code][pads].
///
/// [pads]: https://github.com/jfinkels/PADS/blob/master/pads/smawk.py
///
/// Running time on an *m* ✕ *n* matrix: O(*m* + *n*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
pub fn smawk_row_minima<T: Ord + Copy>(matrix: &Array2<T>) -> Vec<usize> {
    // Benchmarking shows that SMAWK performs roughly the same on row-
    // and column-major matrices.
    let mut minima = vec![0; matrix.rows()];
    smawk_inner(
        &|j, i| matrix[[i, j]],
        &(0..matrix.cols()).collect::<Vec<_>>(),
        &(0..matrix.rows()).collect::<Vec<_>>(),
        &mut minima,
    );
    minima
}

/// Compute column minima in O(*m* + *n*) time.
///
/// This implements the SMAWK algorithm for finding column minima in a
/// totally monotone matrix.
///
/// The SMAWK algorithm is from Agarwal, Klawe, Moran, Shor, and
/// Wilbur, *Geometric applications of a matrix searching algorithm*,
/// Algorithmica 2, pp. 195-208 (1987) and the code here is a
/// translation [David Eppstein's Python code][pads].
///
/// [pads]: https://github.com/jfinkels/PADS/blob/master/pads/smawk.py
///
/// Running time on an *m* ✕ *n* matrix: O(*m* + *n*).
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
pub fn smawk_column_minima<T: Ord + Copy>(matrix: &Array2<T>) -> Vec<usize> {
    let mut minima = vec![0; matrix.cols()];
    smawk_inner(
        &|i, j| matrix[[i, j]],
        &(0..matrix.rows()).collect::<Vec<_>>(),
        &(0..matrix.cols()).collect::<Vec<_>>(),
        &mut minima,
    );
    minima
}

/// Compute column minima in the given area of the matrix. The
/// `minima` slice is updated inplace.
fn smawk_inner<T: Ord + Copy, M: Fn(usize, usize) -> T>(
    matrix: &M,
    rows: &[usize],
    cols: &[usize],
    mut minima: &mut [usize],
) {
    if cols.is_empty() {
        return;
    }

    let mut stack = Vec::with_capacity(cols.len());
    for r in rows {
        // TODO: use stack.last() instead of stack.is_empty() etc
        while !stack.is_empty()
            && matrix(stack[stack.len() - 1], cols[stack.len() - 1])
                > matrix(*r, cols[stack.len() - 1])
        {
            stack.pop();
        }
        if stack.len() != cols.len() {
            stack.push(*r);
        }
    }
    let rows = &stack;

    let mut odd_cols = Vec::with_capacity(1 + cols.len() / 2);
    for (idx, c) in cols.iter().enumerate() {
        if idx % 2 == 1 {
            odd_cols.push(*c);
        }
    }

    smawk_inner(matrix, rows, &odd_cols, &mut minima);

    let mut r = 0;
    for (c, &col) in cols.iter().enumerate().filter(|(c, _)| c % 2 == 0) {
        let mut row = rows[r];
        let last_row = if c == cols.len() - 1 {
            rows[rows.len() - 1]
        } else {
            minima[cols[c + 1]]
        };
        let mut pair = (matrix(row, col), row);
        while row != last_row {
            r += 1;
            row = rows[r];
            pair = std::cmp::min(pair, (matrix(row, col), row));
        }
        minima[col] = pair.1;
    }
}

/// Compute upper-right column minima in O(*m* + *n*) time.
///
/// The input matrix must be totally monotone.
///
/// The function returns a vector of `(usize, i32)` tuples. The
/// `usize` in the tuple at index `j` tells you the row of the minimum
/// value in column `j` and the `i32` value is minimum value itself.
///
/// The algorithm only considers values above the main diagonal, which
/// means that it computes values `v(j)` where:
///
/// ```text
/// v(0) = initial
/// v(j) = min { M[i, j] | i < j } for j > 0
/// ```
///
/// If we let `r(j)` denote the row index of the minimum value in
/// column `j`, the tuples in the result vector become `(r(j), M[r(j),
/// j])`.
///
/// The algorithm is an *online* algorithm, in the sense that `matrix`
/// function can refer back to previously computed column minima when
/// determining an entry in the matrix. The guarantee is that we only
/// call `matrix(i, j)` after having computed `v(i)`. This is
/// reflected in the `&[(usize, i32)]` argument to `matrix`, which
/// grows as more and more values are computed.
pub fn online_column_minima<M: Fn(&[(usize, i32)], usize, usize) -> i32>(
    initial: i32,
    size: usize,
    matrix: M,
) -> Vec<(usize, i32)> {
    let mut result = vec![(0, initial)];

    // State used by the algorithm.
    let mut finished = 0;
    let mut base = 0;
    let mut tentative = 0;

    // Shorthand for evaluating the matrix. We need a macro here since
    // we don't want to borrow the result vector.
    macro_rules! m {
        ($i:expr, $j:expr) => {{
            assert!(
                $i < size && $j < size,
                "index out of bounds: ({}, {}), matrix size: {}",
                $i,
                $j,
                size
            );
            matrix(&result[..finished + 1], $i, $j)
        }};
    }

    // Keep going until we have finished all size columns. Since the
    // columns are zero-indexed, we're done when finished == size - 1.
    while finished < size - 1 {
        // First case: we have already advanced past the previous
        // tentative value. We make a new tentative value by applying
        // smawk_inner to the largest square submatrix that fits under
        // the base.
        let i = finished + 1;
        if i > tentative {
            let rows = (base..finished + 1).collect::<Vec<_>>();
            tentative = std::cmp::min(finished + rows.len(), size - 1);
            let cols = (finished + 1..tentative + 1).collect::<Vec<_>>();
            let mut minima = vec![0; tentative + 1];
            smawk_inner(&|i, j| m![i, j], &rows, &cols, &mut minima);
            for col in cols {
                let row = minima[col];
                let v = m![row, col];
                if col >= result.len() {
                    result.push((row, v));
                } else if v < result[col].1 {
                    result[col] = (row, v);
                }
            }
            finished = i;
            continue;
        }

        // Second case: the new column minimum is on the diagonal. All
        // subsequent ones will be at least as low, so we can clear
        // out all our work from higher rows. As in the fourth case,
        // the loss of tentative is amortized against the increase in
        // base.
        let diag = m![i - 1, i];
        if diag < result[i].1 {
            result[i] = (i - 1, diag);
            base = i - 1;
            tentative = i;
            finished = i;
            continue;
        }

        // Third case: row i-1 does not supply a column minimum in any
        // column up to tentative. We simply advance finished while
        // maintaining the invariant.
        if m![i - 1, tentative] >= result[tentative].1 {
            finished = i;
            continue;
        }

        // Fourth and final case: a new column minimum at tentative.
        // This allows us to make progress by incorporating rows prior
        // to finished into the base. The base invariant holds because
        // these rows cannot supply any later column minima. The work
        // done when we last advanced tentative (and undone by this
        // step) can be amortized against the increase in base.
        base = i - 1;
        tentative = i;
        finished = i;
    }

    result
}

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
pub fn random_monge_matrix<R: Rng, T>(m: usize, n: usize, rng: &mut R) -> Array2<T>
where
    T: Rand + PrimInt,
{
    let mut matrix = Array2::from_elem((m, n), T::zero());
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
    use ndarray::arr2;
    use rand::XorShiftRng;

    #[test]
    fn monge_constant_rows() {
        let mut rng = XorShiftRng::new_unseeded();
        assert_eq!(
            MongePrim::ConstantRows.to_matrix(5, 4, &mut rng),
            arr2(&[
                [15u8, 15, 15, 15],
                [132, 132, 132, 132],
                [11, 11, 11, 11],
                [140, 140, 140, 140],
                [67, 67, 67, 67]
            ])
        );
    }

    #[test]
    fn monge_constant_cols() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::ConstantCols.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(
            matrix,
            arr2(&[
                [15u8, 132, 11, 140],
                [15, 132, 11, 140],
                [15, 132, 11, 140],
                [15, 132, 11, 140],
                [15, 132, 11, 140]
            ])
        );
    }

    #[test]
    fn monge_upper_right_ones() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::UpperRightOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(
            matrix,
            arr2(&[
                [0, 0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ])
        );
    }

    #[test]
    fn monge_lower_left_ones() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix = MongePrim::LowerLeftOnes.to_matrix(5, 4, &mut rng);
        assert!(is_monge(&matrix));
        assert_eq!(
            matrix,
            arr2(&[
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0]
            ])
        );
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
        let matrix = arr2(&[
            [3, 2, 4, 5, 6],
            [2, 1, 3, 3, 4],
            [2, 1, 3, 3, 4],
            [3, 2, 4, 3, 4],
            [4, 3, 2, 1, 1],
        ]);
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
        let matrix = arr2(&[
            [3, 2, 4, 5, 6],
            [2, 1, 3, 3, 4],
            [2, 1, 3, 3, 4],
            [3, 2, 4, 3, 4],
            [4, 3, 2, 1, 1],
        ]);
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
        let matrix = arr2(&[
            [3, 2, 4, 5, 6],
            [2, 1, 3, 3, 4],
            [2, 1, 3, 3, 4],
            [3, 2, 4, 3, 4],
            [4, 3, 2, 1, 1],
        ]);
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
                    let matrix: Array2<i32> = random_monge_matrix(*m, *n, &mut rng);

                    // Compute and test row minima.
                    let brute_force = brute_force_row_minima(&matrix);
                    let recursive = recursive_row_minima(&matrix);
                    let smawk = smawk_row_minima(&matrix);
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
                    let brute_force = brute_force_column_minima(&matrix);
                    let recursive = recursive_column_minima(&matrix);
                    let smawk = smawk_column_minima(&matrix);
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

    #[test]
    fn concave_online() {
        let mut rng = XorShiftRng::new_unseeded();
        let matrix: Array2<i32> = random_monge_matrix(15, 15, &mut rng);

        println!();
        if matrix.rows() < 25 {
            println!("{:2?}", matrix);
        }
        println!("is Monge? {:?}", is_monge(&matrix));
        println!();

        for size in 1..matrix.rows() + 1 {
            let count = std::cell::RefCell::new(0);
            let minima = online_column_minima(99, size, |values, i, j| {
                println!(
                    "- evaluating ({}, {}) -> {:?}, values: {:?}",
                    i,
                    j,
                    matrix.get([i, j]),
                    values
                );
                *count.borrow_mut() += 1;
                matrix[[i, j]]
            });

            println!("minima: {:?}", minima);
            let c = count.into_inner();
            println!(
                "evaluations: {:4} -> {:4} (x{:6.2})",
                size,
                c,
                c as f64 / size as f64
            );
        }
    }

}

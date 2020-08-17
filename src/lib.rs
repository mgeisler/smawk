//! This crate implements various functions that help speed up dynamic
//! programming, most importantly the SMAWK algorithm for finding row
//! or column minima in a totally monotone matrix with *m* rows and
//! *n* columns in time O(*m* + *n*). This is much better than the
//! brute force solution which would take O(*mn*). When *m* and *n*
//! are of the same order, this turns a quadratic function into a
//! linear function.
//!
//! # Examples
//!
//! Computing the column minima of an *m* ✕ *n* Monge matrix can be
//! done efficiently with `smawk_column_minima`:
//!
//! ```
//! use ndarray::arr2;
//! use smawk::smawk_column_minima;
//!
//! let matrix = arr2(&[
//!     [3, 2, 4, 5, 6],
//!     [2, 1, 3, 3, 4],
//!     [2, 1, 3, 3, 4],
//!     [3, 2, 4, 3, 4],
//!     [4, 3, 2, 1, 1],
//! ]);
//! let minima = vec![1, 1, 4, 4, 4];
//! assert_eq!(smawk_column_minima(&matrix), minima);
//! ```
//!
//! The `minima` vector gives the index of the minimum value per
//! column, so `minima[0] == 1` since the minimum value in the first
//! column is 2 (row 1). Note that the smallest row index is returned.
//!
//! # Definitions
//!
//! Some of the functions in this crate only work on matrices that are
//! *totally monotone*, which we will define below.
//!
//! ## Monotone Matrices
//!
//! We start with a helper definition. Given an *m* ✕ *n* matrix `M`,
//! we say that `M` is *monotone* when the minimum value of row `i` is
//! found to the left of the minimum value in row `i'` where `i < i'`.
//!
//! More formally, if we let `rm(i)` denote the column index of the
//! left-most minimum value in row `i`, then we have
//!
//! ```text
//! rm(0) ≤ rm(1) ≤ ... ≤ rm(m - 1)
//! ```
//!
//! This means that as you go down the rows from top to bottom, the
//! row-minima proceed from left to right.
//!
//! The algorithms in this crate deal with finding such row- and
//! column-minima.
//!
//! ## Totally Monotone Matrices
//!
//! We say that a matrix `M` is *totally monotone* when every
//! sub-matrix is monotone. A sub-matrix is formed by the intersection
//! of any two rows `i < i'` and any two columns `j < j'`.
//!
//! This is often expressed as via this equivalent condition:
//!
//! ```text
//! M[i, j] > M[i, j']  =>  M[i', j] > M[i', j']
//! ```
//!
//! for all `i < i'` and `j < j'`.
//!
//! ## Monge Property for Matrices
//!
//! A matrix `M` is said to fulfill the *Monge property* if
//!
//! ```text
//! M[i, j] + M[i', j'] ≤ M[i, j'] + M[i', j]
//! ```
//!
//! for all `i < i'` and `j < j'`. This says that given any rectangle
//! in the matrix, the sum of the top-left and bottom-right corners is
//! less than or equal to the sum of the bottom-left and upper-right
//! corners.
//!
//! All Monge matrices are totally monotone, so it is enough to
//! establish that the Monge property holds in order to use a matrix
//! with the functions in this crate. If your program is dealing with
//! unknown inputs, it can use [`is_monge`] to verify that a matrix is
//! a Monge matrix.
//!
//! [`is_monge`]: fn.is_monge.html

#![doc(html_root_url = "https://docs.rs/smawk/0.2.0")]

pub mod brute_force;
pub mod monge;
pub mod recursive;

/// Minimal matrix trait for two-dimensional arrays.
///
/// This provides the functionality needed to represent a read-only
/// numeric matrix. You can query the size of the matrix and access
/// elements. Modeled after
/// [`ndarray::Array2`](https://docs.rs/ndarray/latest/ndarray/type.Array2.html)
/// from the [ndarray crate ](https://crates.io/crates/ndarray).
pub trait Matrix<T: Copy>: std::ops::Index<[usize; 2], Output = T> {
    /// Return the number of rows.
    fn nrows(&self) -> usize;
    /// Return the number of columns.
    fn ncols(&self) -> usize;
}

/// Adapting `ndarray::Array2` to the `Matrix` trait.
impl<T: Copy> Matrix<T> for ndarray::Array2<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
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
pub fn smawk_row_minima<T: Ord + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    // Benchmarking shows that SMAWK performs roughly the same on row-
    // and column-major matrices.
    let mut minima = vec![0; matrix.nrows()];
    smawk_inner(
        &|j, i| matrix[[i, j]],
        &(0..matrix.ncols()).collect::<Vec<_>>(),
        &(0..matrix.nrows()).collect::<Vec<_>>(),
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
pub fn smawk_column_minima<T: Ord + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    let mut minima = vec![0; matrix.ncols()];
    smawk_inner(
        &|i, j| matrix[[i, j]],
        &(0..matrix.nrows()).collect::<Vec<_>>(),
        &(0..matrix.ncols()).collect::<Vec<_>>(),
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
/// The function returns a vector of `(usize, T)`. The `usize` in the
/// tuple at index `j` tells you the row of the minimum value in
/// column `j` and the `T` value is minimum value itself.
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
/// reflected in the `&[(usize, T)]` argument to `matrix`, which grows
/// as more and more values are computed.
pub fn online_column_minima<T: Copy + Ord, M: Fn(&[(usize, T)], usize, usize) -> T>(
    initial: T,
    size: usize,
    matrix: M,
) -> Vec<(usize, T)> {
    let mut result = vec![(0, initial)];

    // State used by the algorithm.
    let mut finished = 0;
    let mut base = 0;
    let mut tentative = 0;

    // Shorthand for evaluating the matrix. We need a macro here since
    // we don't want to borrow the result vector.
    macro_rules! m {
        ($i:expr, $j:expr) => {{
            assert!($i < $j, "(i, j) not above diagonal: ({}, {})", $i, $j);
            assert!(
                $i < size && $j < size,
                "(i, j) out of bounds: ({}, {}), size: {}",
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

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

    #[test]
    fn online_1x1() {
        let matrix = arr2(&[[0]]);
        let minima = vec![(0, 0)];
        assert_eq!(
            online_column_minima(0, 1, |_, i, j| matrix[[i, j]],),
            minima
        );
    }

    #[test]
    fn online_2x2() {
        let matrix = arr2(&[[0, 2], [0, 0]]);
        let minima = vec![(0, 0), (0, 2)];
        assert_eq!(
            online_column_minima(0, 2, |_, i, j| matrix[[i, j]],),
            minima
        );
    }

    #[test]
    fn online_3x3() {
        let matrix = arr2(&[[0, 4, 4], [0, 0, 4], [0, 0, 0]]);
        let minima = vec![(0, 0), (0, 4), (0, 4)];
        assert_eq!(
            online_column_minima(0, 3, |_, i, j| matrix[[i, j]],),
            minima
        );
    }

    #[test]
    fn online_4x4() {
        let matrix = arr2(&[[0, 5, 5, 5], [0, 0, 3, 3], [0, 0, 0, 3], [0, 0, 0, 0]]);
        let minima = vec![(0, 0), (0, 5), (1, 3), (1, 3)];
        assert_eq!(
            online_column_minima(0, 4, |_, i, j| matrix[[i, j]],),
            minima
        );
    }

    #[test]
    fn online_5x5() {
        let matrix = arr2(&[
            [0, 2, 4, 6, 7],
            [0, 0, 3, 4, 5],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
        ]);
        let minima = vec![(0, 0), (0, 2), (1, 3), (2, 3), (2, 4)];
        assert_eq!(online_column_minima(0, 5, |_, i, j| matrix[[i, j]]), minima);
    }
}

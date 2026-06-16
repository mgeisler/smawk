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
//! Computing the column minima of an *m* × *n* Monge matrix can be
//! done efficiently with `smawk::column_minima`:
//!
//! ```
//! use smawk::Matrix;
//!
//! let matrix = vec![
//!     vec![3, 2, 4, 5, 6],
//!     vec![2, 1, 3, 3, 4],
//!     vec![2, 1, 3, 3, 4],
//!     vec![3, 2, 4, 3, 4],
//!     vec![4, 3, 2, 1, 1],
//! ];
//! let minima = vec![1, 1, 4, 4, 4];
//! assert_eq!(smawk::column_minima(&matrix), minima);
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
//! We start with a helper definition. Given an *m* × *n* matrix `M`,
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
//! unknown inputs, it can use [`monge::is_monge`] to verify that a
//! matrix is a Monge matrix.
//!
//! ## Online Column Minima and Dynamic Programming
//!
//! In the standard offline SMAWK algorithm, the entire matrix must be
//! known and queryable upfront. However, many dynamic programming
//! problems (like the *Least Weight Subsequence* problem) involve
//! finding a sequence of indices to minimize a transition cost of the
//! form:
//!
//! ```text
//! v(0) = initial
//! v(j) = min { v(i) + w(i, j) | 0 <= i < j }  for j > 0
//! ```
//!
//! If the transition weight function `w(i, j)` satisfies the Monge
//! property, the matrix `M[i, j] = v(i) + w(i, j)` is totally
//! monotone. However, the matrix entries in column `j` cannot be
//! computed until the optimal prefix values `v(i)` for all `i < j`
//! are finalized. This is an *online* dependency constraint.
//!
//! The [`online_column_minima`] function solves this online dynamic
//! programming problem in O(*n*) time. It wraps the core SMAWK
//! algorithm inside an online check-and-correct harness (based on the
//! Galil-Park algorithm), executing matrix queries dynamically as the
//! input prefix calculations are completed.
//!
//! A prime practical application of this is the Knuth-Plass paragraph
//! line-breaking algorithm (used in TeX and crates like `textwrap`).
//! By framing word wrapping as a concave least weight subsequence
//! problem, the optimal line breaks of a paragraph of *n* words can
//! be found in O(*n*) time instead of O(*n*²).
//!
//! # References
//!
//! - Alok Aggarwal, Maria M. Klawe, Shlomo Moran, Peter Shor, and Robert Wilber.
//!   **Geometric applications of a matrix searching algorithm**.
//!   *Algorithmica*, 2(1):195–208, 1987.
//! - Robert Wilber. **The concave least-weight subsequence problem
//!   revisited**. *Journal of Algorithms*, 9(3):418–425, 1988.
//! - Zvi Galil and Kunsoo Park. **A linear-time algorithm for concave
//!   one-dimensional dynamic programming**. *Information Processing Letters*,
//!   33(6):309–313, 1990.
//! - Donald E. Knuth and Michael F. Plass. **Breaking paragraphs into lines**.
//!   *Software: Practice and Experience*, 11(11):1119–1184, 1981.

#![no_std]
#![doc(html_root_url = "https://docs.rs/smawk/0.3.3")]
// The s! macro from ndarray uses unsafe internally, so we can only
// forbid unsafe code when building with the default features.
#![cfg_attr(not(feature = "ndarray"), forbid(unsafe_code))]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

#[cfg(feature = "ndarray")]
pub mod brute_force;
pub mod monge;
#[cfg(feature = "ndarray")]
pub mod recursive;

/// Minimal matrix trait for two-dimensional arrays.
///
/// This provides the functionality needed to represent a read-only
/// numeric matrix. You can query the size of the matrix and access
/// elements. Modeled after [`ndarray::Array2`] from the [ndarray
/// crate](https://crates.io/crates/ndarray).
///
/// Enable the `ndarray` Cargo feature if you want to use it with
/// `ndarray::Array2`.
pub trait Matrix<T: Copy> {
    /// Return the number of rows.
    fn nrows(&self) -> usize;
    /// Return the number of columns.
    fn ncols(&self) -> usize;
    /// Return a matrix element.
    fn index(&self, row: usize, column: usize) -> T;
}

/// Simple and inefficient matrix representation used for doctest
/// examples and simple unit tests.
///
/// You should prefer implementing it yourself, or you can enable the
/// `ndarray` Cargo feature and use the provided implementation for
/// [`ndarray::Array2`].
impl<T: Copy> Matrix<T> for Vec<Vec<T>> {
    fn nrows(&self) -> usize {
        self.len()
    }
    fn ncols(&self) -> usize {
        self[0].len()
    }
    fn index(&self, row: usize, column: usize) -> T {
        self[row][column]
    }
}

/// Adapting [`ndarray::Array2`] to the `Matrix` trait.
///
/// **Note: this implementation is only available if you enable the
/// `ndarray` Cargo feature.**
#[cfg(feature = "ndarray")]
impl<T: Copy> Matrix<T> for ndarray::Array2<T> {
    #[inline]
    fn nrows(&self) -> usize {
        self.nrows()
    }
    #[inline]
    fn ncols(&self) -> usize {
        self.ncols()
    }
    #[inline]
    fn index(&self, row: usize, column: usize) -> T {
        self[[row, column]]
    }
}

/// Compute row minima in O(*m* + *n*) time.
///
/// This implements the [SMAWK algorithm] for efficiently finding row
/// minima in a totally monotone matrix.
///
/// The SMAWK algorithm is from Agarwal, Klawe, Moran, Shor, and
/// Wilbur, *Geometric applications of a matrix searching algorithm*,
/// Algorithmica 2, pp. 195-208 (1987) and the code here is a
/// translation [David Eppstein's Python code][pads].
///
/// Running time on an *m* × *n* matrix: O(*m* + *n*).
///
/// # Examples
///
/// ```
/// use smawk::Matrix;
/// let matrix = vec![vec![4, 2, 4, 3],
///                   vec![5, 3, 5, 3],
///                   vec![5, 3, 3, 1]];
/// assert_eq!(smawk::row_minima(&matrix),
///            vec![1, 1, 3]);
/// ```
///
/// # Panics
///
/// It is an error to call this on a matrix with zero columns.
///
/// [pads]: https://github.com/jfinkels/PADS/blob/master/pads/smawk.py
/// [SMAWK algorithm]: https://en.wikipedia.org/wiki/SMAWK_algorithm
pub fn row_minima<T: PartialOrd + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    // Benchmarking shows that SMAWK performs roughly the same on row-
    // and column-major matrices.
    let mut minima = vec![0; matrix.nrows()];
    let (mut rows_scratch, mut cols_scratch) = scratchpads(matrix.ncols(), matrix.nrows());
    smawk_inner(
        &|j, i| matrix.index(i, j),
        (&mut rows_scratch, 0, matrix.ncols()),
        (&mut cols_scratch, 0, matrix.nrows()),
        &mut minima,
    );
    minima
}

#[deprecated(since = "0.3.2", note = "Please use `row_minima` instead.")]
pub fn smawk_row_minima<T: PartialOrd + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    row_minima(matrix)
}

/// Compute column minima in O(*m* + *n*) time.
///
/// This implements the [SMAWK algorithm] for efficiently finding
/// column minima in a totally monotone matrix.
///
/// The SMAWK algorithm is from Agarwal, Klawe, Moran, Shor, and
/// Wilbur, *Geometric applications of a matrix searching algorithm*,
/// Algorithmica 2, pp. 195-208 (1987) and the code here is a
/// translation [David Eppstein's Python code][pads].
///
/// Running time on an *m* × *n* matrix: O(*m* + *n*).
///
/// # Examples
///
/// ```
/// use smawk::Matrix;
/// let matrix = vec![vec![4, 2, 4, 3],
///                   vec![5, 3, 5, 3],
///                   vec![5, 3, 3, 1]];
/// assert_eq!(smawk::column_minima(&matrix),
///            vec![0, 0, 2, 2]);
/// ```
///
/// # Panics
///
/// It is an error to call this on a matrix with zero rows.
///
/// [SMAWK algorithm]: https://en.wikipedia.org/wiki/SMAWK_algorithm
/// [pads]: https://github.com/jfinkels/PADS/blob/master/pads/smawk.py
pub fn column_minima<T: PartialOrd + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    let mut minima = vec![0; matrix.ncols()];
    let (mut rows_scratch, mut cols_scratch) = scratchpads(matrix.nrows(), matrix.ncols());
    smawk_inner(
        &|i, j| matrix.index(i, j),
        (&mut rows_scratch, 0, matrix.nrows()),
        (&mut cols_scratch, 0, matrix.ncols()),
        &mut minima,
    );
    minima
}

#[deprecated(since = "0.3.2", note = "Please use `column_minima` instead.")]
pub fn smawk_column_minima<T: PartialOrd + Copy, M: Matrix<T>>(matrix: &M) -> Vec<usize> {
    column_minima(matrix)
}

/// Pre-allocate empty row and column scratchpads with the capacity
/// needed for the SMAWK algorithm.
///
/// ### Scratchpad Capacity
///
/// At each recursion level of `smawk_inner` with `R` rows and `C` columns:
///
/// 1. We build a stack of survivors by appending up to `C` elements to `rows_scratch`.
/// 2. We select odd columns by appending `C / 2` elements to `cols_scratch`.
/// 3. We recurse with `R' <= C` rows and `C' = C / 2` columns.
///
/// Summing the maximum elements appended across all recursion levels:
///
/// - `rows_scratch` capacity: `R + C + C/2 + C/4 + ... < R + 2 * C`
/// - `cols_scratch` capacity: `C + C/2 + C/4 + C/8 + ... < 2 * C`
///
/// This mathematical guarantee ensures that the scratchpads never need to reallocate/grow.
#[inline(always)]
fn scratchpads_empty(nrows: usize, ncols: usize) -> (Vec<usize>, Vec<usize>) {
    let rows_scratch = vec![0; nrows + 2 * ncols];
    let cols_scratch = vec![0; ncols * 2];
    (rows_scratch, cols_scratch)
}

/// Pre-allocate and populate the row and column scratchpads for the
/// SMAWK algorithm.
#[inline(always)]
fn scratchpads(nrows: usize, ncols: usize) -> (Vec<usize>, Vec<usize>) {
    let (mut rows_scratch, mut cols_scratch) = scratchpads_empty(nrows, ncols);
    for (i, val) in rows_scratch.iter_mut().enumerate().take(nrows) {
        *val = i;
    }
    for (i, val) in cols_scratch.iter_mut().enumerate().take(ncols) {
        *val = i;
    }
    (rows_scratch, cols_scratch)
}

/// Compute column minima in the given area of the matrix. The
/// `minima` slice is updated inplace.
fn smawk_inner<T: PartialOrd + Copy, M: Fn(usize, usize) -> T>(
    matrix: &M,
    (rows_scratch, rows_start, rows_end): (&mut [usize], usize, usize),
    (cols_scratch, cols_start, cols_end): (&mut [usize], usize, usize),
    minima: &mut [usize],
) {
    if cols_start == cols_end {
        return;
    }

    let cols_len = cols_end - cols_start;
    let stack_start = rows_end;
    let odd_cols_start = cols_end;
    let odd_cols_len = cols_len / 2;

    // Upfront assertions using the maximum index trick. Asserting the
    // maximum index accessed in each loop/vector beforehand allows
    // LLVM to prove that all smaller indices accessed in the loop are
    // in bounds, eliminating individual bounds checking inside the
    // loops.
    assert!(rows_end <= rows_scratch.len());
    assert!(stack_start + cols_len <= rows_scratch.len());
    assert!(cols_start + 2 * odd_cols_len <= cols_scratch.len());
    assert!(odd_cols_start + odd_cols_len <= cols_scratch.len());

    let mut stack_len = 0;
    for i in rows_start..rows_end {
        let r = rows_scratch[i];
        while stack_len > 0 {
            let stack_top = rows_scratch[stack_start + stack_len - 1];
            let col = cols_scratch[cols_start + stack_len - 1];
            if matrix(stack_top, col) > matrix(r, col) {
                stack_len -= 1;
            } else {
                break;
            }
        }
        if stack_len < cols_len {
            rows_scratch[stack_start + stack_len] = r;
            stack_len += 1;
        }
    }

    for idx in 0..odd_cols_len {
        let col = cols_scratch[cols_start + 2 * idx + 1];
        cols_scratch[odd_cols_start + idx] = col;
    }

    smawk_inner(
        matrix,
        (&mut *rows_scratch, stack_start, stack_start + stack_len),
        (
            &mut *cols_scratch,
            odd_cols_start,
            odd_cols_start + odd_cols_len,
        ),
        minima,
    );

    let mut r = 0;
    // Assert the final stack boundary before the loop.
    assert!(stack_start + stack_len <= rows_scratch.len());
    for c in 0..cols_len {
        if c % 2 == 0 {
            let col = cols_scratch[cols_start + c];
            let mut row = rows_scratch[stack_start + r];
            let last_row = if c == cols_len - 1 {
                rows_scratch[stack_start + stack_len - 1]
            } else {
                minima[cols_scratch[cols_start + c + 1]]
            };
            let mut pair = (matrix(row, col), row);
            while row != last_row && r < stack_len - 1 {
                r += 1;
                row = rows_scratch[stack_start + r];
                if (matrix(row, col), row) < pair {
                    pair = (matrix(row, col), row);
                }
            }
            minima[col] = pair.1;
        }
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
pub fn online_column_minima<T: Copy + PartialOrd, M: Fn(&[(usize, T)], usize, usize) -> T>(
    initial: T,
    size: usize,
    matrix: M,
) -> Vec<(usize, T)> {
    let mut result = Vec::with_capacity(size);
    result.push((0, initial));

    // State used by the algorithm.
    let mut finished = 0;
    let mut base = 0;
    let mut tentative = 0;

    // Shorthand for evaluating the matrix.
    macro_rules! m {
        ($i:expr, $j:expr) => {{
            matrix(&result[..finished + 1], $i, $j)
        }};
    }

    let (mut rows_scratch, mut cols_scratch) = scratchpads_empty(size, size);
    let mut minima = vec![0; size];

    // Keep going until we have finished all size columns. Since the
    // columns are zero-indexed, we're done when finished == size - 1.
    while finished < size - 1 {
        // First case: we have already advanced past the previous
        // tentative value. We make a new tentative value by applying
        // smawk_inner to the largest square submatrix that fits under
        // the base.
        let i = finished + 1;
        if i > tentative {
            let rows_start = 0;
            let rows_end = finished + 1 - base;
            for (idx, r) in (base..finished + 1).enumerate() {
                rows_scratch[idx] = r;
            }

            tentative = core::cmp::min(finished + rows_end, size - 1);

            let cols_start = 0;
            let cols_end = tentative - finished;
            for (idx, c) in (finished + 1..tentative + 1).enumerate() {
                cols_scratch[idx] = c;
            }

            smawk_inner(
                &|i, j| m![i, j],
                (&mut rows_scratch, rows_start, rows_end),
                (&mut cols_scratch, cols_start, cols_end),
                &mut minima,
            );

            for col in finished + 1..tentative + 1 {
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

    #[test]
    fn smawk_1x1() {
        let matrix = vec![vec![2]];
        assert_eq!(row_minima(&matrix), vec![0]);
        assert_eq!(column_minima(&matrix), vec![0]);
    }

    #[test]
    fn smawk_2x1() {
        let matrix = vec![
            vec![3], //
            vec![2],
        ];
        assert_eq!(row_minima(&matrix), vec![0, 0]);
        assert_eq!(column_minima(&matrix), vec![1]);
    }

    #[test]
    fn smawk_1x2() {
        let matrix = vec![vec![2, 1]];
        assert_eq!(row_minima(&matrix), vec![1]);
        assert_eq!(column_minima(&matrix), vec![0, 0]);
    }

    #[test]
    fn smawk_2x2() {
        let matrix = vec![
            vec![3, 2], //
            vec![2, 1],
        ];
        assert_eq!(row_minima(&matrix), vec![1, 1]);
        assert_eq!(column_minima(&matrix), vec![1, 1]);
    }

    #[test]
    fn smawk_3x3() {
        let matrix = vec![
            vec![3, 4, 4], //
            vec![3, 4, 4],
            vec![2, 3, 3],
        ];
        assert_eq!(row_minima(&matrix), vec![0, 0, 0]);
        assert_eq!(column_minima(&matrix), vec![2, 2, 2]);
    }

    #[test]
    fn smawk_4x4() {
        let matrix = vec![
            vec![4, 5, 5, 5], //
            vec![2, 3, 3, 3],
            vec![2, 3, 3, 3],
            vec![2, 2, 2, 2],
        ];
        assert_eq!(row_minima(&matrix), vec![0, 0, 0, 0]);
        assert_eq!(column_minima(&matrix), vec![1, 3, 3, 3]);
    }

    #[test]
    fn smawk_5x5() {
        let matrix = vec![
            vec![3, 2, 4, 5, 6],
            vec![2, 1, 3, 3, 4],
            vec![2, 1, 3, 3, 4],
            vec![3, 2, 4, 3, 4],
            vec![4, 3, 2, 1, 1],
        ];
        assert_eq!(row_minima(&matrix), vec![1, 1, 1, 1, 3]);
        assert_eq!(column_minima(&matrix), vec![1, 1, 4, 4, 4]);
    }

    #[test]
    fn online_1x1() {
        let matrix = [[0]];
        let minima = vec![(0, 0)];
        assert_eq!(online_column_minima(0, 1, |_, i, j| matrix[i][j]), minima);
    }

    #[test]
    fn online_2x2() {
        let matrix = [
            [0, 2], //
            [0, 0],
        ];
        let minima = vec![(0, 0), (0, 2)];
        assert_eq!(online_column_minima(0, 2, |_, i, j| matrix[i][j]), minima);
    }

    #[test]
    fn online_3x3() {
        let matrix = [
            [0, 4, 4], //
            [0, 0, 4],
            [0, 0, 0],
        ];
        let minima = vec![(0, 0), (0, 4), (0, 4)];
        assert_eq!(online_column_minima(0, 3, |_, i, j| matrix[i][j]), minima);
    }

    #[test]
    fn online_4x4() {
        let matrix = [
            [0, 5, 5, 5], //
            [0, 0, 3, 3],
            [0, 0, 0, 3],
            [0, 0, 0, 0],
        ];
        let minima = vec![(0, 0), (0, 5), (1, 3), (1, 3)];
        assert_eq!(online_column_minima(0, 4, |_, i, j| matrix[i][j]), minima);
    }

    #[test]
    fn online_5x5() {
        let matrix = [
            [0, 2, 4, 6, 7],
            [0, 0, 3, 4, 5],
            [0, 0, 0, 3, 4],
            [0, 0, 0, 0, 4],
            [0, 0, 0, 0, 0],
        ];
        let minima = vec![(0, 0), (0, 2), (1, 3), (2, 3), (2, 4)];
        assert_eq!(online_column_minima(0, 5, |_, i, j| matrix[i][j]), minima);
    }

    #[test]
    fn smawk_works_with_partial_ord() {
        let matrix = vec![
            vec![3.0, 2.0], //
            vec![2.0, 1.0],
        ];
        assert_eq!(row_minima(&matrix), vec![1, 1]);
        assert_eq!(column_minima(&matrix), vec![1, 1]);
    }

    #[test]
    fn online_works_with_partial_ord() {
        let matrix = [
            [0.0, 2.0], //
            [0.0, 0.0],
        ];
        let minima = vec![(0, 0.0), (0, 2.0)];
        assert_eq!(online_column_minima(0.0, 2, |_, i, j| matrix[i][j]), minima);
    }
}

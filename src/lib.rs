extern crate ndarray;

use ndarray::{ArrayView1, ArrayView2};

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

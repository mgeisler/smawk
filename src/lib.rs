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


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

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
        let matrix = arr2(
            &[
                [7, 7, 3, 5, 0],
                [7, 7, 5, 5, 9],
                [0, 0, 3, 1, 3],
                [3, 4, 5, 6, 7],
                [7, 6, 5, 5, 6],
            ],
        );
        assert_eq!(brute_force_row_minima(&matrix.view()), vec![4, 2, 0, 0, 2]);
    }

}

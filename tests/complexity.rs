extern crate ndarray;
extern crate rand;
extern crate smawk;

use ndarray::{Array1, Array2, Axis};
use rand::XorShiftRng;
use smawk::{online_column_minima, random_monge_matrix};
use std::borrow::Borrow;

#[derive(Debug)]
struct LinRegression {
    alpha: f64,
    beta: f64,
    r_squared: f64,
}

/// Square an expression. Works equally well for floats and matrices.
macro_rules! squared {
    ($x:expr) => {
        $x * $x
    };
}

/// Compute the mean of a 1-dimensional array.
fn mean<T, A>(a: A) -> T
where
    T: ndarray::LinalgScalar,
    A: Borrow<Array1<T>>,
{
    a.borrow().mean_axis(Axis(0))[[]]
}

/// Compute a simple linear regression from the list of values.
///
/// See <https://en.wikipedia.org/wiki/Simple_linear_regression>.
fn linear_regression(values: &[(usize, i32)]) -> LinRegression {
    let xs = Array1::from_iter(values.iter().map(|&(x, _)| x as f64));
    let ys = Array1::from_iter(values.iter().map(|&(_, y)| y as f64));

    let xs_mean = mean(&xs);
    let ys_mean = mean(&ys);
    let xs_ys_mean = mean(&xs * &ys);

    let cov_xs_ys = ((&xs - xs_mean) * (&ys - ys_mean)).scalar_sum();
    let var_xs = squared!(&xs - xs_mean).scalar_sum();

    let beta = cov_xs_ys / var_xs;
    let alpha = ys_mean - beta * xs_mean;
    let r_squared = squared!(xs_ys_mean - xs_mean * ys_mean)
        / ((mean(&xs * &xs) - squared!(xs_mean)) * (mean(&ys * &ys) - squared!(ys_mean)));

    LinRegression {
        alpha: alpha,
        beta: beta,
        r_squared: r_squared,
    }
}

/// Check that the number of matrix accesses in `online_column_minima`
/// grows as O(*n*) for *n* ✕ *n* matrix.
#[test]
fn online_linear_complexity() {
    let mut rng = XorShiftRng::new_unseeded();
    let mut data = vec![];

    for &size in &[1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
        let mut matrix: Array2<i32> = random_monge_matrix(size, size, &mut rng);
        let count = std::cell::RefCell::new(0);
        online_column_minima(0, size, |_, i, j| {
            *count.borrow_mut() += 1;
            matrix[[i, j]]
        });
        data.push((size, count.into_inner()));
    }

    let lin_reg = linear_regression(&data);
    assert!(
        lin_reg.r_squared > 0.95,
        "r² = {:.4} is lower than expected for a linear fit\nData points: {:?}\n{:?}",
        lin_reg.r_squared,
        data,
        lin_reg
    );
}

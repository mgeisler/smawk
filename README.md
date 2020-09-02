# SMAWK Algorithm in Rust

[![](https://github.com/mgeisler/smawk/workflows/build/badge.svg)][build-status]
[![](https://codecov.io/gh/mgeisler/smawk/branch/master/graph/badge.svg)][codecov]
[![](https://img.shields.io/crates/v/smawk.svg)][crates-io]
[![](https://docs.rs/smawk/badge.svg)][api-docs]

This crate contains an implementation of the [SMAWK algorithm][smawk]
for finding the smallest element per row in a totally monotone matrix.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
smawk = "0.2"
```

You can now efficiently find row and column minima. Here is an example
where we find the column minima:

```rust
use smawk::smawk_column_minima;

let matrix = vec![
    vec![3, 2, 4, 5, 6],
    vec![2, 1, 3, 3, 4],
    vec![2, 1, 3, 3, 4],
    vec![3, 2, 4, 3, 4],
    vec![4, 3, 2, 1, 1],
];
let minima = vec![1, 1, 4, 4, 4];
assert_eq!(smawk_column_minima(&matrix), minima);
```

The `minima` vector gives the index of the minimum value per column,
so `minima[0] == 1` since the minimum value in the first column is 2
(row 1). Note that the smallest row index is returned.

### Cargo Features

This crate has an optional dependency on the [`ndarray`
crate](https://docs.rs/ndarray/), which provides an efficient matrix
implementation. Enable the `ndarray` Cargo feature to use it.

## Documentation

**[API documentation][api-docs]**

## Changelog

### Version 0.2.0 (2020-07-29)

* [#18](https://github.com/mgeisler/smawk/pull/18): Make
  `online_column_minima` generic in matrix type.
* [#23](https://github.com/mgeisler/smawk/pull/23): Switch to the
  [Rust 2018][rust-2018] edition. We test against the latest stable
  and nightly version of Rust.
* [#29](https://github.com/mgeisler/smawk/pull/29): Drop strict Rust
  2018 compatibility by not testing with Rust 1.31.0.
* [#32](https://github.com/mgeisler/smawk/pull/32): Fix crash on
  overflow in `is_monge`.
* [#33](https://github.com/mgeisler/smawk/pull/33): Update `rand`
  dependency to latest version and get rid of `rand_derive`.
* [#34](https://github.com/mgeisler/smawk/pull/34): Bump `num-traits`
  and `version-sync` dependencies to latest versions.
* [#35](https://github.com/mgeisler/smawk/pull/35): Drop unnecessary
  Windows tests. The assumption is that the numeric computations we do
  are cross-platform.
* [#36](https://github.com/mgeisler/smawk/pull/36): Update `ndarray`
  dependency to the latest version.
* [#37](https://github.com/mgeisler/smawk/pull/37): Automate
  publishing new releases to crates.io.

### Version 0.1.0 — August 7th, 2018

First release with the classical offline SMAWK algorithm as well as a
newer online version where the matrix entries can depend on previously
computed column minima.

## License

SMAWK can be distributed according to the [MIT license][mit].
Contributions will be accepted under the same license.

[build-status]: https://github.com/mgeisler/smawk/actions?query=branch%3Amaster+workflow%3Abuild
[crates-io]: https://crates.io/crates/smawk
[codecov]: https://codecov.io/gh/mgeisler/smawk
[smawk]: https://en.wikipedia.org/wiki/SMAWK_algorithm
[api-docs]: https://docs.rs/smawk/
[rust-2018]: https://doc.rust-lang.org/edition-guide/rust-2018/
[mit]: LICENSE

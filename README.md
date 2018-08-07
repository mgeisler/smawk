# SMAWK Algorithm in Rust

[![](https://travis-ci.org/mgeisler/smawk.svg)][travis-ci]
[![](https://ci.appveyor.com/api/projects/status/github/mgeisler/smawk?branch=master&svg=true)][appveyor]
[![](https://codecov.io/gh/mgeisler/smawk/branch/master/graph/badge.svg)][codecov]

This crate contains an implementation of the [SMAWK algorithm][smawk]
for finding the smallest element per row in a totally monotone matrix.

## Usage

Add this to your `Cargo.toml`:
```toml
[dependencies]
smawk = "0.1"
```
and this to your crate root:
```rust
extern crate smawk;
```

You can now efficiently find row and column minima. Here is an example
where we find the column minima:

```rust
extern crate ndarray;
extern crate smawk;

use ndarray::arr2;
use smawk::smawk_column_minima;

let matrix = arr2(&[
    [3, 2, 4, 5, 6],
    [2, 1, 3, 3, 4],
    [2, 1, 3, 3, 4],
    [3, 2, 4, 3, 4],
    [4, 3, 2, 1, 1],
]);
let minima = vec![1, 1, 4, 4, 4];
assert_eq!(smawk_column_minima(&matrix), minima);
```

The `minima` vector gives the index of the minimum value per column,
so `minima[0] == 1` since the minimum value in the first column is 2
(row 1). Note that the smallest row index is returned.

## Documentation

**[API documentation][api-docs]**

## Release History

This is a changelog describing the most important changes per release.

### Version 0.1.0 — August 7th, 2018

First release with the classical offline SMAWK algorithm as well as a
newer online version where the matrix entries can depend on previously
computed column minima.

## License

SMAWK can be distributed according to the [MIT license][mit].
Contributions will be accepted under the same license.

[travis-ci]: https://travis-ci.org/mgeisler/smawk
[appveyor]: https://ci.appveyor.com/project/mgeisler/smawk
[codecov]: https://codecov.io/gh/mgeisler/smawk
[smawk]: https://en.wikipedia.org/wiki/SMAWK_algorithm
[api-docs]: https://docs.rs/smawk/
[mit]: LICENSE

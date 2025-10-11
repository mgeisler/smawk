# Project Overview

The `smawk` crate offers a highly efficient Rust implementation of the **SMAWK
algorithm**. This algorithm is designed to find the smallest element in each row
of a special type of matrix known as a **totally monotone matrix**.

The key advantage of the SMAWK algorithm is its ability to reduce the time
complexity of certain problems from O(n²) to O(n), transforming a potentially
slow quadratic-time operation into a much faster linear-time one. A practical
application of this is finding optimal line breaks in a paragraph of text, a
feature utilized by the `textwrap` crate.

## Core Concepts

- **Totally Monotone Matrix**: A matrix where every 2x2 submatrix
  `[[a, b], [c, d]]` satisfies `a + d <= b + c`. The algorithms in this crate
  are optimized for these matrices.
- **Monge Property**: A condition stronger than total monotonicity, where for
  any `i < i'` and `j < j'`, `M[i, j] + M[i', j'] <= M[i, j'] + M[i', j]`.
  Matrices with this property are guaranteed to be totally monotone.

## Key Features

- **SMAWK Algorithm**: The primary offering of the crate, providing efficient
  computation of row and column minima.
- **Online Algorithm**: An "online" version of the column minima algorithm is
  available, where matrix values can be computed based on previously determined
  minima.
- **Matrix Abstraction**: A `Matrix` trait allows the algorithms to work with
  different matrix representations, with a default implementation for
  `Vec<Vec<T>>`.
- **`ndarray` Integration**: Optional integration with the popular `ndarray`
  crate for efficient matrix operations.

## Building and Running the Project

The `smawk` crate follows standard Rust and Cargo conventions.

### Building

To build the project, use the following command:

```bash
cargo build
```

To build with `ndarray` support:

```bash
cargo build --features ndarray
```

### Running Tests

Execute the test suite with:

```bash
cargo test
```

To ensure all features are tested, run:

```bash
cargo test --all-features
```

This will run all unit and integration tests, ensuring the correctness of the
algorithms.

### Running Benchmarks

The crate includes benchmarks to compare the performance of different
algorithms. To run them:

```bash
cargo bench
```

## Development Conventions

The `smawk` project adheres to modern Rust development practices.

### Code Style

- **Formatting**: The code is formatted using `dprint`, which is configured in
  `dprint.json`. It invokes tools like `rustfmt` to format the Rust code.
- **Clippy**: The project uses `clippy` for linting and enforcing idiomatic
  Rust.

### Testing

- **Unit Tests**: Each module contains a `tests` submodule with unit tests for
  the functions in that module.
- **Integration Tests**: The `tests` directory contains integration tests that
  cover the public API of the crate.
- **Brute-Force and Recursive Implementations**: The `brute_force.rs` and
  `recursive.rs` modules provide alternative, less efficient implementations of
  the row and column minima algorithms. These are used for testing the
  correctness of the main SMAWK algorithm.

### Continuous Integration

The project uses GitHub Actions for continuous integration. The workflows in
`.github/workflows` define the CI pipeline, which includes:

- **Building and Testing**: On every push and pull request.
- **Code Coverage**: Measured and uploaded to Codecov.
- **Release Automation**: Automated publishing of new releases to crates.io.

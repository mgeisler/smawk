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
- **`no_std` Compatibility**: The core library is `#![no_std]` compatible,
  making it suitable for embedded environments. Maintain this compatibility and
  ensure no `std`-only dependencies are introduced to the core library.

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

The crate uses [Divan](https://github.com/nvzqz/divan) for benchmarking, which
measures both execution time and memory allocations.

To run all benchmarks, use:

```bash
cargo bench --features ndarray
```

#### Available Benchmark Suites

1. **`comparison`**: Compares the brute force, recursive, and SMAWK
   implementations of `row_minima` and `column_minima` across different matrix
   sizes (`50`, `100`, `200`, `400`, `800`).
2. **`online`**: Benchmarks the `online_column_minima` algorithm across the same
   sizes.

#### Running Specific Benchmarks

You can run a specific benchmark suite or filter by benchmark name and const
sizes:

```bash
# Run only the online benchmark suite
cargo bench --bench online --features ndarray

# Filter to run a specific size (e.g., size 800) in the online benchmark
cargo bench --bench online --features ndarray -- smawk_online_column_minima/800
```

## Development Conventions

The `smawk` project adheres to modern Rust development practices.

### Code Style

- **Formatting**: The code is formatted using `dprint`, which is configured in
  `dprint.json`. It invokes tools like `rustfmt` to format the Rust code.
- **Clippy**: The project uses `clippy` for linting and enforcing idiomatic
  Rust.

### Code Safety

- **No Panics**: Avoid introducing code that can panic, such as using
  `.unwrap()` on `Option` or `Result` types. The goal is to write very safe code
  with no runtime crashes.

## Performance and Optimization Invariants

The primary focus of optimization in this crate is the **online SMAWK
algorithm** (`online_column_minima`), which is critical for downstream consumers
like `textwrap`.

This is because optimal paragraph layout (such as the Knuth-Plass line-breaking
algorithm) is solved using dynamic programming, where the cost of breaking a
line at word $j$ depends on the already-computed optimal breaking costs of all
words $i < j$. Because these cost values are computed dynamically based on
previous column minima, the matrix cannot be populated upfront. It must be
queried online, making `online_column_minima` the only algorithm applicable.

The library is optimized to achieve O(1) heap allocation and O(n) execution
speed. To maintain these performance characteristics:

- **Zero-Cost Abstractions**: The `Matrix` trait and lookup closures are generic
  parameters (`M`). This allows Rust to monomorphize the implementation at
  compile time and inline the caller's indexing logic directly into the inner
  loops, eliminating closure allocation and function call overhead.
- **Inlining**: Internal helper functions (like `scratchpads` and
  `scratchpads_empty`) must be marked `#[inline(always)]` to ensure LLVM
  flattens their allocations into the callers.
- **Bounds Check Elimination**: Because the index math inside `smawk_inner`
  depends on runtime matrix comparisons, LLVM's loop optimizer (Scalar
  Evolution) cannot automatically prove that array lookups stay in bounds. We
  use **upfront maximum index assertions** at the start of the function.
  Asserting the maximum index accessed in each vector beforehand allows LLVM to
  prove all smaller index accesses in the loops are in bounds, eliminating
  bounds checking inside the hot loops.
- **Direct Call Sites**: Prefer calling generic functions directly rather than
  wrapping them in unnecessary closures or macros when possible, keeping the
  code simple and avoiding borrow checker complications.

### Unsafe Performance Gap

An experimental version utilizing fully unchecked indexing (using
`get_unchecked` and `uget` throughout the matrix and scratchpad accesses) was
tested. This unsafe version achieved a median runtime of **26.6 µs** (a **~25%
reduction in runtime** compared to the safe optimized version's **35.6 µs**).

This gap exists because LLVM cannot safely eliminate the remaining bounds checks
in the matrix coordinate lookups and slice indices. These lookups rely on
indices that are **values loaded from memory** (e.g., loading `stack_top` from
`rows_scratch` and passing it to `matrix(stack_top, col)`). Even though we
asserted the scratchpad bounds beforehand, LLVM cannot statically prove that the
_values_ stored inside the mutable scratchpad memory are less than the matrix
dimensions or the length of the `minima` slice. Tracking value-range invariants
of dynamically populated arrays is beyond LLVM's static analysis, making these
remaining bounds checks inevitable in safe Rust.

#### Potential Avenues to Close the Gap

To safely eliminate the remaining data-dependent bounds checks without resorting
to unsafe code, future revisions could investigate:

- **Dynamic Re-slicing (Sub-slicing)**: Instead of tracking a mutable integer
  `stack_len` to index the scratchpads, represent the active stack and columns
  as subslices (e.g.,
  `active_stack = &mut rows_scratch[stack_start..stack_start + stack_len]`). By
  shrinking these subslices during a pop (e.g.,
  `active_stack = &mut active_stack[..len - 1]`), we can access the stack top
  via `active_stack[len - 1]`. Because `len - 1` is always less than the slice's
  own length `len` (when non-empty), LLVM can statically prove bounds safety on
  every iteration without needing to track the data-dependent loop invariants.
- **Consolidating the Scratchpads**: Storing row and column index pairs together
  rather than in separate `rows_scratch` and `cols_scratch` arrays. This would
  halve the number of bounds check branches by retrieving both values in a
  single index lookup.
- **Safe Cursor Abstraction**: Wrapping the scratchpad access in a simple, local
  cursor struct that uses raw pointers internally but exposes a safe interface
  to `smawk_inner`, isolating the unsafe code to a tiny, verified boundary.

## Verifying Assembly and Bounds Check Elimination

When optimizing core hot loops, you must verify that LLVM successfully optimizes
away bounds checks and panic branches. Do not guess; compile and analyze the raw
assembly output.

#### 1. Emitting Assembly

To emit assembly for the benchmarks (where the generic functions are
monomorphized with concrete types), run:

```bash
cargo rustc --release --bench online --features ndarray -- --emit=asm
```

_Note: Because the core SMAWK functions (`smawk_inner`, `row_minima`,
`column_minima`) are generic over the element type `T` and the matrix type `M`,
Rust cannot generate machine code/assembly for them when compiling the library
itself. They are only monomorphized and compiled to machine code inside the
final binaries (like benchmarks, tests, or examples) that instantiate them with
concrete types. Therefore, you must always inspect the assembly of a concrete
caller (like the benchmarks)._

#### 2. Locating Assembly Files

The generated assembly files (`.s`) are stored in the Cargo target output
directory. You can locate them using:

```bash
ls target/release/deps/online-*.s
# or
ls target/release/deps/libsmawk-*.s
```

#### 3. Analyzing the Assembly

When a bounds check fails or an assertion is triggered in Rust, it branches to
panic/bounds-check handlers. Search the assembly file for these indicator
symbols:

- **`panic_bounds_check`**: Called when a slice/array indexing operation is out
  of bounds (e.g., `core::panicking::panic_bounds_check`).
- **`panic` / `panic_fmt`**: Standard panic entry points.
- **`ud2`**: The "Undefined Instruction" opcode. In release/optimized builds, if
  LLVM can prove a panic path is inevitable or if panic-on-abort is configured,
  it replaces the call to the panic handler with a `ud2` trap instruction.

You can run `grep` to check for occurrences:

```bash
grep -E "panic_bounds_check|panic_fmt|ud2" target/release/deps/online-*.s
```

#### 4. Isolating Specific Functions

Because some panics might remain in other parts of the binary (like benchmark
setup/teardown code), the presence of panic symbols in the file does not
necessarily mean the hot loop has them. You must verify the specific function:

1. Search the assembly file for the mangled function name label (e.g.,
   `_ZN5smawk11smawk_inner` or simply `smawk_inner`).
2. Read the assembly instructions block until the next function label.
3. Verify that there are no `call` instructions targeting panic functions and no
   `ud2` instructions inside the block of instructions for that function.

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

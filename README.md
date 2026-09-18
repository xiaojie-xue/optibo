<!-- markdownlint-disable MD033 MD041 -->

<div align="center">

<h1>Optibo</h1>

<p><strong>A family of optimization libraries for Rust</strong></p>

<p>
  <strong>English</strong> | <a href="README_zh.md">简体中文</a>
</p>

<p>
  <a href="https://github.com/xiaojie-xue/optibo/actions/workflows/rust.yml"><img alt="CI" src="https://github.com/xiaojie-xue/optibo/actions/workflows/rust.yml/badge.svg?branch=main"></a>
  <a href="https://crates.io/crates/optibo"><img alt="crates.io" src="https://img.shields.io/crates/v/optibo.svg?color=CE422B&amp;logo=rust&amp;logoColor=white"></a>
  <a href="https://github.com/xiaojie-xue/optibo/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

</div>

Optibo is a growing family of optimization libraries for Rust. Its current
Differential Evolution implementation supports
[Calibo](https://github.com/xiaojie-xue/calibo), a robot calibration tool, by fitting
model parameters to identify data. Future development will add other optimization
algorithms to support additional capabilities and applications beyond robot calibration.

## Quick start

Add Optibo from crates.io to your `Cargo.toml`. Differential evolution is enabled by default:

```toml
[dependencies]
optibo = "0.1.0"
```

```rust
use optibo::de::{minimize, Config};

fn main() -> Result<(), optibo::de::DeError> {
    let config = Config { seed: 42, atol: 1e-12, ..Config::default() };
    let result = minimize(&[(-5.0, 5.0); 2], &config, |x| {
        x.iter().map(|v| v * v).sum()
    })?;
    println!("{:?}: {:?}, cost={}", result.termination, result.x, result.fun);
    Ok(())
}
```

## Features

### Differential Evolution: `de`

The only optimization algorithm feature currently available is `de`, which provides
**differential evolution (DE)** through `optibo::de`. DE searches for parameters that
minimize an objective within specified bounds, using a population of candidate solutions.
It requires no gradients and is useful for nonlinear parameter estimation, including
minimizing the error between a robot model's predictions and calibration measurements.

The implementation supports:

- `best1bin` and `rand1bin` strategies, with seeded runs for reproducibility.
- Parameter bounds and fixed parameters, with configurable evaluation and generation limits.
- Scalar and batch objectives, plus progress callbacks and cancellation.
- Optional parallel scalar evaluation through the `parallel` feature.

`de` enables differential evolution. It is enabled by default, runs serially by
default, and adds no third-party dependencies. The quick start above uses this configuration.

### Optional acceleration: `parallel`

When individual objective evaluations are expensive, enable `parallel` to use
Rayon to evaluate multiple candidates across CPU threads. This accelerates
differential evolution; it does not add another optimization algorithm.

Parallel evaluation requires two steps:

1. Enable `parallel` in the dependency, which also enables `de`:

   ```toml
   [dependencies]
   optibo = { version = "0.1.0", features = ["parallel"] }
   ```

2. Set `Config::parallel = true` when configuring the solver:

   ```rust
   let config = Config {
       parallel: true,
       ..Config::default()
   };
   ```

Enabling the Cargo feature alone leaves evaluation serial by default. This option
only affects scalar objectives; batch objectives manage their own parallelism.
For inexpensive objectives, scheduling overhead may outweigh the benefit.

The project's Rust code forbids `unsafe`; optional dependencies such as Rayon may
use unsafe code internally.

## Documentation

- [API documentation on docs.rs](https://docs.rs/optibo) (available after the first crates.io release and successful documentation build)

Build and open the API documentation locally with `cargo doc --all-features --no-deps --open`.
The docs.rs build includes all features.

## Development and validation

```bash
cargo test --locked
cargo test --no-default-features --locked
cargo test --no-default-features --features de --locked
cargo test --all-features --locked
```

Tests cover algorithm operators, numerical edge cases, error handling, reproducibility,
serial/parallel consistency, deterministic comparisons with SciPy, and a synthetic calibration problem.
GitHub Actions checks feature combinations, formatting, and Clippy.

## License

Licensed under the [MIT License](LICENSE).

//! Optibo provides bounded numerical optimization in safe Rust.
//!
//! The currently available optimizer is **differential evolution (DE)**: a
//! population-based method that minimizes a scalar cost without derivatives.
//! Use it for bounded parameter fitting when gradients are unavailable or the
//! objective has several local minima. Each run evaluates many candidates, so
//! set an evaluation budget when model evaluations are expensive.
//!
//! # Fit a model to measurements
//!
//! Add the crate with `cargo add optibo`. The default `de` feature includes the
//! optimizer without third-party dependencies.
//!
//! This example estimates the slope and intercept of `y = a * t + b`. The objective
//! returns the sum of squared residuals; the optimizer receives one bound per
//! parameter, in the same order as the coordinates passed to the closure.
//!
//! ```
//! # #[cfg(feature = "de")]
//! # fn main() -> Result<(), optibo::de::DeError> {
//! use optibo::de::{minimize, Config};
//!
//! let measurements = [(0.0, 1.0), (1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
//! let bounds = [(-5.0, 5.0), (-5.0, 5.0)]; // [slope, intercept]
//! let config = Config {
//!     seed: 42,
//!     max_evaluations: 20_000,
//!     atol: 1e-12,
//!     ..Config::default()
//! };
//! let result = minimize(&bounds, &config, |x| {
//!     measurements.iter().map(|&(t, y)| {
//!         let residual = x[0] * t + x[1] - y;
//!         residual * residual
//!     }).sum()
//! })?;
//!
//! println!("slope={}, intercept={}", result.x[0], result.x[1]);
//! println!("cost={}, stopped={:?}, evaluations={}",
//!     result.fun, result.termination, result.evaluations);
//! assert!(result.fun < 1e-8);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "de"))]
//! # fn main() {}
//! ```
//!
//! Linear least squares has specialized solvers; it is used here to illustrate the
//! same objective interface you would use for a nonlinear simulation or calibration
//! model. Captured measurements can be borrowed: the closure does not need to be
//! `'static`.
//!
//! # Read the result before accepting a fit
//!
//! `Ok(result)` means the solve returned a usable best candidate. It can still have
//! stopped because its budget ran out or a callback cancelled it. Inspect
//! `result.termination`, `result.fun`, and the residuals relevant to your problem.
//!
//! Convergence tests the spread of **population costs**, not distance to the true
//! solution. A flat objective can converge with poorly identified parameters.
//! `result.success()` reports convergence; it does not certify a global optimum.
//! For consequential fits, compare several seeds and validate predictions against
//! measurements not used in the objective.
//!
//! # Find the right entry point
//!
//! All optimizer APIs live in the `optibo::de` module (requires feature `de`).
//!
//! | Task | Entry point / guide |
//! | --- | --- |
//! | Minimize a scalar cost | [`de::minimize`](de/fn.minimize.html) |
//! | Report model or data failures | [`de::minimize_fallible`](de/fn.minimize_fallible.html) |
//! | Track progress or stop a solve | [`de::minimize_with_callback`](de/fn.minimize_with_callback.html) |
//! | Evaluate many candidates together | [`de::BatchObjective`](de/trait.BatchObjective.html) and [`de::minimize_batch`](de/fn.minimize_batch.html) |
//! | Choose budgets, tolerances, and initial values | [`de::Config`](de/struct.Config.html) and the [DE guide](de/index.html#configure-a-run) |
//! | Interpret the final state | [`de::DeResult`](de/struct.DeResult.html) and [`de::Termination`](de/enum.Termination.html) |
//!
//! # Features and parallel evaluation
//!
//! | Cargo feature | Effect |
//! | --- | --- |
//! | `de` (default) | Differential evolution with no third-party dependencies. |
//! | `parallel` | Enables `de` and Rayon-backed scalar objective evaluation. |
//! | No default features | No optimizer modules or third-party dependencies. |
//!
//! To evaluate scalar candidates in parallel, enable the feature:
//!
//! ```text
//! cargo add optibo --features parallel
//! ```
//!
//! Then set `Config { parallel: true, ..Config::default() }`. Both steps are
//! required; enabling the feature alone keeps runs serial. Batch objectives manage
//! their own execution and ignore this setting. See the [parallel guide](de/index.html#parallel-evaluation)
//! for execution and reproducibility details. docs.rs includes all features.
//!
//! # Scope and reproducibility
//!
//! - Only finite box bounds are built in; equal bounds fix a coordinate.
//! - No gradients, general nonlinear constraints, integer search, or automatic local
//!   polishing are provided.
//! - A fixed seed makes a run reproducible when the objective is deterministic.
//!   This is not a promise of identical results across future crate versions.
//! - Authored Rust code forbids unsafe code; optional dependency internals may use it.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Bounded differential evolution. Requires the `de` feature.
#[cfg(feature = "de")]
pub mod de;

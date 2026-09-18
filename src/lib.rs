//! Numerical optimization in safe Rust, with algorithms selected through Cargo features.
//!
//! # Quick start
//!
//! Add `optibo = "0.1"` to your dependencies. Differential evolution is enabled
//! by default and minimizes a scalar objective within box bounds:
//!
//! ```
//! # #[cfg(feature = "de")]
//! # fn main() -> Result<(), optibo::de::DeError> {
//! use optibo::de::{minimize, Config};
//!
//! let config = Config { seed: 42, atol: 1e-12, ..Config::default() };
//! let result = minimize(&[(-5.0, 5.0); 2], &config, |x| {
//!     x.iter().map(|v| v * v).sum()
//! })?;
//! assert!(result.fun < 1e-8);
//! println!("{:?}: {:?}, cost={}", result.termination, result.x, result.fun);
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "de"))]
//! # fn main() {}
//! ```
//!
//! # Cargo features
//!
//! | Feature | Behavior |
//! | --- | --- |
//! | `de` (default) | Differential evolution, with no third-party dependencies. |
//! | `parallel` | Enables `de` and Rayon-backed scalar objective evaluation. |
//! | No features | Builds without optimizer modules or third-party dependencies. |
//!
//! Enable `de` (enabled by default) to use differential evolution via
//! `optibo::de`. The optional `parallel` feature enables `de` and Rayon-backed
//! scalar evaluation. With no default features, no optimizer modules or external
//! dependencies are enabled.
//! To use parallel evaluation, also set `Config::parallel = true` at runtime.
//! docs.rs builds this documentation with all features enabled.
//!
//! # Choosing an API
//!
//! The `de` module provides scalar, fallible scalar, and batch objective APIs,
//! each with an optional progress and cancellation callback. Start with
//! `de::minimize` and `de::Config`, then inspect `de::DeResult` and
//! `de::Termination` to understand why optimization stopped.
//!
//! # Safety and reproducibility
//!
//! Authored Rust code forbids unsafe code. This restriction does not apply to
//! dependency internals: optional Rayon and its dependencies use unsafe internally.
//! A seeded run is reproducible when objective evaluation is deterministic.
//! Convergence measures population dispersion; it does not certify a global minimum.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Bounded differential evolution. Requires the `de` feature.
#[cfg(feature = "de")]
pub mod de;

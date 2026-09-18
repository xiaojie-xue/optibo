//! Optibo provides numerical optimization algorithms behind Cargo features.
//!
//! Enable `de` (enabled by default) to use differential evolution via
//! `optibo::de`. The optional `parallel` feature enables `de` and Rayon-backed
//! scalar evaluation. With no default features, no optimizer modules or external
//! dependencies are enabled.
//!
//! Authored Rust code forbids unsafe code. This restriction does not apply to
//! dependency internals: optional Rayon and its dependencies use unsafe internally.

#![forbid(unsafe_code)]

/// Bounded differential evolution. Requires the `de` feature.
#[cfg(feature = "de")]
pub mod de;

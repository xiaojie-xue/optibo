//! Bounded, deterministic differential evolution with deferred population updates.
//!
//! The algorithm follows the documented `best1bin` / `rand1bin` equations and
//! convergence criterion of SciPy, but is an independent implementation, not a
//! bit-for-bit port. [`Config::population_size`](crate::de::Config::population_size)
//! is an **actual individual count**.
//!
//! ```
//! use optibo::de::{minimize, Config};
//! let config = Config { seed: 42, atol: 1e-12, ..Config::default() };
//! let result = minimize(&[(-5.0, 5.0); 2], &config,
//!     |x| x.iter().map(|v| v * v).sum()).unwrap();
//! assert!(result.fun < 1e-8);
//! ```
//!
//! Only box constraints are built in. Nonfinite objective values reject a
//! candidate; [`EvaluationError`](crate::de::EvaluationError) aborts a solve.
//! Panics are not caught. Objective
//! evaluation must be deterministic to reproduce a seeded run. Parallel scalar
//! evaluation requires the optional `parallel` feature, but all random draws and
//! population selection remain serial and independent of evaluation order.
//!
//! # API guide
//!
//! | Objective | Without callback | With callback |
//! | --- | --- | --- |
//! | Scalar cost | [`minimize`](crate::de::minimize) | [`minimize_with_callback`](crate::de::minimize_with_callback) |
//! | Fallible scalar cost | [`minimize_fallible`](crate::de::minimize_fallible) | [`minimize_fallible_with_callback`](crate::de::minimize_fallible_with_callback) |
//! | Batch costs | [`minimize_batch`](crate::de::minimize_batch) | [`minimize_batch_with_callback`](crate::de::minimize_batch_with_callback) |
//!
//! Bounds contain one finite `(lower, upper)` pair per coordinate. Equal endpoints
//! fix a coordinate. Objectives receive physical coordinates, including fixed
//! ones. Invalid bounds, configuration, or initial populations return [`DeError`](crate::de::DeError).
//!
//! # Progress and cancellation
//!
//! Callbacks receive [`Progress`](crate::de::Progress) after initialization and each complete or partial
//! generation. Return [`Control::Stop`](crate::de::Control::Stop) to keep the current best candidate and
//! finish with [`Termination::Cancelled`](crate::de::Termination::Cancelled):
//!
//! ```
//! use optibo::de::{minimize_with_callback, Config, Control, Termination};
//!
//! let result = minimize_with_callback(
//!     &[(-5.0, 5.0); 2],
//!     &Config::default(),
//!     |x| x.iter().map(|v| v * v).sum(),
//!     |progress| {
//!         if progress.evaluations >= 40 { Control::Stop } else { Control::Continue }
//!     },
//! )?;
//! assert_eq!(result.termination, Termination::Cancelled);
//! # Ok::<(), optibo::de::DeError>(())
//! ```

//! # Configure a run
//!
//! Start with [`Config::default()`](crate::de::Config::default), set finite bounds based on the model, choose
//! an explicit evaluation budget, and set `seed` to reproduce an experiment.
//! The full defaults are listed on [`Config`](crate::de::Config).
//!
//! - **Population and budget:** `population_size` is the total candidate count, not
//!   a multiplier of dimension. With population size `P`, initialization costs `P`
//!   evaluations and each full generation costs another `P`. For example, `P = 40`
//!   and `max_evaluations = 4_000` allow initialization plus at most 99 full
//!   generations. Early convergence or cancellation can reduce this count.
//!   Increasing the population increases coverage and cost per generation.
//! - **Stopping tolerances:** convergence requires all costs to be finite and
//!   `std(costs) <= atol + tol * abs(mean(costs))`, using population standard
//!   deviation. `atol` has the same units as your cost. Near a zero minimum, the
//!   relative term shrinks, so choose an absolute tolerance that reflects the
//!   useful precision of your objective. Neither tolerance is a coordinate tolerance.
//! - **Exploration:** [`Mutation::Dither`](crate::de::Mutation::Dither) varies the differential weight once per
//!   generation. `crossover` controls how many donor coordinates are used, with at
//!   least one free coordinate forced. Start with the defaults before changing
//!   strategy or these parameters.
//! - **Initialization:** the default Latin hypercube samples across each free
//!   coordinate's range. Set `initial_guess` to include a known feasible point, or
//!   use [`Initialization::Population`](crate::de::Initialization::Population) to supply the whole population. A guess
//!   replaces the first member, so its evaluation is included in the normal budget.
//!
//! ## Bounds and fixed parameters
//!
//! Bounds must be nonempty, finite, and ordered `lower <= upper`. Objectives always
//! receive the full vector in physical units, including fixed coordinates:
//!
//! ```
//! use optibo::de::{minimize, Config};
//!
//! let result = minimize(&[(-5.0, 5.0), (2.0, 2.0)], &Config::default(), |x| {
//!     assert_eq!(x[1], 2.0);
//!     (x[0] - 1.0).powi(2) + x[1].powi(2)
//! })?;
//! assert_eq!(result.x[1], 2.0);
//! # Ok::<(), optibo::de::DeError>(())
//! ```
//!
//! If every coordinate is fixed, only one candidate is evaluated. Configuration
//! and any supplied initial population are still validated. Custom population rows
//! must be finite and have the full dimension; out-of-bounds values are clipped.
//! An `initial_guess` must already be within bounds and is not clipped.
//!
//! # Understand termination
//!
//! A returned [`DeResult`](crate::de::DeResult) contains the best candidate even on cancellation or
//! budget exhaustion. An `Err` indicates invalid input or an evaluation failure
//! and does not contain a partial result.
//!
//! | Termination | Meaning and next step |
//! | --- | --- |
//! | [`Termination::Converged`](crate::de::Termination::Converged) | Population cost dispersion met the tolerance, or all coordinates were fixed. Check fit quality separately. |
//! | [`Termination::MaxEvaluations`](crate::de::Termination::MaxEvaluations) | Candidate budget exhausted. Inspect the best result before spending more evaluations. |
//! | [`Termination::MaxGenerations`](crate::de::Termination::MaxGenerations) | Generation limit reached. Inspect progress and consider another seed or a larger budget. |
//! | [`Termination::Cancelled`](crate::de::Termination::Cancelled) | Your callback requested a stop. The best evaluated candidate is retained. |
//!
//! Checks run after initialization and each full or partial generation in this
//! order: callback, convergence, evaluation budget, generation limit. A callback
//! stop therefore wins over convergence. A final partial generation evaluates a
//! prefix of the population, counts its evaluations, and does not increment
//! `generations`. Setting `max_generations = 0` still evaluates initialization.
//!
//! # Invalid candidates versus evaluation failures
//!
//! Return a nonfinite cost (`NaN`, positive infinity, or negative infinity) to reject
//! an individual candidate. If initialization has no finite cost, the solve returns
//! [`DeError::NoFiniteObjective`](crate::de::DeError::NoFiniteObjective). Invalid candidates still consume evaluation budget.
//!
//! Use [`minimize_fallible`](crate::de::minimize_fallible) when the evaluator itself can fail. An
//! [`EvaluationError`](crate::de::EvaluationError) aborts the solve as [`DeError::Evaluation`](crate::de::DeError::Evaluation); it is not treated
//! as a poor candidate. Objective panics propagate.
//!
//! ```
//! use optibo::de::{minimize_fallible, Config, EvaluationError};
//!
//! fn model(x: &[f64]) -> Result<f64, EvaluationError> {
//!     if x[0] <= 0.0 {
//!         return Ok(f64::INFINITY); // outside this model's domain
//!     }
//!     Ok((x[0].ln() - 1.0).powi(2))
//!     // A failed external evaluator would return Err(EvaluationError::new(...)).
//! }
//!
//! let result = minimize_fallible(&[(0.1, 10.0)], &Config::default(), model)?;
//! assert!(result.fun.is_finite());
//! # Ok::<(), optibo::de::DeError>(())
//! ```
//!
//! # Parallel evaluation
//!
//! Enable Cargo feature `parallel` and set [`Config::parallel`](crate::de::Config::parallel) to `true`:
//!
//! ```
//! # #[cfg(feature = "parallel")]
//! # {
//! use optibo::de::{minimize, Config};
//!
//! let config = Config { parallel: true, seed: 42, ..Config::default() };
//! let result = minimize(&[(-5.0, 5.0); 3], &config, |x| {
//!     x.iter().map(|v| v * v).sum()
//! })?;
//! assert!(result.fun.is_finite());
//! # }
//! # Ok::<(), optibo::de::DeError>(())
//! ```
//!
//! This uses Rayon's current thread pool. Scalar closures require `Sync` even for
//! serial runs. Captured immutable data is suitable; call-order-dependent state can
//! break reproducibility. Random draws and population updates remain serial, so
//! serial and parallel runs use the same search trajectory for the same deterministic
//! objective and configuration. Parallel scheduling can cost more than it saves for
//! cheap objectives; measure using your actual model.
//!
//! Callbacks run after evaluation, never inside an individual objective call.
//! Cancellation cannot interrupt a running batch. A parallel batch may finish other
//! candidate evaluations before reporting an evaluator error.
//!
//! # Evaluate a population in batches
//!
//! Implement [`BatchObjective`](crate::de::BatchObjective) to use a vectorized model, a device, or an existing
//! batch service. The input is flattened row-major: for two dimensions,
//! `[x0, y0, x1, y1, ...]`. Write one cost per candidate, in matching order.
//!
//! ```
//! use optibo::de::{minimize_batch, BatchObjective, Config, EvaluationError};
//!
//! struct Sphere;
//! impl BatchObjective for Sphere {
//!     fn evaluate_batch(
//!         &self,
//!         candidates: &[f64],
//!         dimension: usize,
//!         costs: &mut [f64],
//!     ) -> Result<(), EvaluationError> {
//!         for (x, cost) in candidates.chunks_exact(dimension).zip(costs.iter_mut()) {
//!             *cost = x.iter().map(|v| v * v).sum();
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let config = Config { seed: 42, atol: 1e-12, ..Config::default() };
//! let result = minimize_batch(&[(-5.0, 5.0); 2], &config, &Sphere)?;
//! assert!(result.fun < 1e-8);
//! # Ok::<(), optibo::de::DeError>(())
//! ```
//!
//! Do not assume every call contains `population_size` candidates: a final partial
//! generation or an all-fixed problem can be smaller. `candidates.len()` equals
//! `dimension * costs.len()`. Outputs start as `NaN`; unwritten outputs reject their
//! candidates. `Config::parallel` has no effect here; the implementation controls
//! its own threading. Evaluation counts measure candidates, not batch calls.

mod domain;
mod operators;
mod rng;
mod solver;

pub use solver::{
    minimize, minimize_batch, minimize_batch_with_callback, minimize_fallible,
    minimize_fallible_with_callback, minimize_with_callback,
};

use std::fmt;

/// Binomial-crossover differential evolution strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    /// best + F * (r1 - r2)
    Best1Bin,
    /// r0 + F * (r1 - r2)
    Rand1Bin,
}

/// Differential weight F. Dithering draws one value for the entire generation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mutation {
    /// A finite value in [0, 2).
    Fixed(f64),
    /// Uniformly sample [min, max), with 0 <= min < max < 2.
    Dither {
        /// Inclusive lower endpoint.
        min: f64,
        /// Exclusive upper endpoint.
        max: f64,
    },
}

/// Initial population generation, in normalized free-variable coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Initialization {
    /// Independently sample each free coordinate uniformly within its bounds.
    Random,
    /// Sample each stratum once per free coordinate, then shuffle pairings.
    LatinHypercube,
    /// User-supplied rows in physical coordinates, including fixed variables.
    /// At least four rows are required; finite coordinates are clipped to bounds.
    /// Clipped physical coordinates are evaluated without a normalization round-trip.
    /// The row count overrides [`Config::population_size`].
    Population(Vec<Vec<f64>>),
}

/// Validated before any objective evaluation. Defaults use seed 0, LHS, and
/// serial evaluation. No implicit local polishing or constraint penalties occur.
///
/// # Defaults
///
/// | Field | Default |
/// | --- | --- |
/// | `strategy` | [`Strategy::Best1Bin`] |
/// | `population_size` | `40` |
/// | `max_generations` | `1000` |
/// | `max_evaluations` | `usize::MAX` |
/// | `mutation` | `Dither { min: 0.5, max: 1.0 }` |
/// | `crossover` | `0.7` |
/// | `tol` / `atol` | `0.01` / `0.0` |
/// | `seed` | `0` |
/// | `init` | [`Initialization::LatinHypercube`] |
/// | `initial_guess` | `None` |
/// | `parallel` | `false` |
///
/// Use struct update syntax to override only the fields you need:
///
/// ```
/// use optibo::de::Config;
/// let config = Config {
///     seed: 42,
///     max_evaluations: 4_000,
///     atol: 1e-10,
///     ..Config::default()
/// };
/// ```
///
/// See the [configuration guide](index.html#configure-a-run) for budget accounting
/// and the meaning of tolerances.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    /// Mutation strategy. Defaults to [`Strategy::Best1Bin`].
    pub strategy: Strategy,
    /// Actual count, not SciPy's per-free-dimension multiplier. At least four.
    pub population_size: usize,
    /// Number of complete evolutionary generations, excluding initialization.
    pub max_generations: usize,
    /// Candidate evaluation budget, including initialization and invalid costs.
    /// Must cover the initial population (or one candidate when all bounds fixed).
    /// A last partial generation is evaluated in population index order.
    pub max_evaluations: usize,
    /// Differential weight. Defaults to dithering over [0.5, 1.0).
    pub mutation: Mutation,
    /// Binomial crossover probability in [0, 1]. One donor coordinate is forced.
    pub crossover: f64,
    /// Stop when population std <= atol + tol * abs(population mean).
    /// All population costs must be finite. Both tolerances must be finite >= 0.
    pub tol: f64,
    /// Absolute population-dispersion tolerance. Defaults to zero; see [`Self::tol`].
    pub atol: f64,
    /// Seed for the solver's deterministic random number generator. Defaults to zero.
    pub seed: u64,
    /// Initial population construction. Defaults to Latin hypercube sampling.
    pub init: Initialization,
    /// Physical coordinates; replaces the first initial member. Must be in bounds.
    /// Evaluated at the exact supplied coordinates. Subsequent search still uses
    /// normalized f64 coordinates, whose resolution is limited for very wide bounds.
    pub initial_guess: Option<Vec<f64>>,
    /// Rayon scalar evaluation; needs the `parallel` feature. Batch objectives
    /// control their own execution and ignore this flag.
    pub parallel: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            strategy: Strategy::Best1Bin,
            population_size: 40,
            max_generations: 1000,
            max_evaluations: usize::MAX,
            mutation: Mutation::Dither { min: 0.5, max: 1.0 },
            crossover: 0.7,
            tol: 0.01,
            atol: 0.0,
            seed: 0,
            init: Initialization::LatinHypercube,
            initial_guess: None,
            parallel: false,
        }
    }
}

/// Whole-population evaluator. `candidates` contains row-major physical vectors
/// with `dimension` elements each; `costs.len()` is the candidate count. Write
/// every output. Outputs are reset to NaN before each invocation, so unwritten
/// entries are rejected. Calls can contain fewer candidates in the final batch.
///
/// Return nonfinite costs for invalid candidates; return an error for a model or
/// data failure that must abort the optimization. The number of candidates, not
/// the number of batch invocations, counts against the evaluation budget.
pub trait BatchObjective {
    /// Evaluate every row in `candidates`, writing its scalar cost into `costs`.
    /// See the trait documentation for layout and error handling requirements.
    fn evaluate_batch(
        &self,
        candidates: &[f64],
        dimension: usize,
        costs: &mut [f64],
    ) -> Result<(), EvaluationError>;
}

/// An objective-level failure distinct from a single invalid candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationError {
    /// Human-readable description of the objective failure.
    pub message: String,
}

impl EvaluationError {
    /// Create a fatal objective error with a descriptive message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for EvaluationError {}

/// Invalid inputs or a fatal failure encountered while minimizing an objective.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeError {
    /// A configuration value or feature requirement is invalid.
    InvalidConfig(String),
    /// Bounds are empty or contain an invalid endpoint pair.
    InvalidBounds {
        /// Zero-based coordinate index.
        index: usize,
        /// Description of the invalid value.
        reason: String,
    },
    /// The initial population or initial guess has invalid dimensions or values.
    InvalidPopulation(String),
    /// The objective reported a fatal error.
    Evaluation(EvaluationError),
    /// Initialization produced no finite objective value. Supply a valid seed
    /// population or revise the objective domain instead of reporting convergence.
    NoFiniteObjective,
}

impl fmt::Display for DeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid configuration: {message}"),
            Self::InvalidBounds { index, reason } => write!(f, "invalid bound {index}: {reason}"),
            Self::InvalidPopulation(message) => write!(f, "invalid population: {message}"),
            Self::Evaluation(error) => write!(f, "objective evaluation failed: {error}"),
            Self::NoFiniteObjective => {
                f.write_str("initial population has no finite objective value")
            }
        }
    }
}

impl std::error::Error for DeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Evaluation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<EvaluationError> for DeError {
    fn from(error: EvaluationError) -> Self {
        Self::Evaluation(error)
    }
}

/// Why the solver stopped. Convergence is a population-dispersion test, not a
/// certificate of a global minimum or parameter identifiability.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    /// Population dispersion met the tolerance, or all coordinates were fixed.
    Converged,
    /// The configured generation limit was reached.
    MaxGenerations,
    /// The candidate evaluation budget was exhausted.
    MaxEvaluations,
    /// A progress callback requested cancellation.
    Cancelled,
}

/// Best candidate and final population, including the reason optimization stopped.
#[derive(Clone, Debug, PartialEq)]
pub struct DeResult {
    /// Best candidate in physical coordinates, including fixed variables.
    pub x: Vec<f64>,
    /// Objective value at [`Self::x`].
    pub fun: f64,
    /// Physical coordinates, including fixed variables, in population index order.
    pub population: Vec<Vec<f64>>,
    /// Invalid members are represented by positive infinity.
    pub population_energies: Vec<f64>,
    /// Complete generations only. A budget-limited partial generation is excluded.
    pub generations: usize,
    /// Number of evaluated candidates, including initialization and invalid costs.
    pub evaluations: usize,
    /// Reason the solver stopped; inspect this before interpreting the result.
    pub termination: Termination,
}

impl DeResult {
    /// True only when the dispersion criterion was met (or all variables fixed).
    pub fn success(&self) -> bool {
        self.termination == Termination::Converged && self.fun.is_finite()
    }
}

/// Snapshot after initialization and after each full or partial generation.
/// Cancellation is checked at these boundaries, not inside objective evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Progress<'a> {
    /// Current best candidate in physical coordinates.
    pub x: &'a [f64],
    /// Objective value of the current best candidate.
    pub fun: f64,
    /// Number of completed generations, excluding any partial generation.
    pub generations: usize,
    /// Number of candidates evaluated so far, including invalid costs.
    pub evaluations: usize,
}

/// Decision returned by a progress callback.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    /// Continue optimization unless another stopping criterion is met.
    Continue,
    /// Stop and return the current best candidate with [`Termination::Cancelled`].
    Stop,
}

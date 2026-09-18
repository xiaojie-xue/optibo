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
    Dither { min: f64, max: f64 },
}

/// Initial population generation, in normalized free-variable coordinates.
#[derive(Clone, Debug, PartialEq)]
pub enum Initialization {
    Random,
    LatinHypercube,
    /// User-supplied rows in physical coordinates, including fixed variables.
    /// At least four rows are required; finite coordinates are clipped to bounds.
    /// The row count overrides [`Config::population_size`].
    Population(Vec<Vec<f64>>),
}

/// Validated before any objective evaluation. Defaults use seed 0, LHS, and
/// serial evaluation. No implicit local polishing or constraint penalties occur.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub strategy: Strategy,
    /// Actual count, not SciPy's per-free-dimension multiplier. At least four.
    pub population_size: usize,
    /// Number of complete evolutionary generations, excluding initialization.
    pub max_generations: usize,
    /// Candidate evaluation budget, including initialization and invalid costs.
    /// Must cover the initial population (or one candidate when all bounds fixed).
    /// A last partial generation is evaluated in population index order.
    pub max_evaluations: usize,
    pub mutation: Mutation,
    /// Binomial crossover probability in [0, 1]. One donor coordinate is forced.
    pub crossover: f64,
    /// Stop when population std <= atol + tol * abs(population mean).
    /// All population costs must be finite. Both tolerances must be finite >= 0.
    pub tol: f64,
    pub atol: f64,
    pub seed: u64,
    pub init: Initialization,
    /// Physical coordinates; replaces the first initial member. Must be in bounds.
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
    pub message: String,
}

impl EvaluationError {
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeError {
    InvalidConfig(String),
    InvalidBounds {
        index: usize,
        reason: String,
    },
    InvalidPopulation(String),
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
    Converged,
    MaxGenerations,
    MaxEvaluations,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeResult {
    pub x: Vec<f64>,
    pub fun: f64,
    /// Physical coordinates, including fixed variables, in population index order.
    pub population: Vec<Vec<f64>>,
    /// Invalid members are represented by positive infinity.
    pub population_energies: Vec<f64>,
    /// Complete generations only. A budget-limited partial generation is excluded.
    pub generations: usize,
    pub evaluations: usize,
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
    pub x: &'a [f64],
    pub fun: f64,
    pub generations: usize,
    pub evaluations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Stop,
}

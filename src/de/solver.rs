use super::domain::Domain;
use super::operators::{initialize_latin_hypercube, initialize_random, trial_population};
use super::rng::Rng;
use super::{
    BatchObjective, Config, Control, DeError, DeResult, EvaluationError, Initialization, Mutation,
    Progress, Termination,
};

/// Minimize a scalar objective. NaN and either infinity reject a candidate.
pub fn minimize<F>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: F,
) -> Result<DeResult, DeError>
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    minimize_with_callback(bounds, config, objective, |_| Control::Continue)
}

/// Scalar minimization with cancellation/progress at batch boundaries.
pub fn minimize_with_callback<F, C>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: F,
    callback: C,
) -> Result<DeResult, DeError>
where
    F: Fn(&[f64]) -> f64 + Sync,
    C: FnMut(&Progress<'_>) -> Control,
{
    minimize_fallible_with_callback(bounds, config, |x| Ok(objective(x)), callback)
}

/// Distinguishes an invalid candidate (nonfinite Ok value) from a fatal error.
pub fn minimize_fallible<F>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: F,
) -> Result<DeResult, DeError>
where
    F: Fn(&[f64]) -> Result<f64, EvaluationError> + Sync,
{
    minimize_fallible_with_callback(bounds, config, objective, |_| Control::Continue)
}

/// Fallible scalar evaluation with a progress/cancellation callback.
pub fn minimize_fallible_with_callback<F, C>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: F,
    callback: C,
) -> Result<DeResult, DeError>
where
    F: Fn(&[f64]) -> Result<f64, EvaluationError> + Sync,
    C: FnMut(&Progress<'_>) -> Control,
{
    if config.parallel && !cfg!(feature = "parallel") {
        return Err(DeError::InvalidConfig(
            "parallel scalar evaluation requires the `parallel` crate feature".into(),
        ));
    }
    let adapter = ScalarObjective {
        objective,
        parallel: config.parallel,
    };
    minimize_batch_with_callback(bounds, config, &adapter, callback)
}

struct ScalarObjective<F> {
    objective: F,
    parallel: bool,
}

impl<F> BatchObjective for ScalarObjective<F>
where
    F: Fn(&[f64]) -> Result<f64, EvaluationError> + Sync,
{
    fn evaluate_batch(
        &self,
        candidates: &[f64],
        dimension: usize,
        costs: &mut [f64],
    ) -> Result<(), EvaluationError> {
        #[cfg(feature = "parallel")]
        if self.parallel {
            use rayon::prelude::*;
            // Indexed collection preserves candidate order, including which error
            // is reported. Random generation never occurs on worker threads.
            let results: Vec<_> = candidates
                .par_chunks_exact(dimension)
                .map(&self.objective)
                .collect();
            for (output, result) in costs.iter_mut().zip(results) {
                *output = result?;
            }
            return Ok(());
        }
        #[cfg(not(feature = "parallel"))]
        debug_assert!(!self.parallel);
        for (output, x) in costs.iter_mut().zip(candidates.chunks_exact(dimension)) {
            *output = (self.objective)(x)?;
        }
        Ok(())
    }
}

/// Minimize using row-major, whole-batch evaluations. `config.parallel` is ignored:
/// a batch implementation may use its own vectorization, threading, or device.
pub fn minimize_batch<B: BatchObjective + ?Sized>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: &B,
) -> Result<DeResult, DeError> {
    minimize_batch_with_callback(bounds, config, objective, |_| Control::Continue)
}

/// Batch minimization with cancellation/progress at evaluation boundaries.
/// A callback stop takes precedence over convergence and budget termination.
pub fn minimize_batch_with_callback<B, C>(
    bounds: &[(f64, f64)],
    config: &Config,
    objective: &B,
    mut callback: C,
) -> Result<DeResult, DeError>
where
    B: BatchObjective + ?Sized,
    C: FnMut(&Progress<'_>) -> Control,
{
    let domain = Domain::new(bounds)?;
    validate(config)?;
    let size = match &config.init {
        Initialization::Population(rows) => rows.len(),
        _ => config.population_size,
    };
    if size < 4 {
        return Err(match config.init {
            Initialization::Population(_) => {
                DeError::InvalidPopulation("at least four rows are required".into())
            }
            _ => DeError::InvalidConfig("population_size must be at least four".into()),
        });
    }
    let all_fixed = domain.free_dimension() == 0;
    let initial_evaluations = if all_fixed { 1 } else { size };
    if config.max_evaluations < initial_evaluations {
        return Err(DeError::InvalidConfig(format!(
            "max_evaluations must cover initialization ({initial_evaluations} candidates)"
        )));
    }
    if size.checked_mul(domain.dimension()).is_none() {
        return Err(DeError::InvalidConfig(
            "population dimensions overflow usize".into(),
        ));
    }
    let guess = config
        .initial_guess
        .as_ref()
        .map(|x| domain.encode(x, false))
        .transpose()?;
    // Validate all user data even for a completely fixed optimization problem.
    let custom_population = match &config.init {
        Initialization::Population(rows) => Some(
            rows.iter()
                .map(|x| domain.encode(x, true))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        _ => None,
    };
    let mut rng = Rng::new(config.seed);
    let mut population = if all_fixed {
        vec![Vec::new()]
    } else if let Some(rows) = custom_population {
        rows
    } else {
        match config.init {
            Initialization::Random => initialize_random(size, domain.free_dimension(), &mut rng),
            _ => initialize_latin_hypercube(size, domain.free_dimension(), &mut rng),
        }
    };
    if let Some(guess) = guess {
        population[0] = guess;
    }
    let mut energies = evaluate(objective, &domain, &population)?;
    if !energies.iter().any(|cost| cost.is_finite()) {
        return Err(DeError::NoFiniteObjective);
    }
    let mut evaluations = energies.len();
    let mut generations = 0;
    let termination = loop {
        let best = best_index(&energies);
        let x = domain.decode(&population[best]);
        if callback(&Progress {
            x: &x,
            fun: energies[best],
            generations,
            evaluations,
        }) == Control::Stop
        {
            break Termination::Cancelled;
        }
        if all_fixed || converged(&energies, config.tol, config.atol) {
            break Termination::Converged;
        }
        if evaluations >= config.max_evaluations {
            break Termination::MaxEvaluations;
        }
        if generations >= config.max_generations {
            break Termination::MaxGenerations;
        }
        let mutation = match config.mutation {
            Mutation::Fixed(value) => value,
            Mutation::Dither { min, max } => dither(min, max, rng.uniform()),
        };
        let trials = trial_population(
            &population,
            best,
            config.strategy,
            mutation,
            config.crossover,
            &mut rng,
        );
        let count = population.len().min(config.max_evaluations - evaluations);
        let trial_energies = evaluate(objective, &domain, &trials[..count])?;
        evaluations += count;
        // All trial vectors came from the unchanged old generation. An accepted
        // improvement cannot influence another candidate until the next generation.
        for (i, (trial, energy)) in trials.into_iter().zip(trial_energies).enumerate() {
            if energy.is_finite() && energy <= energies[i] {
                population[i] = trial;
                energies[i] = energy;
            }
        }
        if count == population.len() {
            generations += 1;
        }
    };
    let best = best_index(&energies);
    Ok(DeResult {
        x: domain.decode(&population[best]),
        fun: energies[best],
        population: population.iter().map(|z| domain.decode(z)).collect(),
        population_energies: energies,
        generations,
        evaluations,
        termination,
    })
}

fn validate(config: &Config) -> Result<(), DeError> {
    let invalid = |message: &str| DeError::InvalidConfig(message.into());
    if config.max_evaluations == 0 {
        return Err(invalid("max_evaluations must be positive"));
    }
    if !config.crossover.is_finite() || !(0.0..=1.0).contains(&config.crossover) {
        return Err(invalid("crossover must be finite and in [0, 1]"));
    }
    if !config.tol.is_finite() || config.tol < 0.0 || !config.atol.is_finite() || config.atol < 0.0
    {
        return Err(invalid("tol and atol must be finite and nonnegative"));
    }
    let valid_weight = |x: f64| x.is_finite() && (0.0..2.0).contains(&x);
    match config.mutation {
        Mutation::Fixed(x) if !valid_weight(x) => {
            return Err(invalid("fixed mutation must be in [0, 2)"))
        }
        Mutation::Dither { min, max } if !valid_weight(min) || !valid_weight(max) || min >= max => {
            return Err(invalid("mutation dithering needs 0 <= min < max < 2"));
        }
        _ => {}
    }
    Ok(())
}

fn evaluate<B: BatchObjective + ?Sized>(
    objective: &B,
    domain: &Domain,
    population: &[Vec<f64>],
) -> Result<Vec<f64>, DeError> {
    let candidates: Vec<_> = population.iter().flat_map(|z| domain.decode(z)).collect();
    let mut costs = vec![f64::NAN; population.len()];
    objective.evaluate_batch(&candidates, domain.dimension(), &mut costs)?;
    for cost in &mut costs {
        if !cost.is_finite() {
            *cost = f64::INFINITY;
        }
    }
    Ok(costs)
}

fn best_index(energies: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..energies.len() {
        if energies[i] < energies[best] {
            best = i;
        }
    }
    best
}

fn dither(min: f64, max: f64, uniform: f64) -> f64 {
    // Adding an almost-one jitter can round to max. Keep the advertised
    // half-open interval even when min and max are adjacent floating values.
    (min + (max - min) * uniform).min(f64::from_bits(max.to_bits() - 1))
}

/// Compute the dispersion criterion in scaled units so that large, finite
/// energies neither overflow the mean/variance nor falsely signal convergence.
fn converged(energies: &[f64], tol: f64, atol: f64) -> bool {
    if energies.is_empty() || energies.iter().any(|x| !x.is_finite()) {
        return false;
    }
    if energies.iter().all(|x| *x == energies[0]) {
        return true;
    }
    let max_abs = energies.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    // A power-of-two scale avoids an extra rounding step in ordinary inputs
    // (e.g. std([1,3]) == 1 must meet atol == 1 exactly). Include subnormals.
    let bits = max_abs.to_bits();
    let exponent = bits & (0x7ff_u64 << 52);
    let scale = if exponent != 0 {
        f64::from_bits(exponent)
    } else {
        f64::from_bits(1_u64 << (63 - bits.leading_zeros()))
    };
    let n = energies.len() as f64;
    let mean = energies.iter().map(|x| x / scale).sum::<f64>() / n;
    let variance = energies
        .iter()
        .map(|x| (x / scale - mean).powi(2))
        .sum::<f64>()
        / n;
    variance.sqrt() <= atol / scale + tol * mean.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_uses_population_std_and_relative_absolute_tolerances() {
        assert!(!converged(&[1.0, 3.0], 0.0, 0.99));
        assert!(converged(&[1.0, 3.0], 0.0, 1.0));
        assert!(converged(&[1.0, 3.0], 0.5, 0.0));
        assert!(converged(&[-1.0, -3.0], 0.5, 0.0));
        assert!(!converged(&[-1.0, 1.0], 10.0, 0.0));
        assert!(converged(&[0.0; 4], 0.0, 0.0));
        assert!(converged(&[0.1; 100], 0.0, 0.0));
    }

    #[test]
    fn convergence_is_safe_for_extreme_energies() {
        assert!(!converged(&[-f64::MAX, f64::MAX], 0.01, 0.0));
        assert!(converged(&[f64::MAX; 4], 0.0, 0.0));
        let tiny = f64::from_bits(1);
        assert!(!converged(&[-tiny, tiny], 0.0, 0.0));
        assert!(converged(&[-tiny, tiny], 0.0, tiny));
        assert!(!converged(&[f64::INFINITY; 4], 1.0, 1.0));
        assert!(!converged(&[1.0, f64::NAN], 1.0, 1.0));
    }

    #[test]
    fn equal_best_values_keep_first_index() {
        assert_eq!(best_index(&[f64::INFINITY, 1.0, -2.0, -2.0]), 2);
    }

    #[test]
    fn dithering_keeps_its_upper_endpoint_open_after_rounding() {
        let almost_one = f64::from_bits(1.0_f64.to_bits() - 1);
        for (min, max) in [
            (0.0, f64::from_bits(1)),
            (0.5, 1.0),
            (1.0, f64::from_bits(1.0_f64.to_bits() + 1)),
        ] {
            for uniform in [0.0, 0.5, almost_one] {
                let f = dither(min, max, uniform);
                assert!(f >= min && f < max);
            }
        }
    }

    #[test]
    fn mutation_and_crossover_parameter_endpoints() {
        for f in [0.0, 0.5, f64::from_bits(2.0_f64.to_bits() - 1)] {
            assert!(validate(&Config {
                mutation: Mutation::Fixed(f),
                ..Config::default()
            })
            .is_ok());
        }
        for f in [-1.0, 2.0, f64::NAN, f64::INFINITY] {
            assert!(validate(&Config {
                mutation: Mutation::Fixed(f),
                ..Config::default()
            })
            .is_err());
        }
        for cr in [0.0, 1.0] {
            assert!(validate(&Config {
                crossover: cr,
                ..Config::default()
            })
            .is_ok());
        }
    }
}

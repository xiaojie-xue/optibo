#![cfg(feature = "de")]

use optibo::de::{
    minimize, minimize_batch, minimize_fallible, minimize_with_callback, BatchObjective, Config,
    Control, DeError, EvaluationError, Initialization, Mutation, Strategy, Termination,
};
use std::sync::atomic::{AtomicUsize, Ordering};

fn run_config() -> Config {
    Config {
        population_size: 8,
        max_generations: 5,
        max_evaluations: 1_000,
        seed: 47,
        tol: 0.0,
        atol: 0.0,
        parallel: false,
        ..Config::default()
    }
}

fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|value| value * value).sum()
}

#[test]
fn zero_generations_only_evaluates_initial_population() {
    let calls = AtomicUsize::new(0);
    let config = Config {
        max_generations: 0,
        ..run_config()
    };
    let result = minimize(&[(-2.0, 2.0); 3], &config, |x| {
        calls.fetch_add(1, Ordering::Relaxed);
        sphere(x)
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 8);
    assert_eq!(result.evaluations, 8);
    assert_eq!(result.generations, 0);
    assert!(matches!(result.termination, Termination::MaxGenerations));
    assert!(!result.success());
}

#[test]
fn generation_limit_counts_initial_evaluations_separately() {
    let config = run_config();
    let result = minimize(&[(-2.0, 2.0); 4], &config, sphere).unwrap();
    assert_eq!(result.generations, config.max_generations);
    assert_eq!(
        result.evaluations,
        config.population_size * (config.max_generations + 1)
    );
    assert!(matches!(result.termination, Termination::MaxGenerations));
}

#[test]
fn evaluation_budget_allows_a_partial_last_generation() {
    let calls = AtomicUsize::new(0);
    let config = Config {
        max_evaluations: 19,
        ..run_config()
    };
    let result = minimize(&[(-2.0, 2.0); 4], &config, |x| {
        calls.fetch_add(1, Ordering::Relaxed);
        sphere(x)
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 19);
    assert_eq!(result.evaluations, 19);
    assert_eq!(result.generations, 1);
    assert!(matches!(result.termination, Termination::MaxEvaluations));
    assert!(!result.success());
}

#[test]
fn budget_can_end_exactly_after_initialization() {
    let config = Config {
        max_evaluations: 8,
        ..run_config()
    };
    let result = minimize(&[(-2.0, 2.0); 3], &config, sphere).unwrap();
    assert_eq!(result.evaluations, 8);
    assert_eq!(result.generations, 0);
    assert!(matches!(result.termination, Termination::MaxEvaluations));
}

#[test]
fn callback_can_cancel_before_the_first_generation() {
    let mut callback_calls = 0;
    let result = minimize_with_callback(&[(-2.0, 2.0); 3], &run_config(), sphere, |progress| {
        callback_calls += 1;
        assert_eq!(progress.evaluations, 8);
        assert_eq!(progress.generations, 0);
        assert_eq!(sphere(progress.x), progress.fun);
        Control::Stop
    })
    .unwrap();
    assert_eq!(callback_calls, 1);
    assert_eq!(result.evaluations, 8);
    assert!(matches!(result.termination, Termination::Cancelled));
    assert!(!result.success());
}

#[test]
fn callback_observes_completed_and_partial_generations() {
    let config = Config {
        max_evaluations: 19,
        ..run_config()
    };
    let mut history = Vec::new();
    let result = minimize_with_callback(&[(-2.0, 2.0); 4], &config, sphere, |progress| {
        history.push((progress.generations, progress.evaluations, progress.fun));
        Control::Continue
    })
    .unwrap();
    assert_eq!(
        history.iter().map(|&(g, n, _)| (g, n)).collect::<Vec<_>>(),
        [(0, 8), (1, 16), (1, 19)]
    );
    assert!(history.windows(2).all(|pair| pair[1].2 <= pair[0].2));
    assert_eq!(history.last().unwrap().2, result.fun);
}

#[test]
fn callback_can_cancel_after_a_completed_generation() {
    let result = minimize_with_callback(&[(-2.0, 2.0); 3], &run_config(), sphere, |progress| {
        if progress.generations == 2 {
            Control::Stop
        } else {
            Control::Continue
        }
    })
    .unwrap();
    assert_eq!(result.generations, 2);
    assert_eq!(result.evaluations, 24);
    assert!(matches!(result.termination, Termination::Cancelled));
}

#[test]
fn latin_hypercube_initialization_occupies_each_stratum_once() {
    let config = Config {
        population_size: 12,
        max_generations: 0,
        init: Initialization::LatinHypercube,
        ..run_config()
    };
    let result = minimize(&[(0.0, 1.0); 4], &config, sphere).unwrap();
    for dimension in 0..4 {
        let mut bins = result
            .population
            .iter()
            .map(|point| (point[dimension] * 12.0).floor() as usize)
            .collect::<Vec<_>>();
        bins.sort_unstable();
        assert_eq!(bins, (0..12).collect::<Vec<_>>());
    }
}

#[test]
fn custom_population_is_clipped_and_its_actual_size_overrides_configuration() {
    let config = Config {
        population_size: 40,
        max_generations: 0,
        max_evaluations: 4,
        init: Initialization::Population(vec![
            vec![-5.0, 0.25],
            vec![0.25, 5.0],
            vec![0.75, -5.0],
            vec![5.0, 0.75],
        ]),
        ..run_config()
    };
    let result = minimize(&[(0.0, 1.0); 2], &config, sphere).unwrap();
    assert_eq!(result.population.len(), 4);
    assert_eq!(result.evaluations, 4);
    for point in [
        vec![0.0, 0.25],
        vec![0.25, 1.0],
        vec![0.75, 0.0],
        vec![1.0, 0.75],
    ] {
        assert!(result.population.contains(&point));
    }
}

#[test]
fn supplied_initial_guess_is_evaluated_and_can_be_the_best() {
    let config = Config {
        initial_guess: Some(vec![0.125, -0.375]),
        max_generations: 0,
        ..run_config()
    };
    let result = minimize(&[(-1.0, 1.0); 2], &config, |x| {
        (x[0] - 0.125).powi(2) + (x[1] + 0.375).powi(2)
    })
    .unwrap();
    assert_eq!(result.x, [0.125, -0.375]);
    assert_eq!(result.fun, 0.0);
    assert_eq!(result.evaluations, 8);
}

#[test]
fn fixed_coordinates_and_hard_bounds_hold_for_every_evaluation() {
    let bounds = [(-2.0, 2.0), (7.0, 7.0), (-1e-9, 1e-9), (-3.0, -3.0)];
    for strategy in [Strategy::Best1Bin, Strategy::Rand1Bin] {
        let config = Config {
            strategy,
            mutation: Mutation::Fixed(1.99),
            crossover: 1.0,
            max_generations: 25,
            ..run_config()
        };
        let result = minimize(&bounds, &config, |x| {
            for (&value, &(lower, upper)) in x.iter().zip(&bounds) {
                assert!(value >= lower && value <= upper);
            }
            assert_eq!(x[1], 7.0);
            assert_eq!(x[3], -3.0);
            x[0].powi(2) + (x[2] / 1e-9).powi(2)
        })
        .unwrap();
        for point in &result.population {
            assert_eq!(point[1], 7.0);
            assert_eq!(point[3], -3.0);
        }
    }
}

#[test]
fn identical_seed_reproduces_every_population_member_and_cost() {
    for init in [Initialization::Random, Initialization::LatinHypercube] {
        let config = Config {
            init,
            ..run_config()
        };
        let first = minimize(&[(-3.0, 3.0); 3], &config, sphere).unwrap();
        let second = minimize(&[(-3.0, 3.0); 3], &config, sphere).unwrap();
        assert_eq!(first.x, second.x);
        assert_eq!(first.fun, second.fun);
        assert_eq!(first.population, second.population);
        assert_eq!(first.population_energies, second.population_energies);
        assert_eq!(first.evaluations, second.evaluations);
        assert_eq!(first.generations, second.generations);
    }
}

#[test]
fn seed_changes_the_initial_population() {
    let first = minimize(
        &[(-3.0, 3.0); 3],
        &Config {
            seed: 10,
            max_generations: 0,
            ..run_config()
        },
        sphere,
    )
    .unwrap();
    let second = minimize(
        &[(-3.0, 3.0); 3],
        &Config {
            seed: 11,
            max_generations: 0,
            ..run_config()
        },
        sphere,
    )
    .unwrap();
    assert_ne!(first.population, second.population);
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_and_serial_evaluation_have_identical_deferred_updates() {
    let serial = minimize(&[(-3.0, 3.0); 5], &run_config(), sphere).unwrap();
    let parallel = minimize(
        &[(-3.0, 3.0); 5],
        &Config {
            parallel: true,
            ..run_config()
        },
        sphere,
    )
    .unwrap();
    assert_eq!(serial.x, parallel.x);
    assert_eq!(serial.fun, parallel.fun);
    assert_eq!(serial.population, parallel.population);
    assert_eq!(serial.population_energies, parallel.population_energies);
    assert_eq!(serial.evaluations, parallel.evaluations);
}

#[cfg(not(feature = "parallel"))]
#[test]
fn scalar_parallel_request_without_feature_is_rejected() {
    let result = minimize(
        &[(-1.0, 1.0); 2],
        &Config {
            parallel: true,
            ..run_config()
        },
        sphere,
    );
    assert!(matches!(result, Err(DeError::InvalidConfig(_))));
}

#[test]
fn constant_finite_objective_satisfies_energy_convergence() {
    let result = minimize(&[(-2.0, 2.0); 3], &run_config(), |_| 7.0).unwrap();
    assert!(matches!(result.termination, Termination::Converged));
    assert!(result.success());
    assert_eq!(result.fun, 7.0);
}

#[test]
fn all_nonfinite_initial_energies_fail_explicitly() {
    for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let result = minimize(&[(-1.0, 1.0); 2], &run_config(), |_| invalid);
        assert!(matches!(result, Err(DeError::NoFiniteObjective)));
    }
}

#[test]
fn mixed_finite_and_nonfinite_candidates_preserve_a_finite_best() {
    let config = Config {
        max_generations: 0,
        init: Initialization::Population(vec![vec![-1.0], vec![-0.5], vec![0.25], vec![1.0]]),
        ..run_config()
    };
    let result = minimize(&[(-1.0, 1.0)], &config, |x| {
        if x[0] < 0.0 {
            f64::NAN
        } else {
            x[0].powi(2)
        }
    })
    .unwrap();
    assert_eq!(result.x, [0.25]);
    assert_eq!(result.fun, 0.0625);
    assert_eq!(
        result
            .population_energies
            .iter()
            .filter(|value| value.is_infinite())
            .count(),
        2
    );
    assert!(!result
        .population_energies
        .iter()
        .any(|value| value.is_nan()));
    assert!(!result.success());
}

#[test]
fn fatal_objective_errors_are_not_treated_as_bad_candidates() {
    let result = minimize_fallible(&[(-1.0, 1.0); 2], &run_config(), |_| {
        Err(EvaluationError::new("invalid measurement dataset"))
    });
    assert!(matches!(result, Err(DeError::Evaluation(_))));
}

#[test]
fn result_best_and_population_costs_match_the_objective() {
    let result = minimize(&[(-3.0, 3.0); 4], &run_config(), sphere).unwrap();
    assert_eq!(result.population.len(), result.population_energies.len());
    for (point, &energy) in result.population.iter().zip(&result.population_energies) {
        assert_eq!(sphere(point), energy);
    }
    assert_eq!(sphere(&result.x), result.fun);
    assert_eq!(
        result.fun,
        result
            .population_energies
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    );
}

struct BatchSphere {
    evaluated: AtomicUsize,
    batches: AtomicUsize,
}

impl BatchObjective for BatchSphere {
    fn evaluate_batch(
        &self,
        candidates: &[f64],
        dimension: usize,
        costs: &mut [f64],
    ) -> Result<(), EvaluationError> {
        assert_eq!(dimension, 3);
        assert_eq!(candidates.len(), costs.len() * dimension);
        self.evaluated.fetch_add(costs.len(), Ordering::Relaxed);
        self.batches.fetch_add(1, Ordering::Relaxed);
        for (point, cost) in candidates.chunks_exact(dimension).zip(costs) {
            *cost = sphere(point);
        }
        Ok(())
    }
}

#[test]
fn batch_api_matches_scalar_api_including_a_partial_generation() {
    let config = Config {
        max_evaluations: 19,
        ..run_config()
    };
    let objective = BatchSphere {
        evaluated: AtomicUsize::new(0),
        batches: AtomicUsize::new(0),
    };
    let scalar = minimize(&[(-3.0, 3.0); 3], &config, sphere).unwrap();
    let batch = minimize_batch(&[(-3.0, 3.0); 3], &config, &objective).unwrap();
    assert_eq!(batch.x, scalar.x);
    assert_eq!(batch.population, scalar.population);
    assert_eq!(batch.population_energies, scalar.population_energies);
    assert_eq!(objective.evaluated.load(Ordering::Relaxed), 19);
    assert_eq!(objective.batches.load(Ordering::Relaxed), 3);
    assert_eq!(batch.evaluations, 19);
}

struct FailingBatch;

impl BatchObjective for FailingBatch {
    fn evaluate_batch(&self, _: &[f64], _: usize, _: &mut [f64]) -> Result<(), EvaluationError> {
        Err(EvaluationError::new("batch data mismatch"))
    }
}

#[test]
fn fatal_batch_errors_propagate() {
    let result = minimize_batch(&[(-1.0, 1.0); 2], &run_config(), &FailingBatch);
    assert!(matches!(result, Err(DeError::Evaluation(_))));
}

struct UnwrittenBatch;

impl BatchObjective for UnwrittenBatch {
    fn evaluate_batch(&self, _: &[f64], _: usize, _: &mut [f64]) -> Result<(), EvaluationError> {
        Ok(())
    }
}

#[test]
fn unwritten_batch_outputs_cannot_masquerade_as_zero_cost_solutions() {
    let result = minimize_batch(&[(-1.0, 1.0); 2], &run_config(), &UnwrittenBatch);
    assert!(matches!(result, Err(DeError::NoFiniteObjective)));
}

#[test]
fn one_best1_generation_uses_the_actual_best_member_with_zero_mutation() {
    // This has a closed-form next generation independent of the RNG: F=0
    // makes every donor equal to the best vector, and CR=1 copies every
    // coordinate. The best deliberately starts at the last population index.
    let config = Config {
        strategy: Strategy::Best1Bin,
        mutation: Mutation::Fixed(0.0),
        crossover: 1.0,
        max_generations: 1,
        init: Initialization::Population(vec![
            vec![4.0, 3.0],
            vec![3.0, 4.0],
            vec![2.0, 2.0],
            vec![1.0, 1.0],
        ]),
        ..run_config()
    };
    let result = minimize(&[(-8.0, 8.0); 2], &config, sphere).unwrap();
    assert_eq!(result.generations, 1);
    assert_eq!(result.evaluations, 8);
    assert_eq!(result.population, vec![vec![1.0, 1.0]; 4]);
    assert_eq!(result.population_energies, vec![2.0; 4]);
    assert_eq!(result.x, [1.0, 1.0]);
    assert!(matches!(result.termination, Termination::Converged));
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_failure_reports_the_first_failing_candidate_in_population_order() {
    let config = Config {
        parallel: true,
        init: Initialization::Population(vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]]),
        ..run_config()
    };
    // Repetition exercises different worker scheduling without asserting which
    // worker runs first: the externally reported error must be index-stable.
    for _ in 0..16 {
        let result = minimize_fallible(&[(0.0, 8.0)], &config, |x| {
            if x[0] >= 2.0 {
                Err(EvaluationError::new(format!("candidate {}", x[0])))
            } else {
                Ok(x[0])
            }
        });
        match result {
            Err(DeError::Evaluation(error)) => assert_eq!(error.message, "candidate 2"),
            other => panic!("expected deterministic objective failure, got {other:?}"),
        }
    }
}

#[test]
fn invalid_trials_never_replace_even_an_invalid_existing_member() {
    let initial_population = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let config = Config {
        init: Initialization::Population(initial_population.clone()),
        max_generations: 3,
        parallel: false,
        ..run_config()
    };
    let calls = AtomicUsize::new(0);
    let result = minimize(&[(0.0, 8.0)], &config, |x| {
        // Deliberately inject evaluation failures after initialization to test
        // selection bookkeeping, rather than model a deterministic objective.
        let evaluation_index = calls.fetch_add(1, Ordering::Relaxed);
        if evaluation_index < 4 && x[0] == 1.0 {
            1.0
        } else {
            f64::NAN
        }
    })
    .unwrap();
    assert_eq!(result.population, initial_population);
    assert_eq!(
        result.population_energies,
        [1.0, f64::INFINITY, f64::INFINITY, f64::INFINITY]
    );
    assert_eq!(result.x, [1.0]);
    assert_eq!(result.fun, 1.0);
    assert_eq!(result.evaluations, 16);
    assert_eq!(result.generations, 3);
    assert!(matches!(result.termination, Termination::MaxGenerations));
}

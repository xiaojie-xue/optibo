#![cfg(feature = "de")]

use optibo::de::{minimize, Config, DeError, Initialization, Mutation, Strategy, Termination};
use std::sync::atomic::{AtomicUsize, Ordering};

fn no_iteration_config() -> Config {
    Config {
        population_size: 8,
        max_generations: 0,
        seed: 23,
        ..Config::default()
    }
}

#[test]
fn rejects_empty_reversed_and_nonfinite_bounds_before_evaluation() {
    let calls = AtomicUsize::new(0);
    let cases = [
        vec![],
        vec![(1.0, -1.0)],
        vec![(f64::NAN, 1.0)],
        vec![(0.0, f64::NAN)],
        vec![(f64::NEG_INFINITY, 1.0)],
        vec![(0.0, f64::INFINITY)],
    ];
    for bounds in cases {
        let result = minimize(&bounds, &Config::default(), |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            0.0
        });
        assert!(result.is_err(), "accepted invalid bounds: {bounds:?}");
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn identifies_the_invalid_bound_index() {
    let result = minimize(&[(0.0, 1.0), (2.0, -2.0)], &Config::default(), |_| 0.0);
    assert!(matches!(
        result,
        Err(DeError::InvalidBounds { index: 1, .. })
    ));
}

#[test]
fn rejects_invalid_solver_options() {
    let mut cases = Vec::new();
    cases.push(Config {
        population_size: 3,
        ..Config::default()
    });
    cases.push(Config {
        max_evaluations: 0,
        ..Config::default()
    });
    for crossover in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        cases.push(Config {
            crossover,
            ..Config::default()
        });
    }
    for value in [-0.1, 2.0, f64::NAN, f64::INFINITY] {
        cases.push(Config {
            mutation: Mutation::Fixed(value),
            ..Config::default()
        });
    }
    for (min, max) in [
        (-0.1, 0.8),
        (0.8, 0.8),
        (1.0, 0.5),
        (0.5, 2.0),
        (f64::NAN, 0.8),
        (0.5, f64::NAN),
    ] {
        cases.push(Config {
            mutation: Mutation::Dither { min, max },
            ..Config::default()
        });
    }
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        cases.push(Config {
            tol: value,
            ..Config::default()
        });
        cases.push(Config {
            atol: value,
            ..Config::default()
        });
    }
    let calls = AtomicUsize::new(0);
    for (index, config) in cases.into_iter().enumerate() {
        let result = minimize(&[(-1.0, 1.0)], &config, |_| {
            calls.fetch_add(1, Ordering::Relaxed);
            0.0
        });
        assert!(
            matches!(result, Err(DeError::InvalidConfig(_))),
            "invalid option case {index} was not rejected as a configuration error"
        );
    }
    assert_eq!(calls.load(Ordering::Relaxed), 0);
}

#[test]
fn accepts_extreme_valid_mutation_and_crossover_options() {
    for mutation in [Mutation::Fixed(0.0), Mutation::Fixed(1.999)] {
        for crossover in [0.0, 1.0] {
            let config = Config {
                mutation,
                crossover,
                ..no_iteration_config()
            };
            assert!(minimize(&[(-1.0, 1.0)], &config, |x| x[0] * x[0]).is_ok());
        }
    }
}

#[test]
fn rejects_malformed_custom_populations() {
    let cases = [
        vec![],
        vec![vec![0.0, 0.0]; 3],
        vec![vec![0.0]; 4],
        vec![vec![0.0, 0.0, 0.0]; 4],
        vec![vec![0.0, f64::NAN]; 4],
        vec![vec![f64::INFINITY, 0.0]; 4],
        vec![vec![0.0, 0.0], vec![0.0], vec![0.0, 0.0], vec![0.0, 0.0]],
    ];
    for population in cases {
        let config = Config {
            init: Initialization::Population(population),
            ..no_iteration_config()
        };
        assert!(matches!(
            minimize(&[(-1.0, 1.0); 2], &config, |_| 0.0),
            Err(DeError::InvalidPopulation(_))
        ));
    }
}

#[test]
fn rejects_malformed_or_out_of_bounds_initial_guesses() {
    for guess in [vec![], vec![0.0], vec![0.0, f64::NAN], vec![2.0, 0.0]] {
        let config = Config {
            initial_guess: Some(guess),
            ..no_iteration_config()
        };
        assert!(minimize(&[(-1.0, 1.0); 2], &config, |_| 0.0).is_err());
    }
}

#[test]
fn rejects_budget_too_small_to_evaluate_initial_population() {
    let config = Config {
        population_size: 8,
        max_evaluations: 7,
        ..Config::default()
    };
    assert!(matches!(
        minimize(&[(-1.0, 1.0)], &config, |_| 0.0),
        Err(DeError::InvalidConfig(_))
    ));
}

#[test]
fn fixed_bounds_evaluate_the_only_possible_point_once() {
    let calls = AtomicUsize::new(0);
    let config = Config {
        max_evaluations: 1,
        ..Config::default()
    };
    let result = minimize(&[(2.0, 2.0), (-3.0, -3.0)], &config, |x| {
        calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(x, [2.0, -3.0]);
        x.iter().map(|value| value * value).sum()
    })
    .unwrap();
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(result.evaluations, 1);
    assert_eq!(result.generations, 0);
    assert_eq!(result.fun, 13.0);
    assert!(matches!(result.termination, Termination::Converged));
    assert!(result.success());
}

#[test]
fn all_fixed_nonfinite_objective_is_an_error() {
    let result = minimize(&[(2.0, 2.0)], &Config::default(), |_| f64::NAN);
    assert!(matches!(result, Err(DeError::NoFiniteObjective)));
}

#[test]
fn both_strategies_support_the_minimum_population() {
    for strategy in [Strategy::Best1Bin, Strategy::Rand1Bin] {
        let config = Config {
            strategy,
            population_size: 4,
            max_generations: 4,
            tol: 0.0,
            atol: 0.0,
            ..Config::default()
        };
        let result = minimize(&[(-1.0, 1.0); 2], &config, |x| x[0] * x[0] + x[1] * x[1]).unwrap();
        assert!(result.fun.is_finite());
        assert_eq!(result.population.len(), 4);
    }
}

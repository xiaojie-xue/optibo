#![cfg(feature = "de")]

//! Deterministic fixtures observed with SciPy 1.15.3, not stochastic benchmarks.
//!
//! Reproduce the reference JSON with `reference/generate_scipy_fixtures.py`.
//! These tests do not require Python. Both implementations use a supplied
//! population, `best1bin`, F=0, CR=1, deferred updates, and no polishing so that
//! their different random generators cannot affect these expected results.
//!
//! Intentional differences: SciPy swaps its best member to row zero; optibo::de
//! preserves population indices. Optibo DE checks convergence immediately after
//! initialization; SciPy's public `maxiter=0` result reports no convergence. The
//! threshold fixture therefore uses SciPy's convergence predicate directly.

use optibo::de::{minimize, Config, DeResult, Initialization, Mutation, Strategy, Termination};

fn reference_config(population: Vec<Vec<f64>>, max_generations: usize) -> Config {
    Config {
        strategy: Strategy::Best1Bin,
        mutation: Mutation::Fixed(0.0),
        crossover: 1.0,
        init: Initialization::Population(population),
        max_generations,
        tol: 0.0,
        atol: 0.0,
        seed: 42,
        ..Config::default()
    }
}

fn assert_generation_counts(result: &DeResult, generations: usize, evaluations: usize) {
    assert_eq!(result.generations, generations);
    assert_eq!(result.evaluations, evaluations);
}

#[test]
fn best1bin_zero_mutation_matches_scipy_after_one_deferred_generation() {
    let initial = [0.0, 0.25, 0.5, 0.75, 1.0]
        .into_iter()
        .map(|value| vec![value, value])
        .collect();
    let result = minimize(&[(0.0, 1.0); 2], &reference_config(initial, 1), |x| {
        x.iter().map(|value| (value - 0.25).powi(2)).sum()
    })
    .unwrap();

    assert_eq!(result.x, [0.25, 0.25]);
    assert_eq!(result.fun, 0.0);
    assert_eq!(result.population, vec![vec![0.25, 0.25]; 5]);
    assert_eq!(result.population_energies, [0.0; 5]);
    assert_generation_counts(&result, 1, 10);
    assert_eq!(result.termination, Termination::Converged);
    assert!(result.success());
}

#[test]
fn custom_initialization_matches_scipy_clipping_with_documented_row_order() {
    let initial = vec![
        vec![5.0, 5.0],
        vec![0.25, 5.0],
        vec![0.75, -5.0],
        vec![5.0, 0.75],
        vec![0.5, 0.5],
    ];
    let result = minimize(&[(0.0, 1.0); 2], &reference_config(initial, 0), |x| {
        x.iter().map(|value| value * value).sum()
    })
    .unwrap();

    // SciPy promotes initial member four to row zero and swaps old row zero
    // back to row four. Compare every coordinate and associated energy using
    // that explicit permutation, while preserving optibo::de's index contract.
    let scipy_population = [
        [0.5, 0.5],
        [0.25, 1.0],
        [0.75, 0.0],
        [1.0, 0.75],
        [1.0, 1.0],
    ];
    let scipy_energies = [0.5, 1.0625, 0.5625, 1.5625, 2.0];
    for (scipy_row, optibo_row) in [4, 1, 2, 3, 0].into_iter().enumerate() {
        assert_eq!(result.population[optibo_row], scipy_population[scipy_row]);
        assert_eq!(
            result.population_energies[optibo_row],
            scipy_energies[scipy_row]
        );
    }
    assert_eq!(result.x, [0.5, 0.5]);
    assert_eq!(result.fun, 0.5);
    assert_generation_counts(&result, 0, 5);
    assert_eq!(result.termination, Termination::MaxGenerations);
    assert!(!result.success());
}

#[test]
fn mixed_fixed_dimensions_match_scipy_and_keep_first_minimum_on_ties() {
    let initial = vec![
        vec![-1.0, 3.0, 0.0],
        vec![-0.5, 3.0, 0.5],
        vec![0.0, 3.0, 1.0],
        vec![0.5, 3.0, 1.5],
        vec![1.0, 3.0, 2.0],
    ];
    let result = minimize(
        &[(-1.0, 1.0), (3.0, 3.0), (0.0, 2.0)],
        &reference_config(initial, 1),
        |x| (x[0] - 0.25).powi(2) + (x[2] - 1.25).powi(2),
    )
    .unwrap();

    // Initial members two and three tie at 0.125; both choose member two.
    assert_eq!(result.x, [0.0, 3.0, 1.0]);
    assert_eq!(result.fun, 0.125);
    assert_eq!(result.population, vec![vec![0.0, 3.0, 1.0]; 5]);
    assert_eq!(result.population_energies, [0.125; 5]);
    assert_generation_counts(&result, 1, 10);
    assert_eq!(result.termination, Termination::Converged);
    assert!(result.success());
}

#[test]
fn convergence_boundary_matches_scipy_predicate_at_adjacent_float_thresholds() {
    // SciPy's mean([0,0,0,0,5]) = 1 and population std = 2. Its predicate
    // returns false/true/true at these adjacent absolute tolerances.
    let thresholds = [
        f64::from_bits(2.0_f64.to_bits() - 1),
        2.0,
        f64::from_bits(2.0_f64.to_bits() + 1),
    ];
    for (atol, expected) in thresholds.into_iter().zip([false, true, true]) {
        let config = Config {
            atol,
            ..reference_config(
                vec![vec![0.0], vec![0.0], vec![0.0], vec![0.0], vec![5.0]],
                0,
            )
        };
        let result = minimize(&[(0.0, 5.0)], &config, |x| x[0]).unwrap();
        assert_eq!(result.success(), expected, "atol={atol}");
        assert_eq!(result.population_energies, [0.0, 0.0, 0.0, 0.0, 5.0]);
        assert_generation_counts(&result, 0, 5);
        assert_eq!(
            result.termination,
            if expected {
                Termination::Converged
            } else {
                Termination::MaxGenerations
            }
        );
    }
}

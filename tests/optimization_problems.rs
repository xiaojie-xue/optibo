#![cfg(feature = "de")]

//! Deterministic end-to-end tests exercise search quality, not merely API wiring.
//! The seed and generous evaluation budgets make failures reproducible. These
//! examples are deliberately small; passing them is not a global-optimality proof.

use optibo::de::{minimize, Config, Mutation, Strategy};

fn optimization_config() -> Config {
    Config {
        population_size: 60,
        max_generations: 1_000,
        max_evaluations: 60_060,
        mutation: Mutation::Dither { min: 0.5, max: 1.0 },
        crossover: 0.9,
        tol: 1e-10,
        atol: 1e-16,
        seed: 42,
        parallel: false,
        ..Config::default()
    }
}

#[test]
fn both_strategies_recover_a_shifted_sphere_minimum() {
    let target = [0.3, -1.7, 2.4, -0.8];
    for strategy in [Strategy::Best1Bin, Strategy::Rand1Bin] {
        let config = Config {
            strategy,
            ..optimization_config()
        };
        let result = minimize(&[(-5.0, 5.0); 4], &config, |x| {
            x.iter()
                .zip(target)
                .map(|(&value, center)| (value - center).powi(2))
                .sum()
        })
        .unwrap();
        assert!(result.fun < 1e-12, "sphere cost: {}", result.fun);
        for (&estimated, truth) in result.x.iter().zip(target) {
            assert!((estimated - truth).abs() < 1e-6);
        }
    }
}

#[test]
fn normalized_search_handles_widely_different_physical_parameter_scales() {
    let bounds = [(-1e6, 1e6), (-1e-6, 1e-6), (-50.0, 50.0)];
    let target = [137_000.0, 2.8e-7, -12.0];
    let scale = [1e6, 1e-6, 50.0];
    let result = minimize(&bounds, &optimization_config(), |x| {
        (0..3)
            .map(|i| ((x[i] - target[i]) / scale[i]).powi(2))
            .sum()
    })
    .unwrap();
    assert!(result.fun < 1e-12, "scaled sphere cost: {}", result.fun);
    for i in 0..3 {
        assert!(((result.x[i] - target[i]) / scale[i]).abs() < 1e-6);
    }
}

#[test]
fn follows_the_curved_rosenbrock_valley() {
    let config = Config {
        max_generations: 1_500,
        max_evaluations: 90_060,
        ..optimization_config()
    };
    let result = minimize(&[(-2.0, 2.0); 3], &config, |x| {
        x.windows(2)
            .map(|pair| 100.0 * (pair[1] - pair[0].powi(2)).powi(2) + (1.0 - pair[0]).powi(2))
            .sum()
    })
    .unwrap();
    assert!(result.fun < 1e-10, "Rosenbrock cost: {}", result.fun);
    assert!(result.x.iter().all(|value| (value - 1.0).abs() < 1e-4));
}

#[test]
fn escapes_local_wells_on_a_two_dimensional_rastrigin_problem() {
    let config = Config {
        strategy: Strategy::Rand1Bin,
        population_size: 80,
        max_evaluations: 80_080,
        ..optimization_config()
    };
    let result = minimize(&[(-5.12, 5.12); 2], &config, |x| {
        20.0 + x
            .iter()
            .map(|value| value * value - 10.0 * (2.0 * std::f64::consts::PI * value).cos())
            .sum::<f64>()
    })
    .unwrap();
    assert!(result.fun < 1e-9, "Rastrigin cost: {}", result.fun);
    assert!(result.x.iter().all(|value| value.abs() < 1e-5));
}

#[test]
fn approaches_an_optimum_on_the_boundary() {
    let result = minimize(&[(0.0, 1.0), (-2.0, 2.0)], &optimization_config(), |x| {
        // The unconstrained optimum is outside the allowed box.
        (x[0] + 1.0).powi(2) + (x[1] - 0.4).powi(2)
    })
    .unwrap();
    assert!(result.x[0] < 1e-6);
    assert!((result.x[1] - 0.4).abs() < 1e-4);
    assert!((result.fun - 1.0).abs() < 1e-8);
}

fn planar_arm_position(q: [f64; 2], offsets: &[f64]) -> [f64; 2] {
    let shoulder = q[0] + offsets[0];
    let elbow = shoulder + q[1] + offsets[1];
    [
        0.7 * shoulder.cos() + 0.4 * elbow.cos(),
        0.7 * shoulder.sin() + 0.4 * elbow.sin(),
    ]
}

#[test]
fn recovers_identifiable_robot_joint_zero_offsets_and_predicts_held_out_poses() {
    let truth = [0.085, -0.12];
    let measurements = (0..24)
        .map(|index| {
            let phase = f64::from(index) * 0.37;
            let q = [1.2 * phase.sin(), 1.4 * (phase * 1.7 + 0.2).cos()];
            (q, planar_arm_position(q, &truth))
        })
        .collect::<Vec<_>>();
    let result = minimize(&[(-0.3, 0.3); 2], &optimization_config(), |offsets| {
        measurements
            .iter()
            .map(|&(q, measured)| {
                let predicted = planar_arm_position(q, offsets);
                (predicted[0] - measured[0]).powi(2) + (predicted[1] - measured[1]).powi(2)
            })
            .sum::<f64>()
            / measurements.len() as f64
    })
    .unwrap();

    assert!(result.fun < 1e-14, "training MSE: {}", result.fun);
    for (&estimated, actual) in result.x.iter().zip(truth) {
        assert!((estimated - actual).abs() < 1e-6);
    }
    // Validation samples were not used to construct the objective.
    for q in [[-1.1, 0.7], [0.15, -1.2], [1.7, 1.1], [-0.3, -0.4]] {
        let predicted = planar_arm_position(q, &result.x);
        let measured = planar_arm_position(q, &truth);
        let position_error = (predicted[0] - measured[0]).hypot(predicted[1] - measured[1]);
        assert!(
            position_error < 1e-6,
            "held-out position error: {position_error}"
        );
    }
}

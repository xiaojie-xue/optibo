//! Synthetic, noiseless two-link calibration. No URDF or sensor dependencies.
use optibo::de::{minimize, Config};

fn position(q: [f64; 2], offsets: &[f64]) -> [f64; 2] {
    let a = q[0] + offsets[0];
    let b = a + q[1] + offsets[1];
    [0.7 * a.cos() + 0.4 * b.cos(), 0.7 * a.sin() + 0.4 * b.sin()]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let truth = [0.085, -0.12];
    let measurements: Vec<_> = (0..24)
        .map(|i| {
            let t = f64::from(i) * 0.37;
            let q = [1.2 * t.sin(), 1.4 * (1.7 * t + 0.2).cos()];
            (q, position(q, &truth))
        })
        .collect();
    let config = Config {
        population_size: 60,
        max_generations: 1000,
        max_evaluations: 60_060,
        crossover: 0.9,
        seed: 42,
        tol: 1e-10,
        atol: 1e-16,
        ..Config::default()
    };
    let result = minimize(&[(-0.3, 0.3); 2], &config, |offsets| {
        measurements
            .iter()
            .map(|&(q, measured)| {
                let predicted = position(q, offsets);
                (predicted[0] - measured[0]).powi(2) + (predicted[1] - measured[1]).powi(2)
            })
            .sum::<f64>()
            / measurements.len() as f64
    })?;
    let held_out = [[-1.1, 0.7], [0.15, -1.2], [1.7, 1.1], [-0.3, -0.4]];
    let rms = (held_out
        .iter()
        .map(|&q| {
            let actual = position(q, &truth);
            let predicted = position(q, &result.x);
            (predicted[0] - actual[0]).powi(2) + (predicted[1] - actual[1]).powi(2)
        })
        .sum::<f64>()
        / held_out.len() as f64)
        .sqrt();
    println!("Synthetic noiseless planar arm; link lengths 0.7 m and 0.4 m");
    println!("True joint offsets (rad): {truth:?}");
    println!("Estimated offsets (rad): {:?}", result.x);
    println!("Training position MSE (m^2): {:.6e}", result.fun);
    println!("Held-out position RMS (m): {rms:.6e}");
    println!(
        "Termination: {:?}; generations: {}; evaluations: {}",
        result.termination, result.generations, result.evaluations
    );
    if !result.success() || rms > 1e-6 {
        return Err("synthetic calibration validation failed".into());
    }
    Ok(())
}

//! Differential-evolution operators in normalized free-parameter coordinates.

use super::{rng::Rng, Strategy};

pub(crate) fn initialize_random(size: usize, dim: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    (0..size)
        .map(|_| (0..dim).map(|_| rng.uniform()).collect())
        .collect()
}

pub(crate) fn initialize_latin_hypercube(size: usize, dim: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut population = vec![vec![0.0; dim]; size];
    for coordinate in 0..dim {
        let mut column: Vec<_> = (0..size)
            .map(|stratum| stratified_coordinate(stratum, size, rng.uniform()))
            .collect();
        rng.shuffle(&mut column);
        for (individual, value) in population.iter_mut().zip(column) {
            individual[coordinate] = value;
        }
    }
    population
}

fn stratified_coordinate(stratum: usize, size: usize, jitter: f64) -> f64 {
    let value = (stratum as f64 + jitter) / size as f64;
    let upper = (stratum + 1) as f64 / size as f64;
    // Adding jitter very close to one can round up to the next stratum,
    // including exactly 1.0 in the last stratum. Keep its upper edge open.
    value.min(f64::from_bits(upper.to_bits() - 1))
}

/// Build every candidate from the same immutable generation. The caller does
/// objective evaluation and selection only after this function has returned.
pub(crate) fn trial_population(
    population: &[Vec<f64>],
    best_index: usize,
    strategy: Strategy,
    mutation: f64,
    crossover: f64,
    rng: &mut Rng,
) -> Vec<Vec<f64>> {
    if population.is_empty() || population[0].is_empty() {
        return population.to_vec();
    }
    let dim = population[0].len();
    debug_assert!(best_index < population.len());
    debug_assert!(population.iter().all(|individual| individual.len() == dim));
    debug_assert!(mutation.is_finite());
    debug_assert!((0.0..=1.0).contains(&crossover));
    let count = match strategy {
        Strategy::Best1Bin => 2,
        Strategy::Rand1Bin => 3,
    };
    (0..population.len())
        .map(|target| {
            let donors = donor_indices(population.len(), target, count, rng);
            let mutant: Vec<_> = (0..dim)
                .map(|coordinate| {
                    mutant_coordinate(
                        population, best_index, strategy, mutation, &donors, coordinate,
                    )
                })
                .collect();
            binomial_crossover(&population[target], &mutant, crossover, rng)
        })
        .collect()
}

/// Draw an ordered sample without replacement, excluding the target individual.
fn donor_indices(size: usize, target: usize, count: usize, rng: &mut Rng) -> [usize; 3] {
    assert!(count <= 3 && count < size && target < size);
    let mut donors = [usize::MAX; 3];
    for chosen in 0..count {
        loop {
            let index = rng.index(size);
            if index != target && !donors[..chosen].contains(&index) {
                donors[chosen] = index;
                break;
            }
        }
    }
    donors
}

fn mutant_coordinate(
    population: &[Vec<f64>],
    best_index: usize,
    strategy: Strategy,
    mutation: f64,
    donors: &[usize; 3],
    coordinate: usize,
) -> f64 {
    let (base, left, right) = match strategy {
        Strategy::Best1Bin => (best_index, donors[0], donors[1]),
        Strategy::Rand1Bin => (donors[0], donors[1], donors[2]),
    };
    population[base][coordinate]
        + mutation * (population[left][coordinate] - population[right][coordinate])
}

fn binomial_crossover(target: &[f64], mutant: &[f64], crossover: f64, rng: &mut Rng) -> Vec<f64> {
    debug_assert_eq!(target.len(), mutant.len());
    let forced = rng.index(target.len());
    target
        .iter()
        .zip(mutant)
        .enumerate()
        .map(|(coordinate, (&old, &new))| {
            let value = if rng.uniform() < crossover || coordinate == forced {
                new
            } else {
                old
            };
            repair(value, rng)
        })
        .collect()
}

/// SciPy-style boundary handling: redraw only invalid coordinates, without
/// clipping or reflecting. Non-finite coordinates are invalid as well.
fn repair(value: f64, rng: &mut Rng) -> f64 {
    if (0.0..=1.0).contains(&value) {
        value
    } else {
        rng.uniform()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_initialization_has_correct_shape_bounds_and_reproducibility() {
        let first = initialize_random(17, 9, &mut Rng::new(91));
        assert_eq!(first, initialize_random(17, 9, &mut Rng::new(91)));
        assert_eq!(first.len(), 17);
        for individual in first {
            assert_eq!(individual.len(), 9);
            assert!(individual.iter().all(|value| (0.0..1.0).contains(value)));
        }
    }

    #[test]
    fn latin_hypercube_covers_each_stratum_once_in_every_dimension() {
        for size in [1, 2, 7, 32, 101] {
            let population = initialize_latin_hypercube(size, 6, &mut Rng::new(2026));
            assert_eq!(population.len(), size);
            for coordinate in 0..6 {
                let mut strata: Vec<_> = population
                    .iter()
                    .map(|individual| {
                        assert!((0.0..1.0).contains(&individual[coordinate]));
                        (individual[coordinate] * size as f64).floor() as usize
                    })
                    .collect();
                strata.sort_unstable();
                assert_eq!(strata, (0..size).collect::<Vec<_>>());
            }
        }
    }

    #[test]
    fn latin_hypercube_is_seeded_and_shuffles_dimensions_independently() {
        let first = initialize_latin_hypercube(64, 3, &mut Rng::new(42));
        let second = initialize_latin_hypercube(64, 3, &mut Rng::new(42));
        assert_eq!(first, second);
        let columns: Vec<Vec<_>> = (0..3)
            .map(|column| {
                first
                    .iter()
                    .map(|row| (row[column] * 64.0) as usize)
                    .collect()
            })
            .collect();
        assert_ne!(columns[0], columns[1]);
        assert_ne!(columns[1], columns[2]);
    }

    #[test]
    fn latin_hypercube_extreme_jitter_never_rounds_into_the_next_stratum() {
        let almost_one = f64::from_bits(1.0_f64.to_bits() - 1);
        for size in [1, 2, 3, 7, 32, 101] {
            for stratum in 0..size {
                for jitter in [0.0, 0.5, almost_one] {
                    let value = stratified_coordinate(stratum, size, jitter);
                    assert!(value >= stratum as f64 / size as f64);
                    assert!(value < (stratum + 1) as f64 / size as f64);
                    assert!((0.0..1.0).contains(&value));
                }
            }
        }
    }

    #[test]
    fn initializers_handle_empty_shapes() {
        assert_eq!(
            initialize_random(0, 3, &mut Rng::new(0)),
            Vec::<Vec<f64>>::new()
        );
        assert_eq!(
            initialize_latin_hypercube(0, 3, &mut Rng::new(0)),
            Vec::<Vec<f64>>::new()
        );
        assert_eq!(initialize_random(3, 0, &mut Rng::new(0)), vec![vec![]; 3]);
        assert_eq!(
            initialize_latin_hypercube(3, 0, &mut Rng::new(0)),
            vec![vec![]; 3]
        );
    }

    #[test]
    fn donors_are_distinct_and_exclude_target_at_minimum_and_larger_sizes() {
        let mut rng = Rng::new(12345);
        for count in [2, 3] {
            for size in [count + 1, 13, 100] {
                for target in 0..size {
                    for _ in 0..100 {
                        let donors = donor_indices(size, target, count, &mut rng);
                        for (position, donor) in donors[..count].iter().enumerate() {
                            assert!(*donor < size);
                            assert_ne!(*donor, target);
                            assert!(!donors[..position].contains(donor));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn minimum_population_uses_all_other_individuals() {
        let mut donors = donor_indices(4, 2, 3, &mut Rng::new(6));
        donors.sort_unstable();
        assert_eq!(donors, [0, 1, 3]);
    }

    #[test]
    fn mutation_equations_match_best1_and_rand1_definitions() {
        // Powers of two keep these expected equations exactly representable.
        let population = vec![
            vec![0.125, 0.25],
            vec![0.5, 0.75],
            vec![0.25, 0.0],
            vec![0.75, 0.5],
        ];
        let donors = [1, 2, 3];
        assert_eq!(
            mutant_coordinate(&population, 0, Strategy::Best1Bin, 0.5, &donors, 0),
            0.25
        );
        assert_eq!(
            mutant_coordinate(&population, 0, Strategy::Best1Bin, 0.5, &donors, 1),
            0.625
        );
        assert_eq!(
            mutant_coordinate(&population, 0, Strategy::Rand1Bin, 0.5, &donors, 0),
            0.25
        );
        assert_eq!(
            mutant_coordinate(&population, 0, Strategy::Rand1Bin, 0.5, &donors, 1),
            0.5
        );
    }

    #[test]
    fn crossover_zero_selects_exactly_one_mutant_coordinate() {
        let target = vec![0.25; 23];
        let mutant = vec![0.75; 23];
        for seed in 0..100 {
            let trial = binomial_crossover(&target, &mutant, 0.0, &mut Rng::new(seed));
            assert_eq!(trial.iter().filter(|value| **value == 0.75).count(), 1);
            assert_eq!(trial.iter().filter(|value| **value == 0.25).count(), 22);
        }
    }

    #[test]
    fn crossover_one_selects_every_mutant_coordinate() {
        let target = vec![0.25; 17];
        let mutant: Vec<_> = (0..17).map(|index| index as f64 / 32.0).collect();
        for seed in 0..10 {
            assert_eq!(
                binomial_crossover(&target, &mutant, 1.0, &mut Rng::new(seed)),
                mutant
            );
        }
    }

    #[test]
    fn a_single_dimension_is_always_taken_from_the_mutant() {
        for probability in [0.0, 0.3, 1.0] {
            assert_eq!(
                binomial_crossover(&[0.25], &[0.75], probability, &mut Rng::new(10)),
                vec![0.75]
            );
        }
    }

    #[test]
    fn repair_preserves_in_range_values_without_consuming_randomness() {
        let mut rng = Rng::new(87);
        for value in [0.0, 0.125, 0.5, 1.0] {
            assert_eq!(repair(value, &mut rng), value);
        }
        assert_eq!(rng.next_u64(), Rng::new(87).next_u64());
    }

    #[test]
    fn repair_redraws_invalid_values_instead_of_clipping() {
        let mut rng = Rng::new(87);
        let mut reference = rng.clone();
        for value in [-0.01, 1.01, f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            let repaired = repair(value, &mut rng);
            assert_eq!(repaired, reference.uniform());
            assert!(repaired > 0.0 && repaired < 1.0);
        }
    }

    #[test]
    fn trial_population_is_reproducible_bounded_and_does_not_change_parents() {
        let population = initialize_latin_hypercube(25, 12, &mut Rng::new(51));
        let original = population.clone();
        for strategy in [Strategy::Best1Bin, Strategy::Rand1Bin] {
            let first = trial_population(&population, 4, strategy, 1.9, 0.9, &mut Rng::new(8));
            let second = trial_population(&population, 4, strategy, 1.9, 0.9, &mut Rng::new(8));
            assert_eq!(first, second);
            assert_eq!(first.len(), population.len());
            assert_ne!(first, population);
            assert!(first.iter().all(|row| row.len() == 12));
            assert!(first
                .iter()
                .flatten()
                .all(|value| (0.0..=1.0).contains(value)));
            assert_eq!(population, original);
        }
    }

    #[test]
    fn best1_zero_mutation_and_full_crossover_copy_the_best_parent() {
        let population = initialize_random(8, 5, &mut Rng::new(21));
        let trials = trial_population(
            &population,
            3,
            Strategy::Best1Bin,
            0.0,
            1.0,
            &mut Rng::new(9),
        );
        assert_eq!(trials, vec![population[3].clone(); population.len()]);
    }

    #[test]
    fn all_fixed_trials_are_unchanged_and_do_not_consume_randomness() {
        let mut rng = Rng::new(5);
        let population = vec![vec![]; 4];
        assert_eq!(
            trial_population(&population, 0, Strategy::Rand1Bin, 0.5, 0.7, &mut rng),
            population
        );
        assert_eq!(rng.next_u64(), Rng::new(5).next_u64());
    }
}

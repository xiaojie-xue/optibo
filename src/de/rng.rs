//! Small reproducible random source for optimization, not cryptography.
//!
//! SplitMix64 has a completely specified transition, so seeds do not depend on
//! a platform RNG or an external crate version. `uniform` uses the top 53 bits
//! to produce exactly representable values in `[0, 1)`.

#[derive(Clone, Debug)]
pub(crate) struct Rng {
    state: u64,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }

    /// Uniform integer in `[0, upper)`, without modulo bias.
    pub(crate) fn index(&mut self, upper: usize) -> usize {
        assert!(upper > 0, "cannot sample from an empty range");
        let bound = upper as u64;
        // Discard the incomplete residue classes at the bottom of the range.
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }

    pub(crate) fn shuffle<T>(&mut self, values: &mut [T]) {
        for index in (1..values.len()).rev() {
            let other = self.index(index + 1);
            values.swap(index, other);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn splitmix64_matches_seed_zero_reference_vector() {
        let expected = [
            0xe220_a839_7b1d_cdaf,
            0x6e78_9e6a_a1b9_65f4,
            0x06c4_5d18_8009_454f,
            0xf88b_b8a8_724c_81ec,
            0x1b39_896a_51a8_749b,
            0x53cb_9f0c_747e_a2ea,
        ];
        let mut rng = Rng::new(0);
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    #[test]
    fn seeds_and_clone_reproduce_the_full_random_stream() {
        for seed in [0, 1, 42, u64::MAX] {
            let mut left = Rng::new(seed);
            let mut right = Rng::new(seed);
            for _ in 0..128 {
                assert_eq!(left.next_u64(), right.next_u64());
            }
            let mut cloned = left.clone();
            for _ in 0..128 {
                assert_eq!(left.next_u64(), cloned.next_u64());
            }
        }
        assert_ne!(Rng::new(0).next_u64(), Rng::new(1).next_u64());
    }

    #[test]
    fn uniform_values_have_the_documented_interval_and_bit_conversion() {
        let mut rng = Rng::new(1234);
        let mut raw = rng.clone();
        for _ in 0..100_000 {
            let value = rng.uniform();
            assert!((0.0..1.0).contains(&value));
            assert_eq!(
                value,
                (raw.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
            );
        }
    }

    #[test]
    fn index_handles_one_non_powers_of_two_and_large_bounds() {
        let mut rng = Rng::new(123);
        for upper in [1, 2, 3, 7, 256, 1_000_003, usize::MAX] {
            for _ in 0..1000 {
                assert!(rng.index(upper) < upper);
            }
        }
    }

    #[test]
    fn index_rejects_values_in_incomplete_residue_classes() {
        // This bound rejects nearly half of u64 values on a 64-bit target,
        // making the rejection branch deterministic rather than improbable.
        if usize::BITS != 64 {
            return;
        }
        let upper = (1_u64 << 63) + 1;
        let threshold = upper.wrapping_neg() % upper;
        let mut actual = Rng::new(7);
        let mut reference = actual.clone();
        let mut rejected = 0;
        for _ in 0..100 {
            let expected = loop {
                let value = reference.next_u64();
                if value >= threshold {
                    break value % upper;
                }
                rejected += 1;
            };
            assert_eq!(actual.index(upper as usize), expected as usize);
        }
        assert!(rejected > 0);
        assert_eq!(actual.next_u64(), reference.next_u64());
    }

    #[test]
    #[should_panic(expected = "cannot sample from an empty range")]
    fn index_rejects_empty_ranges() {
        Rng::new(0).index(0);
    }

    #[test]
    fn shuffle_is_reproducible_and_preserves_every_element() {
        let mut first: Vec<_> = (0..64).collect();
        let mut second = first.clone();
        Rng::new(99).shuffle(&mut first);
        Rng::new(99).shuffle(&mut second);
        assert_eq!(first, second);
        assert_ne!(first, (0..64).collect::<Vec<_>>());
        first.sort_unstable();
        assert_eq!(first, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn shuffling_empty_and_singleton_slices_does_not_consume_randomness() {
        let mut rng = Rng::new(3);
        let mut empty: [u8; 0] = [];
        rng.shuffle(&mut empty);
        let mut singleton = [17];
        rng.shuffle(&mut singleton);
        assert_eq!(singleton, [17]);
        assert_eq!(rng.next_u64(), Rng::new(3).next_u64());
    }
}

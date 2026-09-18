use super::DeError;

pub(crate) struct Domain {
    bounds: Vec<(f64, f64)>,
    free: Vec<usize>,
}

impl Domain {
    pub(crate) fn new(bounds: &[(f64, f64)]) -> Result<Self, DeError> {
        if bounds.is_empty() {
            return Err(DeError::InvalidBounds {
                index: 0,
                reason: "at least one parameter is required".into(),
            });
        }
        for (index, &(lo, hi)) in bounds.iter().enumerate() {
            if !lo.is_finite() || !hi.is_finite() || lo > hi {
                return Err(DeError::InvalidBounds {
                    index,
                    reason: "endpoints must be finite and lower <= upper".into(),
                });
            }
        }
        Ok(Self {
            bounds: bounds.to_vec(),
            free: bounds
                .iter()
                .enumerate()
                .filter_map(|(i, (lo, hi))| (lo < hi).then_some(i))
                .collect(),
        })
    }

    pub(crate) fn dimension(&self) -> usize {
        self.bounds.len()
    }

    pub(crate) fn free_dimension(&self) -> usize {
        self.free.len()
    }

    pub(crate) fn encode(&self, x: &[f64], clip: bool) -> Result<Vec<f64>, DeError> {
        if x.len() != self.dimension() {
            return Err(DeError::InvalidPopulation(format!(
                "expected {} coordinates, got {}",
                self.dimension(),
                x.len()
            )));
        }
        for (i, (&value, &(lo, hi))) in x.iter().zip(&self.bounds).enumerate() {
            if !value.is_finite() || (!clip && (value < lo || value > hi)) {
                return Err(DeError::InvalidPopulation(format!(
                    "coordinate {i} must be finite{}",
                    if clip { "" } else { " and within bounds" }
                )));
            }
        }
        Ok(self
            .free
            .iter()
            .map(|&i| {
                let (lo, hi) = self.bounds[i];
                let value = x[i].clamp(lo, hi);
                let width = hi - lo;
                let z = if width.is_finite() {
                    (value - lo) / width
                } else {
                    // Opposite, large finite bounds can have an overflowing width.
                    (value * 0.5 - lo * 0.5) / (hi * 0.5 - lo * 0.5)
                };
                z.clamp(0.0, 1.0)
            })
            .collect())
    }

    #[cfg(test)]
    pub(crate) fn decode(&self, z: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.dimension()];
        self.decode_into(z, &mut x);
        x
    }

    pub(crate) fn decode_into(&self, z: &[f64], x: &mut [f64]) {
        debug_assert_eq!(z.len(), self.free.len());
        debug_assert_eq!(x.len(), self.dimension());
        for (value, &(lo, _)) in x.iter_mut().zip(&self.bounds) {
            *value = lo;
        }
        for (&i, &value) in self.free.iter().zip(z) {
            let (lo, hi) = self.bounds[i];
            x[i] = if value <= 0.0 {
                lo
            } else if value >= 1.0 {
                hi
            } else if (hi - lo).is_finite() {
                (lo + value * (hi - lo)).clamp(lo, hi)
            } else {
                // Convex combination avoids forming the overflowing difference.
                ((1.0 - value) * lo + value * hi).clamp(lo, hi)
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physical_normalized_roundtrip_with_fixed_coordinates() {
        let domain = Domain::new(&[(-2.0, 8.0), (3.0, 3.0), (1e-8, 9e-8)]).unwrap();
        assert_eq!(domain.free_dimension(), 2);
        let x = [-1.0, 3.0, 5e-8];
        let z = domain.encode(&x, false).unwrap();
        assert_eq!(z[0], 0.1);
        let decoded = domain.decode(&z);
        for (a, b) in decoded.iter().zip(x) {
            assert!((a - b).abs() <= 1e-20);
        }
    }

    #[test]
    fn extreme_finite_bounds_do_not_overflow() {
        let d = Domain::new(&[(-f64::MAX, f64::MAX)]).unwrap();
        for z in [0.0, 0.1, 0.5, 0.9, 1.0] {
            let x = d.decode(&[z]);
            assert!(x[0].is_finite());
            assert!((d.encode(&x, false).unwrap()[0] - z).abs() < 1e-15);
        }
        assert_eq!(d.decode(&[0.0]), [-f64::MAX]);
        assert_eq!(d.decode(&[1.0]), [f64::MAX]);
    }

    #[test]
    fn subnormal_width_and_adjacent_bounds() {
        let tiny = f64::from_bits(1);
        let d = Domain::new(&[(tiny, 2.0 * tiny)]).unwrap();
        assert_eq!(d.encode(&[tiny], false).unwrap(), [0.0]);
        assert_eq!(d.encode(&[2.0 * tiny], false).unwrap(), [1.0]);
        let lo = 1e300_f64;
        let hi = f64::from_bits(lo.to_bits() + 1);
        let d = Domain::new(&[(lo, hi)]).unwrap();
        assert_eq!(d.decode(&[0.0]), [lo]);
        assert_eq!(d.decode(&[1.0]), [hi]);
    }

    #[test]
    fn clipping_and_strict_guess_validation() {
        let d = Domain::new(&[(0.0, 1.0), (2.0, 2.0)]).unwrap();
        assert_eq!(
            d.decode(&d.encode(&[-1.0, 99.0], true).unwrap()),
            [0.0, 2.0]
        );
        assert!(d.encode(&[-1.0, 2.0], false).is_err());
        assert!(d.encode(&[0.5, 1.0], false).is_err());
        assert!(d.encode(&[f64::NAN, 2.0], true).is_err());
        assert!(d.encode(&[0.5], true).is_err());
    }
}

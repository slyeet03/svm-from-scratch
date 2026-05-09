pub trait Kernel {
    fn compute(&self, xi: &[f64], xj: &[f64]) -> f64;
}

pub struct Linear;

impl Kernel for Linear {
    fn compute(&self, xi: &[f64], xj: &[f64]) -> f64 {
        let k: usize = xi.len();
        let mut sum: f64 = 0.0;

        for n in 0..k {
            sum += xi[n] * xj[n];
        }

        sum
    }
}

// implement sklearn way of calculating sigma wrt the given data later
// then tune it using C later
pub struct RBF {
    pub sigma: f64,
}

impl Kernel for RBF {
    fn compute(&self, xi: &[f64], xj: &[f64]) -> f64 {
        let k: usize = xi.len();
        let mut euc_dist: f64 = 0.0;
        let mut exp: f64 = 0.0;

        for n in 0..k {
            euc_dist += (xi[n] - xj[n]).powf(2.0);
        }

        exp = (-euc_dist) / (2.0 * self.sigma);
        exp.exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_kernel_compute() {
        let kernel = Linear;

        assert_eq!(
            kernel.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]),
            11.0_f64
        );
    }

    #[test]
    fn linear_kernel_symmetric() {
        let kernel = Linear;

        assert_eq!(
            kernel.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]),
            kernel.compute(&[4.0_f64, 1.0_f64], &[2.0_f64, 3.0_f64])
        );
    }

    #[test]
    fn linear_kernel_self_positive() {
        let kernel = Linear;

        assert!(kernel.compute(&[2.0_f64, 3.0_f64], &[2.0_f64, 3.0_f64]) > 0.0);
    }

    #[test]
    fn rbf_kernel_compute() {
        let kernel = RBF { sigma: 1.0 };

        assert_eq!(
            kernel.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]),
            0.01831563888873418_f64
        );
    }

    #[test]
    fn rbf_kernel_symmetric() {
        let kernel = RBF { sigma: 1.0 };

        assert_eq!(
            kernel.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]),
            kernel.compute(&[4.0_f64, 1.0_f64], &[2.0_f64, 3.0_f64])
        );
    }

    #[test]
    fn rbf_kernel_self_positive() {
        let kernel = RBF { sigma: 1.0 };

        assert!(kernel.compute(&[2.0_f64, 3.0_f64], &[2.0_f64, 3.0_f64]) > 0.0);
    }

    #[test]
    fn rbf_kernel_self_one() {
        let kernel = RBF { sigma: 1.0 };

        assert!(kernel.compute(&[2.0_f64, 3.0_f64], &[2.0_f64, 3.0_f64]) == 1.0);
    }
}

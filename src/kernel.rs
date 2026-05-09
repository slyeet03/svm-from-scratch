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

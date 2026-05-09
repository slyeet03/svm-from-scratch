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

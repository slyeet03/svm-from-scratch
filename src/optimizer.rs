use crate::kernel::Kernel;

pub struct SMO {
    all_vectors: Vec<Vec<f64>>,
    bias: f64,
    alpha: Vec<f64>, //for all vector alphas
    kernel: Box<dyn Kernel>,
    label: Vec<f64>, //for all vector labels, yi in prediction fn
    error_cache: Vec<f64>,
}

impl SMO {
    pub fn predict(&self, x: &[f64]) -> f64 {
        let i: usize = self.all_vectors.len();
        let mut fx: f64 = 0.0;
        let mut sum: f64 = 0.0;

        for n in 0..i {
            sum += self.alpha[n] * self.label[n] * self.kernel.compute(&self.all_vectors[n], x);
        }

        fx = sum + self.bias;

        fx
    }

    pub fn kkt_check(&self, alpha: f64, label: f64, fx: f64, C: f64, tol: f64) -> bool {
        let yfx: f64 = label * fx;

        if (alpha > 0.0) && (yfx > (1.0 + tol)) {
            return true;
        }
        if (alpha < C) && (yfx < (1.0 - tol)) {
            return true;
        }

        false
    }
}

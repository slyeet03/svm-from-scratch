use std::cmp;

#[derive(Clone, Copy)]
pub struct Hyperparameters {
    pub C: f64,
    pub sigma: f64,
    pub kkt_tol: f64,
    pub alpha_tol: f64,
    pub max_passes: usize,
}

impl Hyperparameters {
    pub fn default(n: usize) -> Self {
        Hyperparameters {
            C: 1.0,
            sigma: 1.0,
            kkt_tol: 0.001,
            alpha_tol: 1e-5,
            max_passes: cmp::max(100, n.isqrt()),
        }
    }
}

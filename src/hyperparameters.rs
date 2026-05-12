use std::cmp;

use crate::{grid_search::BestHyperParams, kernel::KernelType};

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

    pub fn grid_search_tuning(
        data: &[Vec<f64>],
        label: &[f64],
        c_values: &[f64],
        sigma_values: &[f64],
        k_folds: usize,
        kernel_type: KernelType,
    ) -> Self {
        let besthyperparam =
            BestHyperParams::grid_search(data, label, c_values, sigma_values, k_folds, kernel_type);

        Hyperparameters {
            C: besthyperparam.best_config.C,
            sigma: besthyperparam.best_config.sigma,
            kkt_tol: besthyperparam.best_config.kkt_tol,
            alpha_tol: besthyperparam.best_config.alpha_tol,
            max_passes: besthyperparam.best_config.max_passes,
        }
    }
}

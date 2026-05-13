use crate::{cross_val::cross_validate, hyperparameters::Hyperparameters, kernel::KernelType};

pub struct BestHyperParams {
    pub best_config: Hyperparameters,
    pub best_score: f64,
    pub all_scores: Vec<(f64, f64, f64)>,
}

impl BestHyperParams {
    pub fn grid_search(
        data: &[Vec<f64>],
        label: &[f64],
        c_values: &[f64],
        sigma_values: &[f64],
        k_folds: usize,
        kernel_type: KernelType,
    ) -> Self {
        let n: usize = data.len();
        let mut all_score: Vec<(f64, f64, f64)> = Vec::new();
        let mut best_score: f64 = 0.0;
        let mut hyperparameters = Hyperparameters::default(n);

        for C in c_values {
            for sigma in sigma_values {
                hyperparameters.sigma = *sigma;
                hyperparameters.C = *C;

                let score: f64 =
                    cross_validate(data, label, k_folds, &hyperparameters, &kernel_type);
                all_score.push((*C, *sigma, score));
            }
        }

        for (C, sigma, score) in all_score.clone() {
            if score > best_score {
                best_score = score;

                hyperparameters.sigma = sigma;
                hyperparameters.C = C;
            }
        }

        BestHyperParams {
            best_config: hyperparameters,
            best_score: best_score,
            all_scores: all_score,
        }
    }

    //implement random search later
}

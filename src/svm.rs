use crate::hyperparameters::{self, Hyperparameters};
use crate::kernel::Kernel;
use crate::optimizer::SMO;

pub struct SVM {
    pub support_vectors: Vec<Vec<f64>>,
    pub bias: f64,
    pub alpha: Vec<f64>, //support vector alphas only
    pub kernel: Box<dyn Kernel>,
    pub label: Vec<f64>, //support vector labels only, yi in prediction fn
}

impl SVM {
    pub fn new(kernel: Box<dyn Kernel>) -> Self {
        SVM {
            support_vectors: Vec::new(),
            bias: 0.0,
            label: Vec::new(),
            alpha: Vec::new(),
            kernel: kernel,
        }
    }

    pub fn predict(&self, x: &[f64]) -> i32 {
        let i: usize = self.support_vectors.len();
        let mut fx: f64 = 0.0;
        let mut sum: f64 = 0.0;
        let mut class: i32 = 0;

        for n in 0..i {
            sum += self.alpha[n] * self.label[n] * self.kernel.compute(&self.support_vectors[n], x);
        }

        fx = sum + self.bias;

        if fx >= 0.0 {
            class = 1;
        } else if fx < 0.0 {
            class = -1;
        }

        class
    }

    pub fn predict_raw(&self, x: &[f64]) -> f64 {
        let i: usize = self.support_vectors.len();
        let mut fx: f64 = 0.0;
        let mut sum: f64 = 0.0;
        let mut class: i32 = 0;

        for n in 0..i {
            sum += self.alpha[n] * self.label[n] * self.kernel.compute(&self.support_vectors[n], x);
        }

        fx = sum + self.bias;

        fx
    }

    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<f64>, hyperparameters: Hyperparameters) {
        let n: usize = data.len();

        let mut smo = SMO {
            all_vectors: data,
            label: labels,
            alpha: vec![0.0; n],
            bias: 0.0,
            error_cache: vec![0.0; n],
        };

        let kernel = self.kernel.as_ref();

        let (alphas, bias) = smo.train(
            hyperparameters.kkt_tol,
            hyperparameters.alpha_tol,
            hyperparameters.C,
            hyperparameters.max_passes,
            kernel,
        );

        for i in 0..n {
            if alphas[i] > 1e-5 {
                self.support_vectors.push(smo.all_vectors[i].clone());
                self.alpha.push(alphas[i]);
                self.label.push(smo.label[i]);
            }
        }

        self.bias = bias;
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel::{self, Linear};

    use super::*;

    #[test]
    fn svm_prediction_math_positive() {
        let svm = SVM {
            support_vectors: vec![vec![1.0, 0.0]],
            alpha: vec![1.0],
            label: vec![1.0],
            bias: 0.0,
            kernel: Box::new(Linear {}),
        };

        assert_eq!(svm.predict(&[1.0_f64, 0.0_f64]), 1);
    }

    #[test]
    fn svm_prediction_math_negative() {
        let svm = SVM {
            support_vectors: vec![vec![1.0, 0.0]],
            alpha: vec![1.0],
            label: vec![-1.0],
            bias: 0.0,
            kernel: Box::new(Linear {}),
        };

        assert_eq!(svm.predict(&[1.0_f64, 0.0_f64]), -1);
    }

    #[test]
    fn svm_fit_sv() {
        let data: Vec<Vec<f64>> = vec![
            vec![2.0, 2.0],
            vec![1.0, 1.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];
        let labels: Vec<f64> = vec![1.0, 1.0, -1.0, -1.0];
        let n: usize = data.len();
        let hyperparameters = Hyperparameters::default(n);

        let mut svm = SVM::new(Box::new(Linear));

        svm.fit(data, labels, hyperparameters);

        assert!(svm.support_vectors.len() > 0);
        assert!(svm.support_vectors.len() <= n);
        assert_eq!(svm.support_vectors.len(), svm.alpha.len());
        assert_eq!(svm.support_vectors.len(), svm.label.len());
    }

    #[test]
    fn svm_fit_valid_values() {
        let data: Vec<Vec<f64>> = vec![
            vec![2.0, 2.0],
            vec![1.0, 1.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];
        let labels: Vec<f64> = vec![1.0, 1.0, -1.0, -1.0];
        let n: usize = data.len();
        let hyperparameters = Hyperparameters::default(n);

        let mut svm = SVM::new(Box::new(Linear));

        svm.fit(data, labels, hyperparameters);

        assert!(
            svm.alpha
                .iter()
                .all(|&a| a > 1e-5 && a <= hyperparameters.C)
        );
        assert!(svm.label.iter().all(|&a| a == 1.0 || a == -1.0));
    }

    #[test]
    fn svm_fit_correct_predictions() {
        let data: Vec<Vec<f64>> = vec![
            vec![2.0, 2.0],
            vec![1.0, 1.0],
            vec![-1.0, -1.0],
            vec![-2.0, -2.0],
        ];
        let labels: Vec<f64> = vec![1.0, 1.0, -1.0, -1.0];
        let n: usize = data.len();
        let hyperparameters = Hyperparameters::default(n);

        let mut svm = SVM::new(Box::new(Linear));

        svm.fit(data, labels, hyperparameters);

        assert_eq!(svm.predict(&[2.0, 2.0]), 1);
        assert_eq!(svm.predict(&[1.0, 1.0]), 1);
        assert_eq!(svm.predict(&[-1.0, -1.0]), -1);
        assert_eq!(svm.predict(&[-2.0, -2.0]), -1);
        assert_eq!(svm.predict(&[3.0, 3.0]), 1);
        assert_eq!(svm.predict(&[-3.0, -3.0]), -1);
    }
}

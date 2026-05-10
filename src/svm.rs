use crate::kernel::Kernel;

pub struct SVM {
    support_vectors: Vec<Vec<f64>>,
    bias: f64,
    alpha: Vec<f64>, //support vector alphas only
    kernel: Box<dyn Kernel>,
    label: Vec<f64>, //support vector labels only, yi in prediction fn
}

impl SVM {
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
}

#[cfg(test)]
mod tests {
    use crate::kernel::Linear;

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
}

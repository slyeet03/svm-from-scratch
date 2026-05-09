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
            sum +=
                self.alpha[n] * self.label[n] * self.kernel.compute(&self.support_vectors[n], &x);
        }

        fx = sum + self.bias;

        if fx > 0.0 {
            class = 1;
        }
        if fx < 0.0 {
            class = -1;
        }

        class
    }
}

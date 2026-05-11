use crate::{data, svm::SVM};

pub struct Metrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub F1: f64,
    pub TP: f64,
    pub TN: f64,
    pub FP: f64,
    pub FN: f64,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            F1: 0.0,
            TP: 0.0,
            TN: 0.0,
            FP: 0.0,
            FN: 0.0,
        }
    }

    pub fn compute_accuracy(&mut self) {
        self.accuracy = (self.TP + self.TN) / (self.TN + self.TP + self.FN + self.FP)
    }

    pub fn compute_precision(&mut self) {
        self.precision = self.TP / (self.TP + self.FP);
    }

    pub fn compute_recall(&mut self) {
        self.recall = self.TP / (self.TP + self.FP);
    }

    pub fn compute_F1(&mut self) {
        self.F1 = (2.0 * self.precision * self.recall) / (self.precision + self.recall);
    }

    pub fn compute_metric(model: &SVM, test_set: (Vec<Vec<f64>>, Vec<f64>)) -> Self {
        let mut metrics = Metrics::new();

        for i in 0..test_set.0.len() {
            let class = model.predict(&test_set.0[i]);
            let label = test_set.1[i];

            if class == 1 {
                if label == 1.0 {
                    metrics.TP += 1.0;
                } else if label == -1.0 {
                    metrics.FP += 1.0;
                }
            } else if class == -1 {
                if label == 1.0 {
                    metrics.FN += 1.0;
                } else if label == -1.0 {
                    metrics.TN += 1.0;
                }
            }
        }

        metrics.compute_accuracy();
        metrics.compute_precision();
        metrics.compute_recall();
        metrics.compute_F1();

        metrics
    }
}

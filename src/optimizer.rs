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

    pub fn kkt_check(alpha: f64, label: f64, error: f64, C: f64, tol: f64) -> bool {
        let yE: f64 = label * error;

        if (alpha > 0.0) && (yE > tol) {
            return true;
        }
        if (alpha < C) && (yE < (-tol)) {
            return true;
        }

        false
    }

    pub fn compute_error(fx: f64, label: f64) -> f64 {
        let error: f64 = fx - label;

        error
    }

    pub fn compute_eta(&self, i: usize, j: usize) -> f64 {
        let eta: f64 = self
            .kernel
            .compute(&self.all_vectors[i], &self.all_vectors[i])
            + self
                .kernel
                .compute(&self.all_vectors[j], &self.all_vectors[j])
            - (2.0
                * self
                    .kernel
                    .compute(&self.all_vectors[i], &self.all_vectors[j]));

        eta
    }

    // updating both alpha and bias in one function because we need old and new values of alpha for bias
    pub fn update_alpha_bias(&mut self, i: usize, j: usize, eta: f64, C: f64, tol: f64) {
        let alphaj_old: f64 = self.alpha[j];
        let alphai_old: f64 = self.alpha[i];
        let mut alphai_new: f64 = self.alpha[i];

        // alpha_j updation
        let mut alphaj_new: f64 =
            alphaj_old + ((self.label[j] * (self.error_cache[i] - self.error_cache[j])) / eta);

        alphaj_new = self.clip_alphaj(i, j, alphaj_new, alphai_old, C);

        self.alpha[j] = alphaj_new;

        //alpha_i updation
        if ((alphaj_new - alphaj_old).abs()) >= tol {
            let s: f64 = self.label[i] * self.label[j];
            alphai_new = alphai_old + (s * (alphaj_old - alphaj_new));

            self.alpha[i] = alphai_new;
        }

        //bias updation
        let b1: f64 = self.bias
            - self.error_cache[i]
            - (self.label[i]
                * (alphai_new - alphai_old)
                * self
                    .kernel
                    .compute(&self.all_vectors[i], &self.all_vectors[i]))
            - (self.label[j]
                * (alphaj_new - alphaj_old)
                * self
                    .kernel
                    .compute(&self.all_vectors[i], &self.all_vectors[j]));

        let b2: f64 = self.bias
            - self.error_cache[j]
            - (self.label[i]
                * (alphai_new - alphai_old)
                * self
                    .kernel
                    .compute(&self.all_vectors[i], &self.all_vectors[j]))
            - (self.label[j]
                * (alphaj_new - alphaj_old)
                * self
                    .kernel
                    .compute(&self.all_vectors[j], &self.all_vectors[j]));

        if (alphai_new > 0.0) && (alphai_new < C) {
            self.bias = b1;
        } else if (alphaj_new > 0.0) && (alphaj_new < C) {
            self.bias = b2;
        } else {
            self.bias = (b1 + b2) / 2.0;
        }
    }

    pub fn clip_alphaj(&self, i: usize, j: usize, alphaj: f64, alphai: f64, C: f64) -> f64 {
        let (L, H): (f64, f64) = self.compute_LH(i, j, alphaj, alphai, C);
        let mut alphaj_new: f64 = alphaj;

        if alphaj > H {
            alphaj_new = H;
        } else if alphaj < L {
            alphaj_new = L
        }

        alphaj_new
    }

    pub fn compute_LH(&self, i: usize, j: usize, alphaj: f64, alphai: f64, C: f64) -> (f64, f64) {
        let mut L: f64 = 0.0;
        let mut H: f64 = 0.0;

        if self.label[i] != self.label[j] {
            L = f64::max(0.0, alphaj - alphai);
            H = f64::min(C, C + alphaj - alphai);
        } else if self.label[i] == self.label[j] {
            L = f64::max(0.0, alphai + alphaj - C);
            H = f64::min(C, alphai + alphaj);
        }

        (L, H)
    }
}

use crate::kernel::Kernel;
use rand::RngExt;

pub struct SMO {
    pub all_vectors: Vec<Vec<f64>>,
    pub bias: f64,
    pub alpha: Vec<f64>, //for all vector alphas
    pub label: Vec<f64>, //for all vector labels, yi in prediction fn
    pub error_cache: Vec<f64>,
}

impl SMO {
    pub fn predict(&self, x: &[f64], kernel: &dyn Kernel) -> f64 {
        let i: usize = self.all_vectors.len();
        let mut fx: f64 = 0.0;
        let mut sum: f64 = 0.0;

        for n in 0..i {
            sum += self.alpha[n] * self.label[n] * kernel.compute(&self.all_vectors[n], x);
        }

        fx = sum + self.bias;

        fx
    }

    pub fn kkt_check(&self, alpha: f64, label: f64, error: f64, C: f64, tol: f64) -> bool {
        let yE: f64 = label * error;

        if (alpha > 0.0) && (yE > tol) {
            return true;
        }
        if (alpha < C) && (yE < (-tol)) {
            return true;
        }

        false
    }

    pub fn compute_error(&self, fx: f64, label: f64) -> f64 {
        let error: f64 = fx - label;

        error
    }

    pub fn compute_eta(&self, i: usize, j: usize, kernel: &dyn Kernel) -> f64 {
        let eta: f64 = kernel.compute(&self.all_vectors[i], &self.all_vectors[i])
            + kernel.compute(&self.all_vectors[j], &self.all_vectors[j])
            - (2.0 * kernel.compute(&self.all_vectors[i], &self.all_vectors[j]));

        eta
    }

    // updating both alpha and bias in one function because we need old and new values of alpha for bias
    pub fn update_alpha_bias(
        &mut self,
        i: usize,
        j: usize,
        eta: f64,
        C: f64,
        tol: f64,
        kernel: &dyn Kernel,
    ) -> bool {
        let alphaj_old: f64 = self.alpha[j];
        let alphai_old: f64 = self.alpha[i];
        let mut alphai_new: f64 = self.alpha[i];
        let mut alpha_changed: bool = false;

        // alpha_j updation
        let mut alphaj_new: f64 =
            alphaj_old + ((self.label[j] * (self.error_cache[i] - self.error_cache[j])) / eta);

        alphaj_new = self.clip_alphaj(i, j, alphaj_new, alphai_old, C);

        //alpha_i updation
        if ((alphaj_new - alphaj_old).abs()) >= tol {
            let s: f64 = self.label[i] * self.label[j];
            alphai_new = alphai_old + (s * (alphaj_old - alphaj_new));
            alphai_new = alphai_new.max(0.0).min(C);

            self.alpha[j] = alphaj_new;
            self.alpha[i] = alphai_new;

            //bias updation
            let b1: f64 = self.bias
                - self.error_cache[i]
                - (self.label[i]
                    * (alphai_new - alphai_old)
                    * kernel.compute(&self.all_vectors[i], &self.all_vectors[i]))
                - (self.label[j]
                    * (alphaj_new - alphaj_old)
                    * kernel.compute(&self.all_vectors[i], &self.all_vectors[j]));

            let b2: f64 = self.bias
                - self.error_cache[j]
                - (self.label[i]
                    * (alphai_new - alphai_old)
                    * kernel.compute(&self.all_vectors[i], &self.all_vectors[j]))
                - (self.label[j]
                    * (alphaj_new - alphaj_old)
                    * kernel.compute(&self.all_vectors[j], &self.all_vectors[j]));

            if (alphai_new > 0.0) && (alphai_new < C) {
                self.bias = b1;
            } else if (alphaj_new > 0.0) && (alphaj_new < C) {
                self.bias = b2;
            } else {
                self.bias = (b1 + b2) / 2.0;
            }

            alpha_changed = true;
        }

        alpha_changed
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

    pub fn pick_j(&self, i: usize) -> usize {
        let mut j: usize = 0;
        let k: usize = self.all_vectors.len();

        // implementing argminmax myself cuz the crate for it is in fckin nightly build
        let mut current_best: usize = 0;

        if self.error_cache[i] > 0.0 {
            for n in 0..k {
                if self.error_cache[n] < self.error_cache[current_best] {
                    current_best = n;
                }
            }

            j = current_best;
        } else if self.error_cache[i] <= 0.0 {
            for n in 0..k {
                if self.error_cache[n] > self.error_cache[current_best] {
                    current_best = n;
                }
            }

            j = current_best;
        }

        // if j==i then we just pick a random number that is not equal to i
        while (j == i) {
            j = rand::rng().random_range(0..k);
        }

        j
    }

    pub fn train(
        &mut self,
        kkt_tol: f64,
        alpha_tol: f64,
        C: f64,
        max_passes: usize,
        kernel: &dyn Kernel,
    ) -> (Vec<f64>, f64) {
        let k: usize = self.all_vectors.len();
        for n in 0..k {
            self.alpha[n] = 0.0;
            self.error_cache[n] = -self.label[n];
        }
        self.bias = 0.0;
        let mut passes: usize = 0;

        while passes < max_passes {
            let mut num_changed_alphas: usize = 0;
            for i in 0..k {
                let xi = self.all_vectors[i].clone(); //borrow checker complains otherwise
                let fxi: f64 = self.predict(&xi, kernel);
                self.error_cache[i] = self.compute_error(fxi, self.label[i]);

                if self.kkt_check(
                    self.alpha[i],
                    self.label[i],
                    self.error_cache[i],
                    C,
                    kkt_tol,
                ) {
                    let j: usize = self.pick_j(i);
                    let xj = self.all_vectors[j].clone();
                    let fxj: f64 = self.predict(&xj, kernel);
                    self.error_cache[j] = self.compute_error(fxj, self.label[j]);

                    let eta: f64 = self.compute_eta(i, j, kernel);
                    if eta > 0.0 {
                        let alpha_changed = self.update_alpha_bias(i, j, eta, C, alpha_tol, kernel);

                        let xi = self.all_vectors[i].clone();
                        let fxi: f64 = self.predict(&xi, kernel);
                        self.error_cache[i] = self.compute_error(fxi, self.label[i]);

                        let xj = self.all_vectors[j].clone();
                        let fxj: f64 = self.predict(&xj, kernel);
                        self.error_cache[j] = self.compute_error(fxj, self.label[j]);

                        if alpha_changed {
                            num_changed_alphas += 1;
                        }
                    }
                }
            }
            if num_changed_alphas == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }
        (self.alpha.clone(), self.bias)
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel::Linear;

    use super::*;

    #[test]
    fn smo_compute_error() {
        let smo = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            label: vec![1.0, -1.0],
            alpha: vec![0.0, 0.0],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };

        assert_eq!(smo.compute_error(2.0, 1.0), 1.0);
        assert_eq!(smo.compute_error(1.0, 1.0), 0.0);
        assert_eq!(smo.compute_error(0.5, -1.0), 1.5);
    }

    #[test]
    fn smo_compute_eta() {
        let smo1 = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            label: vec![1.0, -1.0],
            alpha: vec![0.0, 0.0],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };
        let smo2 = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            label: vec![1.0, -1.0],
            alpha: vec![0.0, 0.0],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };
        let smo3 = SMO {
            all_vectors: vec![vec![2.0, 1.0], vec![1.0, 3.0]],
            label: vec![1.0, -1.0],
            alpha: vec![0.0, 0.0],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };

        let kernel = Linear;

        assert_eq!(smo1.compute_eta(0, 1, &kernel), 0.0);
        assert_eq!(smo2.compute_eta(0, 1, &kernel), 2.0);
        assert_eq!(smo3.compute_eta(0, 1, &kernel), 5.0);
    }

    #[test]
    fn smo_compute_LH() {
        let smo1 = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            label: vec![1.0, -1.0],
            alpha: vec![0.3, 0.7],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };
        let smo2 = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            label: vec![1.0, 1.0],
            alpha: vec![0.3, 0.4],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };

        let (L1, H1) = smo1.compute_LH(0, 1, 0.7, 0.3, 1.0);
        assert!((L1 - 0.4).abs() < 1e-10);
        assert!((H1 - 1.0).abs() < 1e-10);

        let (L2, H2) = smo2.compute_LH(0, 1, 0.4, 0.3, 1.0);
        assert!((L2 - 0.0).abs() < 1e-10);
        assert!((H2 - 0.7).abs() < 1e-10);

        let (L, H): (f64, f64) = smo1.compute_LH(0, 1, 0.7, 0.3, 1.0);
        assert!(L <= H);
        assert!(L >= 0.0);
        assert!(H <= 1.0);
    }

    #[test]
    fn smo_clipalphaj() {
        let smo = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            label: vec![1.0, 1.0],
            alpha: vec![0.3, 0.4],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };

        let result = smo.clip_alphaj(0, 1, 0.9, 0.3, 1.0);
        let (L, H) = smo.compute_LH(0, 1, 0.9, 0.3, 1.0);
        assert!(result <= H);
        assert!(result >= L);

        let result2 = smo.clip_alphaj(0, 1, -0.1, 0.3, 1.0);
        assert!(result2 >= 0.0);

        let result3 = smo.clip_alphaj(0, 1, 0.5, 0.3, 1.0);
        let (L3, H3) = smo.compute_LH(0, 1, 0.5, 0.3, 1.0);
        assert!(result3 >= L3 && result3 <= H3);
    }

    #[test]
    fn smo_kkt_check() {
        let smo = SMO {
            all_vectors: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            label: vec![1.0, 1.0],
            alpha: vec![0.3, 0.4],
            bias: 0.0,
            error_cache: vec![0.0, 0.0],
        };

        assert!(!smo.kkt_check(0.5, 1.0, 0.0, 1.0, 0.001));
        assert!(smo.kkt_check(0.5, 1.0, 0.5, 1.0, 0.001));
        assert!(smo.kkt_check(0.5, 1.0, -0.5, 1.0, 0.001));
        assert!(!smo.kkt_check(0.0, 1.0, 0.5, 1.0, 0.001));
        assert!(!smo.kkt_check(1.0, 1.0, -0.5, 1.0, 0.001));
    }

    #[test]
    fn smo_train() {
        let mut smo = SMO {
            all_vectors: vec![
                vec![2.0, 2.0],
                vec![1.0, 1.0],
                vec![-1.0, -1.0],
                vec![-2.0, -2.0],
            ],
            label: vec![1.0, 1.0, -1.0, -1.0],
            alpha: vec![0.0, 0.0, 0.0, 0.0],
            bias: 0.0,
            error_cache: vec![0.0, 0.0, 0.0, 0.0],
        };

        let kernel = Linear;

        let (alphas, bias) = smo.train(0.001, 1e-5, 1.0, 100, &kernel);

        assert!(alphas.iter().any(|&a| a > 1e-5));
        assert!(alphas.iter().all(|&a| a >= 0.0 && a <= 1.0));
        assert!(smo.predict(&[3.0, 3.0], &kernel) > 0.0);
        assert!(smo.predict(&[-3.0, -3.0], &kernel) < 0.0);
    }
}

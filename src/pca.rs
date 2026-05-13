pub struct PCA {
    pub mean: Vec<f64>,
    pub components: Vec<Vec<f64>>,
}

impl PCA {
    pub fn fit(data: &[Vec<f64>], n_components: usize) -> Self {
        let n: usize = data.len();
        let d: usize = data[0].len();
        let mut mean: Vec<f64> = vec![0.0; d];

        for row in data {
            for j in 0..d {
                mean[j] += row[j];
            }
        }

        for i in 0..d {
            mean[i] /= n as f64;
        }

        let centered: Vec<Vec<f64>> = data
            .iter()
            .map(|row| row.iter().enumerate().map(|(j, &v)| v - mean[j]).collect())
            .collect();

        let mut cov_matrix: Vec<Vec<f64>> = vec![vec![0.0; d]; d];
        for row in &centered {
            for j in 0..d {
                for k in 0..d {
                    cov_matrix[j][k] += row[j] * row[k];
                }
            }
        }
        for j in 0..d {
            for k in 0..d {
                cov_matrix[j][k] /= (n - 1) as f64;
            }
        }

        let mut components: Vec<Vec<f64>> = Vec::new();
        let mut deflated = cov_matrix.clone();
        for pc in 0..n_components {
            let mut v = vec![0.0; d];
            v[pc % d] = 1.0;

            for _ in 0..1000 {
                let mut new_v = vec![0.0; d];
                for j in 0..d {
                    for k in 0..d {
                        new_v[j] += deflated[j][k] * v[k];
                    }
                }
                let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-10 {
                    break;
                }
                for j in 0..d {
                    v[j] = new_v[j] / norm;
                }
            }

            let eigenvalue: f64 = (0..d)
                .map(|j| (0..d).map(|k| deflated[j][k] * v[k]).sum::<f64>() * v[j])
                .sum();

            for j in 0..d {
                for k in 0..d {
                    deflated[j][k] -= eigenvalue * v[j] * v[k];
                }
            }

            components.push(v);
        }

        PCA { mean, components }
    }

    pub fn transform(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
        data.iter()
            .map(|row| {
                let centered: Vec<f64> = row
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v - self.mean[j])
                    .collect();
                self.components
                    .iter()
                    .map(|pc| pc.iter().zip(&centered).map(|(a, b)| a * b).sum())
                    .collect()
            })
            .collect()
    }
}

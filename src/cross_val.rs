use crate::{
    evaluation::Metrics,
    hyperparameters::{self, Hyperparameters},
    kernel::{self, Kernel, KernelType, make_kernel},
    svm::SVM,
};

fn kfold_indices(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let indices: Vec<usize> = (0..n).collect();
    let mut folds: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
    let fold_size = n / k;

    for i in 0..k {
        let start = i * fold_size;
        let end = if i == k - 1 { n } else { (i + 1) * fold_size };

        let test_indices = indices[start..end].to_vec();
        let mut train_indices = indices[..start].to_vec();
        train_indices.extend_from_slice(&indices[end..]);

        folds.push((train_indices, test_indices));
    }
    folds
}

fn evaluate_fold(
    data: &[Vec<f64>],
    labels: &[f64],
    train_idx: &[usize],
    test_idx: &[usize],
    hyperparameters: &Hyperparameters,
    kernel_type: &KernelType,
) -> f64 {
    let train_data: Vec<Vec<f64>> = train_idx.iter().map(|&i| data[i].clone()).collect();
    let train_label: Vec<f64> = train_idx.iter().map(|&i| labels[i]).collect();

    let test_data: Vec<Vec<f64>> = test_idx.iter().map(|&i| data[i].clone()).collect();
    let test_label: Vec<f64> = test_idx.iter().map(|&i| labels[i]).collect();

    let kernel = make_kernel(kernel_type, hyperparameters.sigma);
    let mut svm = SVM::new(kernel);
    svm.fit(train_data, train_label, *hyperparameters);

    let metrics = Metrics::compute_metric(&svm, &(test_data, test_label));

    metrics.accuracy
}

pub fn cross_validate(
    data: &[Vec<f64>],
    labels: &[f64],
    k: usize,
    hyperparameters: &Hyperparameters,
    kernel_type: &KernelType,
) -> f64 {
    let n: usize = data.len();
    let mut accuracies: Vec<f64> = Vec::new();
    let fold: Vec<(Vec<usize>, Vec<usize>)> = kfold_indices(n, k);

    for (train_idx, test_idx) in &fold {
        let accuracy = evaluate_fold(
            data,
            labels,
            train_idx,
            test_idx,
            hyperparameters,
            kernel_type,
        );
        accuracies.push(accuracy);
    }

    let sum: f64 = accuracies.iter().sum();
    let acc_avg: f64 = sum / accuracies.len() as f64;

    acc_avg
}

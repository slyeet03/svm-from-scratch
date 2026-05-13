use crate::data::{generate_blobs, load_csv, shuffle_data, split_data};
use crate::evaluation::Metrics;
use crate::hyperparameters::Hyperparameters;
use crate::kernel::{Linear, RBF};
use crate::plots::{
    plot_decision_boundary, plot_decision_boundary_3d, plot_learning_curve, plot_metrics,
};
use crate::svm::SVM;

mod cross_val;
mod data;
mod evaluation;
mod grid_search;
mod hyperparameters;
mod kernel;
mod optimizer;
mod pca;
mod plots;
mod svm;

fn main() {
    svm_on_kaggle_dataset_linear();
    svm_on_kaggle_dataset_rbf();
}

fn svm_on_kaggle_dataset_rbf() {
    let (data, labels) = load_csv("test_datasets/breast_cancer_clean.csv", true);
    let (data, labels) = shuffle_data(data, labels, 42);

    let c_values = vec![0.01, 0.1, 1.0, 10.0, 100.0];
    let sigma_values = vec![0.5, 1.0, 2.0, 5.0, 10.0];

    let (train_set, test_set) = split_data(80, 20, &data, &labels);
    let hyperparameters = Hyperparameters::grid_search_tuning(
        &train_set.0,
        &train_set.1,
        &c_values,
        &sigma_values,
        5,
        kernel::KernelType::RBF,
    );

    let mut svm = SVM::new(Box::new(RBF {
        sigma: hyperparameters.sigma,
    }));
    svm.fit(train_set.0, train_set.1, hyperparameters);

    let metrics = Metrics::compute_metric(&svm, &test_set);
    println!("Accuracy: {}", metrics.accuracy);

    plot_metrics(&metrics, "plots/metrics_rbf.png");
    plot_learning_curve(
        &data,
        &labels,
        &hyperparameters,
        &kernel::KernelType::RBF,
        "plots/learning_curve_rbf.png",
    );
    plot_decision_boundary_3d(&svm, &data, &labels, "plots/decision_boundary_rbf.png");
}

fn svm_on_kaggle_dataset_linear() {
    let (data, labels) = load_csv("test_datasets/banknote_clean.csv", true);
    let (data, labels) = shuffle_data(data, labels, 42);

    let c_values = vec![1.0];
    let sigma_values = vec![0.5, 1.0, 2.0];

    let (train_set, test_set) = split_data(80, 20, &data, &labels);
    let hyperparameters = Hyperparameters::grid_search_tuning(
        &train_set.0,
        &train_set.1,
        &c_values,
        &sigma_values,
        5,
        kernel::KernelType::Linear,
    );

    let mut svm = SVM::new(Box::new(Linear));
    svm.fit(train_set.0, train_set.1, hyperparameters);

    let metrics = Metrics::compute_metric(&svm, &test_set);
    println!("Accuracy: {}", metrics.accuracy);

    plot_metrics(&metrics, "plots/metrics_linear.png");
    plot_learning_curve(
        &data,
        &labels,
        &hyperparameters,
        &kernel::KernelType::Linear,
        "plots/learning_curve_linear.png",
    );
    plot_decision_boundary_3d(&svm, &data, &labels, "plots/decision_boundary_linear.png");
}

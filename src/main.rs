use crate::data::{generate_blobs, load_csv, shuffle_data, split_data};
use crate::evaluation::Metrics;
use crate::hyperparameters::Hyperparameters;
use crate::kernel::{Linear, RBF};
use crate::plots::{
    plot_decision_boundary, plot_grid_search_heatmap, plot_learning_curve, plot_metrics,
};
use crate::svm::SVM;

mod cross_val;
mod data;
mod evaluation;
mod grid_search;
mod hyperparameters;
mod kernel;
mod optimizer;
mod plots;
mod svm;

fn main() {
    svm_using_csv_data_rbf();
}

fn svm_using_generated_data_with_linear_kernel() {
    let (data, labels) = generate_blobs(50, 0.2, 42);
    let mut svm = SVM::new(Box::new(Linear));
    let hyperparameters = Hyperparameters::default(data.len());

    svm.fit(data, labels, hyperparameters);

    println!("\nSVM using Linear Kernel with generated data");
    println!("Training complete");
    println!("Support vectors: {}", svm.support_vectors.len());
    println!("Bias: {}", svm.bias);

    let test_points = vec![
        vec![3.0, 3.0],
        vec![1.5, 1.5],
        vec![0.0, 0.0],
        vec![-1.5, -1.5],
        vec![-3.0, -3.0],
    ];

    for point in &test_points {
        let class = svm.predict(point);
        println!("{:?} → class: {}", point, class);
    }
}

fn svm_using_generated_data_with_RBF_kernel() {
    let data = vec![
        vec![1.0, 1.0],
        vec![-1.0, -1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
    ];
    let labels = vec![1.0, 1.0, -1.0, -1.0];

    let mut svm = SVM::new(Box::new(RBF { sigma: 5.0 }));
    let hyperparameters = Hyperparameters::default(data.len());

    svm.fit(data, labels, hyperparameters);

    println!("\nSVM using RBF Kernel with generated data");
    println!("Training complete");
    println!("Support vectors: {}", svm.support_vectors.len());
    println!("Bias: {}", svm.bias);

    let test_points = vec![
        vec![1.0, 1.0],
        vec![-1.0, -1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
        vec![1.2, 0.8],
        vec![0.8, 1.1],
        vec![-1.1, -0.9],
        vec![-0.8, -1.2],
    ];

    for point in &test_points {
        let class = svm.predict(point);
        println!("{:?} → class: {}", point, class);
    }
}

fn svm_using_csv_data_linear() {
    let (data, labels) = load_csv("test_datasets/linear_svm_dataset.csv", true);
    let c_values = vec![0.1, 1.0, 10.0, 100.0];
    let sigma_values = vec![0.1, 0.5, 1.0, 5.0];

    let hyperparameters = Hyperparameters::grid_search_tuning(
        &data,
        &labels,
        &c_values,
        &sigma_values,
        5,
        kernel::KernelType::Linear,
    );
    let (train_set, test_set) = split_data(80, 20, &data, &labels);

    let mut svm = SVM::new(Box::new(Linear));

    svm.fit(train_set.0, train_set.1, hyperparameters);
    println!("\nSVM using Linear Kernel with csv data");
    println!("Training complete");
    println!("Support vectors: {}", svm.support_vectors.len());
    println!("Bias: {}", svm.bias);

    let test_points = vec![
        vec![4.0, 4.0],
        vec![-4.0, -4.0],
        vec![1.0, 1.0],
        vec![-1.0, -2.0],
    ];

    for point in &test_points {
        let class = svm.predict(point);
        println!("{:?} -> class: {}", point, class);
    }

    let metrics = Metrics::compute_metric(&svm, &test_set);
    println!("Accuracy: {}", metrics.accuracy);
    plot_metrics(&metrics, "plots/metrics.png");
    plot_learning_curve(
        &data,
        &labels,
        &hyperparameters,
        &kernel::KernelType::RBF,
        "plots/learning_curve.png",
    );
    plot_decision_boundary(&svm, &data, &labels, "plots/decision_boundary.png");
}

fn svm_using_csv_data_rbf() {
    let (data, labels) = load_csv("test_datasets/rbf_svm_dataset.csv", true);
    let (data, labels) = shuffle_data(data, labels, 42);
    let c_values = vec![0.1, 1.0, 10.0, 100.0];
    let sigma_values = vec![0.1, 0.5, 1.0, 5.0];

    let hyperparameters = Hyperparameters::grid_search_tuning(
        &data,
        &labels,
        &c_values,
        &sigma_values,
        5,
        kernel::KernelType::RBF,
    );

    let (train_set, test_set) = split_data(80, 20, &data, &labels);

    let mut svm = SVM::new(Box::new(RBF {
        sigma: hyperparameters.sigma,
    }));

    svm.fit(train_set.0, train_set.1, hyperparameters);
    println!("\nSVM using RBF Kernel with csv data");
    println!("Training complete");
    println!("Support vectors: {}", svm.support_vectors.len());
    println!("Bias: {}", svm.bias);

    let test_points = vec![
        vec![1.0, 1.0],
        vec![-1.0, -1.0],
        vec![1.0, -1.0],
        vec![-1.0, 1.0],
    ];

    for point in &test_points {
        let class = svm.predict(point);
        println!("{:?} -> class: {}", point, class);
    }

    let metrics = Metrics::compute_metric(&svm, &test_set);
    println!("Accuracy: {}", metrics.accuracy);

    plot_metrics(&metrics, "plots/metrics.png");
    plot_learning_curve(
        &data,
        &labels,
        &hyperparameters,
        &kernel::KernelType::RBF,
        "plots/learning_curve.png",
    );
    plot_decision_boundary(&svm, &data, &labels, "plots/decision_boundary.png");
}

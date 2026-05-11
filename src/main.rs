use crate::data::generate_blobs;
use crate::kernel::{Linear, RBF};
use crate::svm::SVM;

mod data;
mod kernel;
mod optimizer;
mod svm;

fn main() {
    svm_using_generated_data_with_linear_kernel();
    svm_using_generated_data_with_RBF_kernel();
}

fn svm_using_generated_data_with_linear_kernel() {
    let (data, labels) = generate_blobs(50, 0.2, 42);
    let mut svm = SVM::new(Box::new(Linear));

    svm.fit(data, labels, 1.0, 0.001, 1e-5, 100);

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

    svm.fit(data, labels, 1.0, 0.001, 1e-5, 100);

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

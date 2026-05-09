use crate::kernel::Kernel;

mod kernel;

fn main() {
    let ker = kernel::RBF { sigma: 1.0 };

    let sum: f64 = ker.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]);

    println!("{}", sum);
}

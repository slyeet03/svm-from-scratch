use crate::kernel::Kernel;

mod kernel;

fn main() {
    let ker = kernel::Linear;

    let sum: f64 = ker.compute(&[2.0_f64, 3.0_f64], &[4.0_f64, 1.0_f64]);

    println!("{}", sum);
}

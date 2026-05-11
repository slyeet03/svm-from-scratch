use rand::rngs::StdRng;
use rand::{Rng, RngExt, SeedableRng};
use std::fs::File;
use std::io::{BufRead, BufReader};

pub fn generate_blobs(n: usize, spread: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();

    for i in 0..n {
        let x1 = 2.0 + rng.random_range(-1.0..1.0) * spread;
        let x2 = 2.0 + rng.random_range(-1.0..1.0) * spread;
        points.push(vec![x1, x2]);
        labels.push(1.0);
    }

    for i in 0..n {
        let x1 = -2.0 + rng.random_range(-1.0..1.0) * spread;
        let x2 = -2.0 + rng.random_range(-1.0..1.0) * spread;
        points.push(vec![x1, x2]);
        labels.push(-1.0);
    }

    (points, labels)
}

// last column should be label
// other columns being features
// whether the csv feel has header or not must be given by the user
pub fn load_csv(path: &str, has_header: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
    let file = File::open(path).expect("could not open file");
    let reader = BufReader::new(file);
    let mut points: Vec<Vec<f64>> = Vec::new();
    let mut labels: Vec<f64> = Vec::new();
    let mut first_line = true;

    for line in reader.lines() {
        let line = line.expect("could not read line");
        if has_header && first_line {
            first_line = false;
            continue;
        }

        first_line = false;
        let values: Vec<&str> = line.split(',').collect();
        if values.len() < 2 {
            continue;
        }

        let point: Vec<f64> = values[..values.len() - 1]
            .iter()
            .map(|v| v.trim().parse::<f64>().expect("could not parse feature"))
            .collect();

        let raw_label: f64 = values[values.len() - 1]
            .trim()
            .parse::<f64>()
            .expect("could not parse label");

        let label = if raw_label == 0.0 { -1.0 } else { raw_label };

        points.push(point);
        labels.push(label);
    }

    (points, labels)
}

pub fn split_data(
    train_perc: usize,
    test_perc: usize,
    points: Vec<Vec<f64>>,
    labels: Vec<f64>,
) -> ((Vec<Vec<f64>>, Vec<f64>), (Vec<Vec<f64>>, Vec<f64>)) {
    let mut train_set: (Vec<Vec<f64>>, Vec<f64>) = (Vec::new(), Vec::new());

    let mut test_set: (Vec<Vec<f64>>, Vec<f64>) = (Vec::new(), Vec::new());

    let n = points.len();

    let train_index_end = n * train_perc / 100;
    let test_index_end = train_index_end + (n * test_perc / 100);

    for i in 0..train_index_end {
        train_set.0.push(points[i].clone());
        train_set.1.push(labels[i]);
    }

    for j in train_index_end..test_index_end {
        test_set.0.push(points[j].clone());
        test_set.1.push(labels[j]);
    }

    (train_set, test_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gen_blob_correct_number_of_points() {
        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();

        (points, labels) = generate_blobs(50, 0.5, 42);

        assert_eq!(points.len(), 100);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn gen_blob_labels_only_plus_minus_one() {
        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();

        (points, labels) = generate_blobs(50, 0.5, 42);

        assert!(labels.iter().all(|&l| l == 1.0 || l == -1.0));
    }

    #[test]
    fn gen_blob_equal_number_classes() {
        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();

        (points, labels) = generate_blobs(50, 0.5, 42);

        let pos = labels.iter().filter(|&&l| l == 1.0).count();
        let neg = labels.iter().filter(|&&l| l == -1.0).count();
        assert_eq!(pos, 50);
        assert_eq!(neg, 50);
    }

    #[test]
    fn gen_blob_points_labels_same_length() {
        let mut points: Vec<Vec<f64>> = Vec::new();
        let mut labels: Vec<f64> = Vec::new();

        (points, labels) = generate_blobs(50, 0.5, 42);

        assert_eq!(points.len(), labels.len());
    }

    #[test]
    fn gen_blob_same_seed_same_data() {
        let (data1, labels1) = generate_blobs(50, 0.5, 42);
        let (data2, labels2) = generate_blobs(50, 0.5, 42);

        assert_eq!(data1, data2);
        assert_eq!(labels1, labels2);
    }

    #[test]
    fn load_csv_correct_values() {
        let csv_content = "1.0,2.0,1.0\n-1.0,-2.0,-1.0\n";
        std::fs::write("/tmp/test_data.csv", csv_content).expect("could not write test file");
        let (points, labels) = load_csv("/tmp/test_data.csv", false);
        assert_eq!(points.len(), 2);
        assert_eq!(labels.len(), 2);
        assert!((points[0][0] - 1.0).abs() < 1e-10);
        assert!((points[0][1] - 2.0).abs() < 1e-10);
        assert!((labels[0] - 1.0).abs() < 1e-10);
        assert!((points[1][0] - (-1.0)).abs() < 1e-10);
        assert!((labels[1] - (-1.0)).abs() < 1e-10);
        std::fs::remove_file("/tmp/test_data.csv").expect("could not remove test file");
    }
}

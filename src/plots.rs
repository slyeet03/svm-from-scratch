//used AI for plot.rs

use plotters::{
    chart::ChartBuilder,
    prelude::{BitMapBackend, Circle, IntoDrawingArea, PathElement, Rectangle, Text},
    series::LineSeries,
    style::{BLACK, BLUE, Color, IntoFont, RED, RGBColor, WHITE},
};

use crate::{
    evaluation::Metrics,
    hyperparameters::Hyperparameters,
    kernel::{KernelType, make_kernel},
    svm::SVM,
};

pub fn plot_decision_boundary(svm: &SVM, data: &[Vec<f64>], labels: &[f64], path: &str) {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let x_min = data.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - 1.0;
    let x_max = data.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max) + 1.0;
    let y_min = data.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - 1.0;
    let y_max = data.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max) + 1.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("SVM Decision Boundary", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // grid resolution
    let resolution = 300;
    let x_step = (x_max - x_min) / resolution as f64;
    let y_step = (y_max - y_min) / resolution as f64;

    // draw decision regions
    for i in 0..resolution {
        for j in 0..resolution {
            let x = x_min + i as f64 * x_step;
            let y = y_min + j as f64 * y_step;
            let pred = svm.predict_raw(&[x, y]);
            let color = if pred >= 0.0 {
                RGBColor(180, 210, 255).mix(0.5) // light blue for +1
            } else {
                RGBColor(255, 180, 180).mix(0.5) // light red for -1
            };
            chart
                .draw_series(std::iter::once(Rectangle::new(
                    [(x, y), (x + x_step, y + y_step)],
                    color.filled(),
                )))
                .unwrap();
        }
    }

    // draw data points
    for (i, point) in data.iter().enumerate() {
        let color = if labels[i] == 1.0 { &BLUE } else { &RED };
        chart
            .draw_series(std::iter::once(Circle::new(
                (point[0], point[1]),
                4,
                color.filled(),
            )))
            .unwrap();
    }

    // highlight support vectors with a ring
    for sv in &svm.support_vectors {
        chart
            .draw_series(std::iter::once(Circle::new(
                (sv[0], sv[1]),
                8,
                BLACK.stroke_width(2),
            )))
            .unwrap();
    }

    root.present().unwrap();
}

pub fn plot_metrics(metrics: &Metrics, path: &str) {
    let bars: Vec<(&str, f64)> = vec![
        ("Accuracy", metrics.accuracy),
        ("Precision", metrics.precision),
        ("Recall", metrics.recall),
        ("F1", metrics.F1),
    ];

    let root = BitMapBackend::new(path, (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Metrics", ("sans-serif", 30).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..4f64, 0f64..4f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(0)
        .y_desc("Score")
        .draw()
        .unwrap();

    // bars
    for (i, (name, value)) in bars.iter().enumerate() {
        let x0 = i as f64 + 0.1;
        let x1 = i as f64 + 0.9;
        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, *value)],
                BLUE.filled(),
            )))
            .unwrap();
    }

    // metric names -> x labels
    for (i, (name, _)) in bars.iter().enumerate() {
        let x = i as f64 + 0.5;
        chart
            .draw_series(std::iter::once(Text::new(
                *name,
                (x, -0.05),
                ("sans-serif", 15).into_font(),
            )))
            .unwrap();
    }

    // value label above bars
    for (i, (_, value)) in bars.iter().enumerate() {
        let x = i as f64 + 0.5;
        let label = format!("{:.2}", value);
        chart
            .draw_series(std::iter::once(Text::new(
                label,
                (x, value + 0.02),
                ("sans-serif", 13).into_font(),
            )))
            .unwrap();
    }

    root.present().unwrap();
}

pub fn plot_grid_search_heatmap(all_scores: &[(f64, f64, f64)], path: &str) {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut c_values: Vec<f64> = all_scores.iter().map(|(c, _, _)| *c).collect();
    let mut sigma_values: Vec<f64> = all_scores.iter().map(|(_, s, _)| *s).collect();
    c_values.dedup();
    sigma_values.dedup();
    c_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sigma_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_score = all_scores
        .iter()
        .map(|(_, _, s)| *s)
        .fold(f64::INFINITY, f64::min);
    let max_score = all_scores
        .iter()
        .map(|(_, _, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);

    let nc = c_values.len();
    let ns = sigma_values.len();

    let mut chart = ChartBuilder::on(&root)
        .caption("Grid Search Heatmap", ("sans-serif", 25))
        .margin(40)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0f64..nc as f64, 0f64..ns as f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_labels(nc)
        .y_labels(ns)
        .x_label_formatter(&|x| {
            let i = *x as usize;
            if i < c_values.len() {
                format!("{:.2}", c_values[i])
            } else {
                "".to_string()
            }
        })
        .y_label_formatter(&|y| {
            let i = *y as usize;
            if i < sigma_values.len() {
                format!("{:.2}", sigma_values[i])
            } else {
                "".to_string()
            }
        })
        .draw()
        .unwrap();

    for (c, sigma, score) in all_scores {
        let ci = c_values.iter().position(|v| v == c).unwrap();
        let si = sigma_values.iter().position(|v| v == sigma).unwrap();

        let t = if (max_score - min_score).abs() < 1e-10 {
            1.0
        } else {
            (score - min_score) / (max_score - min_score)
        };

        // blue (low) to yellow (high)
        let r = (255.0 * t) as u8;
        let g = (255.0 * t) as u8;
        let b = (255.0 * (1.0 - t)) as u8;
        let color = RGBColor(r, g, b);

        chart
            .draw_series(std::iter::once(Rectangle::new(
                [(ci as f64, si as f64), (ci as f64 + 1.0, si as f64 + 1.0)],
                color.filled(),
            )))
            .unwrap();
    }

    root.present().unwrap();
}

pub fn plot_learning_curve(
    data: &[Vec<f64>],
    labels: &[f64],
    hyperparameters: &Hyperparameters,
    kernel_type: &KernelType,
    path: &str,
) {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let n = data.len();
    let test_split = (n as f64 * 0.2) as usize;
    let test_data = &data[n - test_split..];
    let test_labels = &labels[n - test_split..];
    let train_data = &data[..n - test_split];
    let train_labels = &labels[..n - test_split];

    let steps: Vec<usize> = (1..=10)
        .map(|i| (train_data.len() * i / 10).max(2))
        .collect();
    let mut train_scores: Vec<(f64, f64)> = Vec::new();
    let mut test_scores: Vec<(f64, f64)> = Vec::new();

    for &size in &steps {
        let d = train_data[..size].to_vec();
        let l = train_labels[..size].to_vec();

        let kernel = make_kernel(kernel_type, hyperparameters.sigma);
        let mut svm = SVM::new(kernel);
        svm.fit(d.clone(), l.clone(), *hyperparameters);

        let train_metrics = Metrics::compute_metric(&svm, &(d, l));
        let test_metrics =
            Metrics::compute_metric(&svm, &(test_data.to_vec(), test_labels.to_vec()));

        train_scores.push((size as f64, train_metrics.accuracy));
        test_scores.push((size as f64, test_metrics.accuracy));
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Learning Curve", ("sans-serif", 25))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(0f64..train_data.len() as f64, 0f64..1f64)
        .unwrap();

    chart
        .configure_mesh()
        .x_desc("Training size")
        .y_desc("Accuracy")
        .draw()
        .unwrap();

    chart
        .draw_series(LineSeries::new(train_scores.clone(), &BLUE))
        .unwrap()
        .label("Train")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(test_scores.clone(), &RED))
        .unwrap()
        .label("Test")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    // dots on the lines
    for (x, y) in &train_scores {
        chart
            .draw_series(std::iter::once(Circle::new((*x, *y), 4, BLUE.filled())))
            .unwrap();
    }
    for (x, y) in &test_scores {
        chart
            .draw_series(std::iter::once(Circle::new((*x, *y), 4, RED.filled())))
            .unwrap();
    }

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()
        .unwrap();

    root.present().unwrap();
}

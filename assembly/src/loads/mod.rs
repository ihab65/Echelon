pub mod series;
pub mod pattern;

pub use series::{TimeSeries, ConstantSeries, LinearSeries, PathSeries};
pub use pattern::{LoadPattern, NodalLoad};
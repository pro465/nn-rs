pub mod dataset;
pub mod nn;

pub type Func = (fn(f64) -> f64, fn(f64) -> f64);

pub(crate) type Matrix = Vec<Vec<f64>>;
pub(crate) mod defaults;
pub(crate) mod helper;
pub(crate) mod neuron;
pub(crate) mod rand;

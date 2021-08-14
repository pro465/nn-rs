//mofule for dataset typedefs
pub mod dataset;
//module for neural network API's
pub mod nn;

//typedef for function-derivative pair
pub type Func = (fn(f64) -> f64, fn(f64) -> f64);

//internal helpers
pub(crate) type Matrix = Vec<Vec<f64>>;
pub(crate) mod defaults;
pub(crate) mod helper;
pub(crate) mod neuron;
pub(crate) mod rand;

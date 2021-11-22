//module for dataset typedefs
pub mod dataset;
//module for neural network API
pub mod nn;

// module for default values
pub mod defaults;

mod rng;

// trait representing an RNG that can be used during init of the nn
pub use rng::Rng;

//typedef for function-derivative pair
pub type Func = (fn(f64) -> f64, fn(f64) -> f64);

//internal helpers

pub(crate) type Matrix = Vec<Vec<f64>>;
pub(crate) mod helper;
pub(crate) mod neuron;

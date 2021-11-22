//default parameters used during intialization of the neural network
use crate::Func;

pub(crate) const MOMENTUM: f64 = 0.9;
pub(crate) const LEARNING_RATE: f64 = 0.01;
pub(crate) const ERR_THRESHOLD: f64 = 0.001;
pub(crate) const FUNC: Func = (
    //leaky ReLU
    |x| f64::max(0.01 * x, x),
    //leaky ReLU derivative
    |x| f64::max((x >= 0.) as u8 as f64, 0.01),
);

// default RNG in case the user is lazy and wanna trade work time with compile time lol
#[cfg(feature = "rand")]
use rand::prelude::{Rng, ThreadRng};

#[cfg(feature = "rand")]
pub struct DefaultRng(ThreadRng);

#[cfg(feature = "rand")]
impl DefaultRng {
    pub fn new() -> Self {
        Self(rand::thread_rng())
    }
}

#[cfg(feature = "rand")]
impl crate::Rng for DefaultRng {
    fn gen(&mut self) -> f64 {
        self.0.gen_range(-3. ..3.)
    }
}

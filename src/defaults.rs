use crate::Func;

pub(crate) const MOMENTUM: f64 = 0.9;
pub(crate) const LEARNING_RATE: f64 = 0.01;
pub(crate) const ERR_THRESHOLD: f64 = 0.001;
pub(crate) const FUNC: Func = (
    |x| f64::max(0.01 * x, x),
    |x| f64::max((x >= 0.) as u8 as f64, 0.01),
);

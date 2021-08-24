//for getting random numbers to initialize weight
use rand::prelude::*;

//a struct representing a neuron
pub(crate) struct Neuron {
    weights: Vec<f64>,
    bias: f64,

    //previous weight and bias update, necessary to implement momentum
    //the Option<...> is needed to replace ".clone()" with less expensive ".take().unwrap()"
    prev_weights_update: Option<Vec<f64>>,
    prev_bias_update: f64,
}

impl Neuron {
    //for initializing a neuron
    pub(crate) fn new(prev_layer_len: usize, rand: &mut ThreadRng) -> Self {
        Self {
            //initializing weights and bias with 5.0 and -5.0
            weights: (0..prev_layer_len)
                .map(|_| rand.gen_range(-5. ..=5.))
                .collect(),
            bias: rand.gen_range(-5. ..=5.),

            prev_weights_update: Some(vec![0.; prev_layer_len]),
            prev_bias_update: 0.,
        }
    }

    pub(crate) fn train(
        &mut self,

        //partial errors and partial errors for next iter(previous layer)
        (errors, next): (&[f64], &mut [Vec<f64>]),

        //learning rate and momentum
        (lr, m): (f64, f64),

        //previous layer's outputs per input in dataset
        outputs: &[Vec<f64>],
    ) {
        //no assummptions to introduce bugs, only assertions to help debug ;)
        assert_eq!(self.weights.len(), outputs.len());
        assert!(outputs.iter().all(|x| x.len() == errors.len()));

        //getting bias update amount from total error of bias
        let total_error_of_bias: f64 = errors.iter().sum::<f64>() / errors.len() as f64;
        let bias_update = total_error_of_bias * lr + m * self.prev_bias_update;

        //training bias and updating previous bias update
        self.bias -= bias_update;
        self.prev_bias_update = bias_update;

        //get next partial errors (for previous layers)
        self.weights.iter().zip(next.iter_mut()).for_each(|(w, n)| {
            errors
                .iter()
                .zip(n.iter_mut())
                .for_each(|(e, n)| *n += w * e)
        });

        //this is the reason behind Option<Vec<_>> instead of Vec<_>
        let mut prev = self.prev_weights_update.take().unwrap();

        //training weights

        self.weights
            .iter_mut()
            .zip(outputs.iter())
            .enumerate()
            .for_each(|(i, (x, o))| {
                let error =
                    o.iter().zip(errors.iter()).map(|(o, e)| o * e).sum::<f64>() / o.len() as f64;

                let update = error * lr + prev[i] * m;

                *x -= update;
                prev[i] = update;
            });

        //updating previous updates to current updates
        self.prev_weights_update = Some(prev);
    }

    pub(crate) fn act(&self, data: &[f64]) -> f64 {
        //weighted sum + bias

        self.weights
            .iter()
            .zip(data.iter())
            .map(|(w, d)| w * d)
            .sum::<f64>()
            + self.bias
    }
}

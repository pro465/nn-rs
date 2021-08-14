//dataset
use crate::dataset::*;
//default parameters
use crate::defaults;
//helper
use crate::helper;
//a struct representing a neuron
use crate::neuron::Neuron;
//to get "random" numbers
use crate::rand::Rand;
//a function-derivative pair
use crate::Func;
//matrix typedef
use crate::Matrix;

pub struct NeuralNetwork {
    input_length: usize,
    output_length: usize,

    network: Vec<Vec<Neuron>>,
    funcs: Vec<Func>,

    lr: f64,
    momentum: f64,
    err_thres: f64,
    total_err: f64,
}

impl NeuralNetwork {
    pub fn new(layout: &[usize]) -> Self {
        assert!(layout.len() >= 2);

        let mut rand = Rand::new();

        let input_length = layout[0];

        let network: Vec<Vec<Neuron>> = layout
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, num_neurons)| {
                let mut layer = Vec::new();

                let last_len = if i == 0 { input_length } else { layout[i - 1] };

                for _ in 0..*num_neurons {
                    layer.push(Neuron::new(last_len, &mut rand));
                }

                layer
            })
            .collect();

        let mut funcs: Vec<Func> = layout.iter().map(|_| defaults::FUNC).collect();

        *funcs.last_mut().unwrap() = (|x| x, |_| 1.);

        let output_length = network[network.len() - 1].len();

        Self {
            input_length,
            output_length,

            network,
            funcs,

            lr: defaults::LEARNING_RATE,
            momentum: defaults::MOMENTUM,
            err_thres: defaults::ERR_THRESHOLD,
            total_err: f64::INFINITY,
        }
    }

    pub fn train<const I: usize, const O: usize>(
        &mut self,
        num_times: u32,
        dataset: &[IOPair<I, O>],
    ) {
        assert_eq!(I, self.input_length);
        assert_eq!(O, self.output_length);

        for _ in 0..num_times {
            self.train_single(dataset);

            let err = self.total_err;
            if err <= self.err_thres {
                break;
            }
        }
    }

    pub fn predict(&self, data: Vec<f64>) -> Vec<f64> {
        assert_eq!(data.len(), self.input_length);

        self.predict_common(data).2
    }

    pub fn set_lr(&mut self, new_lr: f64) {
        assert!(0. < new_lr && new_lr <= 1.);

        self.lr = new_lr;
    }

    pub fn set_momentum(&mut self, new_momentum: f64) {
        assert!((0. ..1.).contains(&new_momentum));

        self.momentum = new_momentum;
    }

    pub fn set_err_thres(&mut self, new_err_thres: f64) {
        assert!(new_err_thres > 0.);

        self.err_thres = new_err_thres;
    }

    pub fn set_act_func(&mut self, layer: usize, new_func: Func) {
        self.funcs[layer] = new_func;
    }

    fn train_single<const I: usize, const O: usize>(&mut self, dataset: &[IOPair<I, O>]) {
        let (inputs, expected_outputs): (Vec<_>, Vec<_>) = dataset.iter().cloned().unzip();

        let iter = inputs.into_iter().map(|x| self.predict_common(x.to_vec()));

        let (a, o, outputs) = helper::unwrap(iter);

        let lr = self.lr;
        let momentum = self.momentum;

        let mut errors =
            vec![Vec::with_capacity(expected_outputs.len()); expected_outputs[0].len()];

        for (o, e) in outputs.iter().zip(expected_outputs.iter()) {
            o.iter()
                .zip(e.iter())
                .enumerate()
                .for_each(|(i, (o, e))| errors[i].push(o - e));
        }

        self.total_err = errors
            .iter()
            .map(|x| x.iter().cloned().map(f64::abs).sum::<f64>() / x.len() as f64)
            .sum::<f64>();

        let mut layer_no = self.network.len() - 1;

        loop {
            let cond = layer_no == 0;

            let prev_len = self.input_length * cond as usize
                + self.network[layer_no - 1].len() * (!cond) as usize;

            let mut next = vec![vec![0.; expected_outputs.len()]; prev_len];

            let prev_layer_o = helper::transpose(o.iter().map(|x| &*x[layer_no]).collect());
            let (_func, der) = self.funcs[layer_no];

            for (i, ref mut error) in errors.iter_mut().enumerate() {
                let curr_node_a = a.iter().map(|x| x[layer_no][i]).collect();

                self.network[layer_no][i].train(
                    (error, &mut next),
                    (&*prev_layer_o, curr_node_a),
                    (lr, momentum),
                    der,
                )
            }

            if cond {
                break;
            }

            errors = next;
            layer_no -= 1;
        }
    }

    #[inline]
    fn predict_common(&self, mut input: Vec<f64>) -> (Matrix, Matrix, Vec<f64>) {
        let mut a: Vec<Vec<f64>> = Vec::with_capacity(self.network.len());
        let mut o: Vec<Vec<f64>> = vec![input.clone()];

        for (i, x) in self.network.iter().enumerate() {
            let (func, _der) = self.funcs[i];

            let curr_a: Vec<f64> = x.iter().map(|x| x.act(&input)).collect();

            let curr_o: Vec<f64> = curr_a.iter().cloned().map(func).collect();

            a.push(curr_a);
            o.push(curr_o.clone());
            input = curr_o
        }

        (a, o, input)
    }
}

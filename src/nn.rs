use crate::dataset::IOPair;
use crate::defaults;
use crate::helper;
use crate::neuron::Neuron;
use crate::Func;
use crate::Matrix;
use crate::Rng;

use std::iter::once;

//struct representing a neural network
pub struct NeuralNetwork {
    //lengths of input and output layers, to assert and help debug if unexpected things occur
    input_length: usize,
    output_length: usize,

    //layers of neuronns and their activation-derivative function pair
    network: Vec<Vec<Neuron>>,
    funcs: Vec<Func>,

    //learning rate
    lr: f64,

    //momentum
    momentum: f64,

    //error theshold to compare against total_err
    //and halt training process if total_err < err_thres
    err_thres: f64,
}

impl NeuralNetwork {
    //method to initialize a new neural network
    pub fn new(
        // layout
        layout: &[usize],

        // RNG
        rng: &mut impl Rng,
    ) -> Self {
        //layout should have at least an input layer and an output layer
        assert!(
            layout.len() >= 2,
            "layout should have at least an input layer and an output layer"
        );

        //input layer length
        let input_length = layout[0];

        let network: Vec<Vec<Neuron>> = layout
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, num_neurons)| {
                let mut layer = Vec::new();

                let last_len = layout[i - 1];

                for _ in 0..*num_neurons {
                    layer.push(Neuron::new(last_len, rng));
                }

                // a new layer with `num_neurons` neurons
                layer
            })
            .collect();

        //default function-derivative pair per layer
        let mut funcs: Vec<Func> = layout.iter().map(|_| defaults::FUNC).collect();

        //default output layer function-derivative pair
        *funcs.last_mut().unwrap() = (|x| x, |_| 1.);

        //number of output layer's neurons
        let output_length = network[network.len() - 1].len();

        Self {
            input_length,
            output_length,

            network,
            funcs,

            lr: defaults::LEARNING_RATE,
            momentum: defaults::MOMENTUM,
            err_thres: defaults::ERR_THRESHOLD,
        }
    }

    //public training interface
    pub fn train(&mut self, num_times: u32, dataset: &[IOPair]) {
        assert!(dataset
            .iter()
            .all(|(i, o)| i.len() == self.input_length && o.len() == self.output_length));

        //extracting inputs and expected output from dataset
        let (inputs, expected_outputs): (Vec<_>, Vec<_>) =
            dataset.iter().map(|x| (&x.0, &x.1)).unzip();

        // cached vec, to avoid allocating multiple times
        let mut cache: (Vec<Matrix>, Vec<Matrix>) = (
            vec![
                self.network
                    .iter()
                    .map(|x| Vec::with_capacity(x.len()))
                    .collect();
                dataset.len()
            ],
            vec![
                once(self.input_length)
                    .chain(self.network.iter().map(Vec::len))
                    .map(Vec::with_capacity)
                    .collect();
                dataset.len()
            ],
        );

        cache
            .1
            .iter_mut()
            .zip(inputs.into_iter())
            .for_each(|(o, i)| o[0].extend_from_slice(i));

        //error to be passed to each neuron to teach themselves
        let mut errors =
            vec![Vec::with_capacity(expected_outputs.len()); expected_outputs[0].len()];

        let mut next: Vec<Matrix> = once(self.input_length)
            .chain(self.network.iter().map(Vec::len))
            .take(self.network.len())
            .map(|prev_len| vec![vec![0.; expected_outputs.len()]; prev_len])
            .collect();

        for _ in 0..num_times {
            let err = self.train_single(
                &expected_outputs,
                (&mut cache.0, &mut cache.1),
                (&mut errors, &mut next),
            );

            if err <= self.err_thres {
                break;
            }
        }
    }

    //public prediction method
    #[must_use]
    pub fn predict(&self, mut input: Vec<f64>) -> Vec<f64> {
        assert_eq!(input.len(), self.input_length);

        for (i, layer) in self.network.iter().enumerate() {
            //extracting activation function for current layer
            let (func, _der) = self.funcs[i];

            // output of curr layer
            let mut curr = Vec::with_capacity(layer.len());

            for neuron in layer.iter() {
                let output = func(neuron.act(&input));

                curr.push(output)
            }

            // output of curr layer is gonna be the input of next layer
            input = curr;
        }

        input
    }

    //====
    //setter interfaces
    //====

    //learning rate
    pub fn set_lr(&mut self, new_lr: f64) {
        assert!(0. < new_lr && new_lr <= 1.);

        self.lr = new_lr;
    }

    //momentum
    pub fn set_momentum(&mut self, new_momentum: f64) {
        assert!((0. ..1.).contains(&new_momentum));

        self.momentum = new_momentum;
    }

    //error threshold
    pub fn set_err_thres(&mut self, new_err_thres: f64) {
        assert!(new_err_thres > 0.);

        self.err_thres = new_err_thres;
    }

    //activation-derivative pair for a particular layer
    pub fn set_act_func(&mut self, layer: usize, new_func: Func) {
        self.funcs[layer] = new_func;
    }

    //====
    //end setter interfaces
    //====

    //to train a single time

    #[inline]
    fn train_single(
        &mut self,
        expected_outputs: &[&Vec<f64>],
        (a, o): (&mut [Matrix], &mut [Matrix]),
        (errors, next): (&mut Matrix, &mut [Matrix]),
    ) -> f64 {
        //extracting activations and outputs per neuron per layer per input and output of final
        //layer per input from the returned value of predict_common
        let outputs = a
            .iter_mut()
            .zip(o.iter_mut())
            .map(|x| self.predict_and_record(x));

        //to allow for less typing :)
        let lr = self.lr;
        let momentum = self.momentum;

        //error init
        for (o, e) in outputs.zip(expected_outputs.iter()) {
            o.iter()
                .zip(e.iter())
                .enumerate()
                .for_each(|(i, (o, e))| errors[i].push(o - e));
        }

        //total error, to be compared to err_thres and stop learning if it is tolerable
        let total_err = errors
            .iter()
            .map(|x| x.iter().copied().map(f64::abs).sum::<f64>() / x.len() as f64)
            .sum::<f64>();

        let mut layer_no = self.network.len() - 1;

        loop {
            let next = &mut next[layer_no];

            //layer-level constants
            let prev_layer_o = helper::transpose(o.iter().map(|x| &*x[layer_no]), a.len());
            let (_func, der) = self.funcs[layer_no];

            for (i, ref mut error) in errors.iter_mut().enumerate() {
                //current neuron's activation per IOPair
                let curr_node_a = a.iter_mut().map(|x| x[layer_no][i]);

                //to get complete error from partial error
                error
                    .iter_mut()
                    .zip(curr_node_a)
                    .for_each(|(x, a)| *x *= der(a));

                //asking each neuron to train itself
                self.network[layer_no][i].train((error, next), (lr, momentum), &*prev_layer_o)
            }

            if layer_no == 0 {
                break;
            }

            std::mem::swap(errors, next);
            layer_no -= 1;
        }

        for i in next {
            std::mem::swap(errors, i);
            i.iter_mut().for_each(|x| x.fill(0.));
        }

        // cleanup
        for i in a {
            i.iter_mut().for_each(Vec::clear);
        }

        for i in o {
            i.iter_mut().skip(1).for_each(Vec::clear);
        }

        for i in errors {
            i.clear();
        }

        total_err
    }

    //like `predict`, but also records activations and outputs of each neuron

    #[inline]
    fn predict_and_record(&self, (a, o): (&mut Matrix, &mut Matrix)) -> Vec<f64> {
        for (i, layer) in self.network.iter().enumerate() {
            //extracting activation function for current layer
            let (func, _der) = self.funcs[i];

            for neuron in layer.iter() {
                let act = neuron.act(&o[i]);
                a[i].push(act);

                let output = func(act);
                o[i + 1].push(output);
            }
        }

        //output of output layer
        o.last().unwrap().clone()
    }
}

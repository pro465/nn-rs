use crate::dataset::*;
use crate::neuron::Neuron;
use crate::rand::Rand;
use crate::helper;
use crate::defaults;
use crate::Func;
use crate::Matrix;

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

    //total error of previous prediction
    total_err: f64,
}

impl NeuralNetwork {
    //method to initialize a new neural network
    pub fn new(layout: &[usize]) -> Self {
        //layout should have at least an input layer and an output layer
        assert!(layout.len() >= 2);

        //"random" number generator
        let mut rand = Rand::new();

        //input layer length
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

                // a new layer with num_neurons neurons
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
            total_err: f64::INFINITY,
        }
    }

    //public traiming interface
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

    //public prediction method
    pub fn predict(&self, data: Vec<f64>) -> Vec<f64> {
        assert_eq!(data.len(), self.input_length);

        self.predict_common(data).2
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

    //sctivation-derivative pair for a particular layer
    pub fn set_act_func(&mut self, layer: usize, new_func: Func) {
        self.funcs[layer] = new_func;
    }

    //====
    //end setter interfaces
    //====

    //to train a single time

    #[inline]
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

    //the core prediction function,
    //which is reused by both train_single and predict method
    
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

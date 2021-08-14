//this is a simple example that uses a neural network with 
//   2 input layer neurons, 
//   3 hidden layer neurons 
//   and 1 output layer neuron 
//to approximate a XOR function

use nn::dataset;
use nn::nn::NeuralNetwork;

fn main() {
    let mut nn = NeuralNetwork::new(&[2,3,1]);
    nn.train(10000, &dataset!(
        [0., 0.] => [0.],
        [0., 1.] => [1.],
        [1., 0.] => [1.],
        [1., 1.] => [0.],
    ));

    println!("{:?}", nn.predict(vec![0., 0.]));
    println!("{:?}", nn.predict(vec![0., 1.]));
    println!("{:?}", nn.predict(vec![1., 0.]));
    println!("{:?}", nn.predict(vec![1., 1.]));
}

# nn-rs
a minimalistic feedforward neural network library in Rust
# examples
## Simple usage
   this is a simple code that uses a neural network with 2 input layer neurons, 3 hidden layer neurons and 1 output layer neuron to approximate a xor function
   ```rust
      use nn::dataset::dataset;
      use nn::nn::NeuralNetwork;
      let nn = NeuralNetwork::new(&[2,3,1]);
      nn.train(10000, &dataset!(
          [0, 0] => [0],
          [0, 1] => [1],
          [1, 0] => [1],
          [1, 1] => [0],
      ));

      println!("{:?}", nn.predict(vec![0, 0]));
      println!("{:?}", nn.predict(vec![0, 1]));
      println!("{:?}", nn.predict(vec![1, 0]));
      println!("{:?}", nn.predict(vec![1, 1]));
   ```
## Complex example
   coming soon

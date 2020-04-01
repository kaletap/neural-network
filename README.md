# NNet - implementing neural networks with numpy

In this project (made for a deep learning course at Warsaw University of Technology)
I developed a simple framework for training deep neural networks consisting of a bunch of sequential layers. It's only dependency is numpy - a python library for matrix operations and tqdm for progress bars. Below I will present how it works.

## Math
The goal is to construct a function f(x), that takes a vector x and makes a best prediction of y based on data we have. In deep learning this function is a neural network.

The main building blocks of any neural networks are **layers** - the way of transforming one representation of data into the next one. There are two things we need to specify for each layer: how many neurons the output vector has and the activation function g.

At the end of computations we obtain our output y_hat. In order for the network to work, we need to specify the loss function based on a true value of y.

More details can be found here:
* Video by Andrej Karpathy: https://www.youtube.com/watch?v=i94OvYb6noo 
* Deep learning course by Andrew Ng: https://www.coursera.org/learn/neural-networks-deep-learning/lecture/6dDj7/backpropagation-intuition-optional

## API
* `NeuralNetwork` - a class representing neural network. It is initalized with:
    * input_size: int - size of an input vector
    * layers: Iterable[Layer] - List of tuples representing consecutive layers with (number of neurons, activation function)
    * loss - loss function
    
    It's main method is `fit`(x, y, [n_iter, lr]) which finds optimal parameters of the network.
    
* `nnet.activations` - includes various classes representing activation functions, deriving from base Activation. Each of them has to implement two methods: forward and backward 
* `nnet.losses` - includes various classes representing loss functions, deriving from base Loss. Each of them has to implement two methods: forward and backward

## Example 
```buildoutcfg
net = NeuralNetwork(2, [(15, Relu()), (1, Identity())], QuadraticLoss(), sd=1e-3)
net.fit(x_train, y_train, n_iter=1000, lr=0.01)
```

In this simple code we
* defined neural network that takes input of size 2, with two layers: 
hidden layer of dimensionality 15 and output layer of dimensionality 1.
To optimize the network we use squared error.
* fitted the network on input matrix x_train of size n_features x n_observations
and output matrix of size 1 x n_observations, did 1000 iterations of gradient descent 
with learning rate of 0.01

More examples (applications for regression and classification) 
can be found in `01_demo_regression.ipynb` and `02_demo_classification.ipynb` notebooks.

## Extensions
This package is easily extensible with new loss and activation functions.
For example, to define a new activation function we implement a class 
derived from base Activation that implements forward (forward pass) and backward (derivative) methods.
Here is a definition of a Relu activation function:
```buildoutcfg
class Relu(Activation):
    def forward(self, x):
        return np.where(x >= 0, x, 0)

    def backward(self, x):
        return np.where(x >= 0, 1, 0)
```
Similarly, we can defined any other activation and loss function.

## Installation
* [Optionally] Create new virtual environment with `python3 -m venv venv` and activate it with `source venv/bin/activate`.
* Install package with `pip3 install .`
* Development: run tests with `python3 -m pytest`

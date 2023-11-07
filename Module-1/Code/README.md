# The Math Behind The Neural Network

We will be dabbling with the MNIST dataset where we will have images that are 28x28 which result in 784px where each pixel ranges from 0 to 255 where 0 is black and 255 is white.  
We will represent the dataset as a matrix where each column is an example from the dataset and each row is a pixel from that example.

## Neural Network Structure

The basic structure of the neural network will be formed as the following:

Input Layer - 1st Hidden Layer - 2nd Hidden Layer - Output Layer

Where the input layer will be having 784 neurons corresponding to the pixels.  
and the 1st and 2nd hidden layer will have 16 neurons just because.
and the output layer will be having 10 neurons corresponding to each digit.

## Forward Pass

The Forward Pass (Forward Propagation) is simply when we pass a certain data and it goes through the network to compute whats the output is going to be.

A0 = X where X is a (784 x m) matrix where m is the no. of data => (its just the input layer).

Z1 = W1\*A0 + b1 (Z1 is the inactivated 1st hidden layer) W1 is (16x784) matrix and b is (16x1) matrix. => (Weights corresponds with every relation with neurons from the layer before this layer and biases are just for each neuron in this layer).

A1 = ReLU(Z1) => (Activation Function).

Z2 = W2\*A1 + b2 (Same as the notes for Z1 but notice the difference in A where in Z1 it had the Activation of the input layer, here its the Activation of the 1st hidden layer, basically we multiply by the Activation of the layer before the current layer)

A2 = Softmax(Z2) (Activation Function for Output Layer)

## Backpropagation

Its the algorithm that makes all the magic happens, where it define the way how should the cost function behave with the gradient descent. After the forward pass we would backwardly propagate the network to nudge the weights and biases to lower the cost function.

for the 2nd hidden layer:

dZ2 = A2 - y => (calculates the error where y is matrix representing the labels through one-hot encoding).

dW2 = 1/m _ dZ2 _ A1^T => (we calculate the derivative of the loss function with respect to the weights in layer 2 to find the Gradient Descent basically).

dB2 = 1/m sum(dZ2) => (for the deviations of the bias we calculate the average of the absolute error).

for the 1st hidden layer:

dZ1 = W2^T _ dZ2 _ g-prime(Z1) => (Calculates the error of 1st hidden layer, g-prime is the derivate of the activation function to get the proper error of the first layer).

dW1 = 1/m _ dZ1 _ X^T (where X is the original input).

dB1 = 1/m sum(dZ1).

## Update Weights and Biases

After the previous steps we have to update our weights and biases accordingly to enhance the accuracy.

W1 = W1 - a _ dW1
b1 = b1 - a _ dB1
W2 = W2 - a _ dW2
b2 = b2 - a _ dB2

where a is the learning rate which is a hyper-parameter that we decide.

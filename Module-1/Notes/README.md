# Notes About Module 1
Module 1 talks the basics of Artificial Neural Networks (ANNs) and how would a Neural Network learn through backpropagation.

## What is a Neuron?
Neuron is the basic fundamental unit in a Neural Network, its a representation of the human brain cell. To put it in simple terms, it's computational unit that holds a value between one and zero which is called its activation. This activation is calculated through processing input data and calculation the activation through a weighted sum and an activation function.

## What Is The Architecture of a Neural Network?
The architecture of a neural network fundamentally follows the following form.

Input Layer - Hidden Layers - Output Layer

The `Input Layer` is basically a collection of neurons that represents the input data. And its only goal is to pass the input for the hidden layer where the magic would happen.  

The `Hidden Layer` is an intermediate layer of artificial neurons that comes between the input layer and the output layer. It's called "hidden" because it doesn't directly interact with the external data or provide the final output of the network. Instead, the hidden layer's purpose is to perform complex transformations on the input data to learn and capture patterns or features in the data that are not easily discernible in the raw input.


The `Output Layer` is the final layer of artificial neurons, and its primary function is to produce the network's predictions or outputs based on the information processed by the preceding layers, including any hidden layers. The number of neurons in the output layer and their activation functions depend on the specific task the neural network is designed for.

## How Does Activation Work?
Activation functions in neural networks introduce non-linearity and help the network model complex relationships in the data, enabling it to learn and make better predictions. They act on the weighted sum of input values for each neuron and determine whether the neuron should be activated (produce an output) or not. Common activation functions include the sigmoid (which squashes values to a range of 0 to 1), the hyperbolic tangent (tanh, which squashes values to -1 to 1), and the rectified linear unit (ReLU, which outputs the input for positive values and 0 for negative values), each serving different purposes in network architectures.

## How are Weights and Biases Being Calculated?
In a neural network, the weights and biases are learned through a process called training, typically using an optimization algorithm like stochastic gradient descent (SGD). Here's a brief overview of how it works:

1. **Initialization**: Initially, the weights and biases are set to small random values. These are the parameters that the network needs to learn.
2. **Forward Pass**: During the training process, input data is passed forward through the network. Neurons calculate a weighted sum of their inputs and apply an activation function to produce an output (activation). This forward pass computes the network's predictions.
3. **Loss Calculation**: The predictions are compared to the actual (ground truth) values using a loss function, which measures how far off the predictions are from the true values.
4. **Backpropagation**: The key to training is backpropagation, which calculates gradients of the loss with respect to the weights and biases. This step quantifies how much each weight and bias contributed to the error.
5. **Update Weights and Biases**: The optimization algorithm, often SGD, uses these gradients to update the weights and biases in a direction that reduces the loss. The learning rate is a hyperparameter that controls the size of these updates.
6. **Iteration**: Steps 2-5 are repeated iteratively on batches of data for a certain number of epochs. Over time, the network's weights and biases are adjusted to minimize the loss, making the predictions more accurate.

This iterative process of forward pass, loss calculation, backpropagation, and weight/bias updates continues until the model converges to a state where the loss is minimized, and the network has learned the optimal weights and biases for the given task.

The activation values for each neuron are calculated during the forward pass as I mentioned earlier, where the weighted sum of inputs is passed through the activation function, and the result becomes the neuron's activation. This activation is used to determine the neuron's contribution to the final prediction.

## What is The Cost Function?
its a function that basically calculates the accuracy of the neural network, to tell the network how is it doing and how to improve. What we call learning is trying to improve this cost function.

## What is The Gradient Descent
It answers the question of what direction decreases the function quickly, in multivarient calculus calculating the gradient gives us the steepest increase and the negative of that gives us the gradient descent.

Which in the terms of a neural network, how we should nudge the weights and biases of each neuron to go in the direction of the gradient descent.

## How Does Backpropagation Works?
Its the algorithm that makes all the magic happens, where it define the way how should the cost function behave with the gradient descent. After the forward pass we would backwardly propagate the network to nudge the weights and biases to lower the cost function. 

Backpropagation is a crucial algorithm in the realm of machine learning, especially in the training of neural networks. It is responsible for orchestrating how the cost function interacts with the gradient descent optimization process.

After the forward pass, where input data is processed through the network to make predictions, backpropagation is employed to propagate the error backward through the network. This involves calculating the gradients of the network's parameters, such as weights and biases, with respect to the cost function. These gradients indicate how much each parameter should be adjusted to minimize the cost function, essentially nudging the weights and biases in a direction that reduces the error or loss.

By iteratively applying backpropagation and gradient descent, a neural network gradually fine-tunes its parameters to improve its performance on a specific task, ultimately making it capable of making better predictions or classifications.

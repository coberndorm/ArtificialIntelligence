import numpy as np
from activationFunctions import *
from dataManipulation import * 

class Perceptron(object):
    def __init__(self,  num_inputs: int, num_neurons: list, num_outputs: int, func_activation: list, eta = 0.01) -> None:
        """ 
        Initializes the multi-layer perceptron with the specified architecture.

        Args:
        - num_inputs (int): Number of input features.
        - num_neurons (list of int): Number of neurons in each layer.
        - num_output (int): Number of output features.
        - func_activation (list): List of activation functions for each layer.

        """

        # Set the size of the network
        n_layers = [num_inputs] + num_neurons + [num_outputs]
        self.num_layers = len(n_layers)
        self.layers = [0]*self.num_layers
        self.num_neurons = n_layers
        

        # Define all of the layers
        input = num_inputs
        for i, neuron in enumerate(n_layers):
            self.layers[i] = Layer(input, neuron, func_activation[i], eta)
            input = neuron

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the neural network.

        Args:
        - input (array): The input data for the forward pass.

        Returns:
        - output (array): The output prediction of the neural network.

        """
        input = input.reshape(-1,1)
        # Iterate through all neurons
        for i in range(self.num_layers):
            input = self.layers[i].forward(input)

        self.output = input
        return self.output
    
    def backward(self, error: np.ndarray) -> list(float):
        """
        Performs the backward pass to update the network's weights.

        Args:
        - error (array): The error in the prediction.

        Returns:
        - gradients_prom (list of float): List of average gradients for each layer.

        """
        #Initializing an array for the mean of the gradients in every layer
        gradients = [0] * self.n_layers

        # Backpropagation for output layer
        gradients[-1] = self.neurons[-1].backward(-np.sum(error), 1)
        
        # Backpropagation for all the other layers
        for i in reversed(range(self.n_layers - 1)):
            gradients[i] = self.neurons[i].backward(self.neurons[i + 1].weights, gradients[i+1])

        return gradients
    


class Layer():
    def __init__(self, num_inputs: int, num_neurons:int, activation: object, eta:float) -> None:
        """
        Initializes a neural network layer.

        Args:
        - n_inputs (int): Number of input features.
        - n_neurons (int): Number of neurons in the layer.
        - activation (Activation): Activation function for the layer.

        """

        # Initialize weights randomly (you can uncomment the random initialization below)
        #self.weights = np.random.randn(num_neurons, num_inputs) * 2 - 1
        self.weights = np.ones((num_neurons, num_inputs))

        # Initialize bias randomly (you can uncomment the random initialization below)
        #self.bias = np.random.randn(num_neurons, 1) * 2 - 1
        self.bias = np.ones((num_neurons, 1))
        self.weights = np.hstack((self.weights, self.bias))
        
        # Set Activation function
        self.activation = activation  # Create an instance of the provided activation function
        # Learning rate (you can adjust this)
        self.eta = eta

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass through the layer.

        Args:
        - input (array): The input stimuli for the layer.

        """

        # Save the input input for later use in backward pass
        self.input = input

        # Compute the initial linear combination of input and weights
        self.field = np.matmul(self.weights, np.vstack((self.input,1)))

        # Apply the activation function to the linear combination
        self.output = self.activation.forward(self.field)
        return self.output
    
    def backward(self, weights_prev, local_gradient_prev):
        """
        Performs backpropagation for the layer.

        Args:
        - weights_prev (array): Weights from the next layer.
        - local_gradient_prev (array): Local gradient from the next layer.

        Returns:
        - local_gradient (array): Local gradient for this layer.

        """

        phi_prime = self.activation.backward(self.field)  # Compute the derivative of the activation function

        # Compute local gradient for this layer using chain rule and weights from the next layer
        local_gradient = np.multiply(np.vstack((phi_prime,1)), np.dot(weights_prev.T, local_gradient_prev))

        # Compute weight update using the local gradient and input stimuli
        delta = np.dot(local_gradient, self.stimuli) # Weight and bias change
        assert delta.shape == self.weights.shape

        # Update weights using the learning rate and calculated delta
        weights_new = self.weights + self.eta * delta
        self.weights = weights_new

        return local_gradient
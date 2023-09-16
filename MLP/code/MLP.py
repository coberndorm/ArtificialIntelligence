import numpy as np
from activationFunctions import *
from dataManipulation import * 

class perceptron:
    """
    A simple implementation of a multi-layer perceptron (MLP) neural network.

    Attributes:
    - n_layers (int): Number of layers in the neural network.
    - n_neurons (list of int): Number of neurons in each layer.
    - neurons (list of Layer): List containing the layers of the neural network.
    - input (array): The input data for the forward pass.
    - output (array): The output prediction of the neural network.

    Methods:
    - __init__(self, n_inputs, n_neurons, n_activation): Initializes the perceptron.
    - forward(self, input): Performs the forward pass through the neural network.
    - backward(self, error): Performs the backward pass to update the network's weights.
    - train(self, x, Y, epochs): Trains the neural network on the given data.

    """

    def __init__(self, n_inputs, n_neurons, n_outputs, n_activation, eta = 1):
        """
        Initializes the multi-layer perceptron with the specified architecture.

        Args:
        - n_inputs (int): Number of input features.
        - n_neurons (list of int): Number of neurons in each layer.
        - n_activation (list): List of activation functions for each layer.

        """

        # Set the size of the network
        n_layers = [n_inputs] + n_neurons + [n_outputs]
        self.n_layers = len(n_layers)

        self.n_neurons = n_layers

        # Define an array to hold all the layers
        self.neurons = [Layer(n_inputs, n_inputs, n_activation[0], eta)]

        # Define all hidden layers
        for i in range(0, self.n_layers-1):
            self.neurons.append(Layer(n_layers[i], n_layers[i+1], n_activation[i+1], eta))
        
        #self.neurons.append(Layer(n_outputs, n_neurons[-1], n_activation[-1], eta))

    def forward(self, input):
        """
        Performs the forward pass through the neural network.

        Args:
        - input (array): The input data for the forward pass.

        Returns:
        - output (array): The output prediction of the neural network.

        """

        # Save the input
        self.input = np.array(input)
        # Pass the input to the first layer
        output = self.neurons[0].forward(np.transpose(input))

        # Iterate through all neurons
        for i in range(1, self.n_layers):
            output = self.neurons[i].forward(output)

        # Set the MLP's output as the output of the last layer
        self.output = output
        return self.output

    def backward(self, error):
        """
        Performs the backward pass to update the network's weights.

        Args:
        - error (array): The error in the prediction.

        Returns:
        - gradients_prom (list of float): List of average gradients for each layer.

        """
        #Initializing an array for the mean of the gradients in every layer
        gradients_mean = [0] * self.n_layers

        # Backpropagation for output layer
        local_gradient_prev = self.neurons[-1].backward(np.sum(error), 1)
        gradients_mean[-1] = np.mean(local_gradient_prev)

        # Backpropagation for all the other layers
        for i in reversed(range(self.n_layers - 1)):
            local_gradient_prev = self.neurons[i].backward(self.neurons[i + 1].weights, local_gradient_prev)
            gradients_mean[i] = np.mean(local_gradient_prev)

        return gradients_mean


    def train(self, data, epochs):
        """
        Trains the neural network on the given data using backpropagation.

        Args:
        - data (array): Input data with the last value the target column.
        - epochs (int): Number of training epochs.

        Returns:
        - gradients (list): List of gradients computed during training.

        """

        #This part of the code cannot yet withstand multiple outputs
        # Defining the train and test set
        train, test, _ = train_test_val(data, (75,25,0))
        train_x, train_y = train[:,0:-1], train[:,-1]
        test_x, test_y = test[:,0:-1], test[:,-1]

        # Check Y dimensions are of the size of the output layer
        #assert len(train_y[0]) == self.n_neurons[-1]

        # Define auxiliary variables
        data_len = len(train_x)
        gradients = [0] * data_len * epochs;  gradient_epochs = [0]*epochs
        instant_energy_train = [0] * data_len * epochs;  instant_average_energy_train = [0]*epochs
        instant_average_energy_test = [0]*epochs
        counter = 0

        for epoch in range(epochs):
            for i, input in enumerate(train_x):
              y_pred = self.forward(np.array([input]))
              error = np.array(train_y[i] - y_pred)
              instant_energy_train[counter] = np.sum(error**2)/2
              gradients[counter] = self.backward(error)
              counter += 1

            # Epoch relevant information
            gradient_epochs[epoch] = np.mean(gradients[epoch*data_len:epoch*data_len+data_len], axis = 0)
            instant_average_energy_train[epoch] = np.mean(instant_energy_train[epoch*data_len:epoch*data_len+data_len])

            #Test error
            error_test = test_y - self.forward(test_x)
            instant_average_energy_test[epoch] = np.mean(error_test**2)/2
            if instant_average_energy_test[epoch] < 0.02:
              break

            #print(error_p.shape, test_y.shape, self.forward(np.array(test_x)).shape)

        return gradients, gradient_epochs, instant_energy_train, instant_average_energy_train, instant_average_energy_test
    
    def train_batch(self, data, epochs=50, iterations=10):
        """
        Train the neural network using mini-batch gradient descent.

        Args:
        - data (array): Input data with the last value as the target column.
        - epochs (int): Number of training epochs.
        - iterations (int): Number of mini-batch iterations per epoch.

        Returns:
        - gradients (list): List of gradients computed during training.
        - gradient_epochs (list): List of average gradients for each epoch.
        - instant_energy_train (list): List of instant energy values during training.
        - instant_average_energy_train (list): List of average energy values during training.
        - instant_average_energy_test (list): List of average energy values during testing.

        """
        # Split data into training and testing sets
        train, test, _ = train_test_val(data, (75, 25, 0))

        # Initialize counters and auxiliary variables
        counter = 0
        data_len = len(train)
        gradients = [0] * iterations * epochs;   gradient_epochs = [0] * iterations
        instant_energy_train = [0] * epochs * iterations;  instant_average_energy_train = [0] * iterations
        instant_average_energy_test = [0] * epochs * iterations

        
        # Split the training data into input and target
        train_x, train_y = train[:, 0:-1], train[:, -1]
        test_x, test_y = test[:, 0:-1], test[:, -1]

        for iter in range(iterations):

            for epoch in range(epochs):
                # Forward pass for the entire training set
                y_pred = self.forward(train_x)
                error = train_y - y_pred

                # Find the index with the largest error
                idx = np.argmax(error[0])

                # Compute instant energy and gradients for the mini-batch
                instant_energy_train[counter] = np.mean(error**2) / 2
                self.forward(np.array([train_x[idx, :]]))
                gradients[counter] = self.backward(error[0][idx])

                # Test error
                error_test = test_y - self.forward(test_x)
                instant_average_energy_test[counter] = np.mean(error_test**2) / 2

                # Early stopping if test error is below a threshold
                if instant_average_energy_test[counter] < 0.02:
                    print("Paroooo")
                    counter += epochs-epoch-1
                    break
                elif instant_average_energy_test[counter] > instant_energy_train[counter] +0.5 :
                    # If the testing error is considerably higher than the training one, resample
                    print("Resampleooo")
                    train_x, train_y = train[:, 0:-1], train[:, -1]
                    test_x, test_y = test[:, 0:-1], test[:, -1]
                counter += 1

            # Compute average gradient and average energy for the current iteration
            gradient_epochs[iter] = np.mean(gradients[iter * epochs:iter * epochs + epoch +1], axis=0)
            instant_average_energy_train[iter] = np.mean(instant_energy_train[iter * epochs:iter * epochs + epoch+1])

        return gradients, gradient_epochs, instant_energy_train, instant_average_energy_train, instant_average_energy_test
    

class Layer:
    """
    A class representing a single layer in a neural network.

    Attributes:
    - weights (array): Weight matrix for the layer's connections.
    - activation (Activation): The activation function for the layer.
    - stimuli (array): Input stimuli for the layer.
    - field (array): Linear combination of stimuli and weights.
    - output (array): Output of the layer after activation.
    - local_gradient (array): Local gradient used in backpropagation.

    Methods:
    - __init__(self, n_inputs, n_neurons, activation): Initializes the layer.
    - forward(self, stimuli): Performs the forward pass through the layer.
    - backward(self, weights_prev, local_gradient_prev): Performs backpropagation for the layer.

    """

    def __init__(self, n_inputs, n_neurons, activation, eta = 1):
        """
        Initializes a neural network layer.

        Args:
        - n_inputs (int): Number of input features.
        - n_neurons (int): Number of neurons in the layer.
        - activation (Activation): Activation function for the layer.

        """

        # Initialize weights with ones (you can uncomment the random initialization below)
        self.weights = np.random.randn(n_neurons, n_inputs) * 2 - 1
        #self.weights = np.ones((n_neurons, n_inputs))
        self.activation = activation()  # Create an instance of the provided activation function
        # Learning rate (you can adjust this)
        self.eta = eta

    def forward(self, stimuli):
        """
        Performs the forward pass through the layer.

        Args:
        - stimuli (array): The input stimuli for the layer.

        """

        # Save the input stimuli for later use in backward pass
        self.stimuli = stimuli

        # Compute the initial linear combination of stimuli and weights
        self.field = np.matmul(self.weights, self.stimuli)

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
        self.local_gradient = np.multiply(phi_prime, np.dot(weights_prev.T, local_gradient_prev))

        # Compute weight update using the local gradient and input stimuli
        # print(phi_prime.shape, self.local_gradient.shape)
        delta = np.dot(self.local_gradient, self.stimuli.T) # Weight change
        assert delta.shape == self.weights.shape

        # Update weights using the learning rate and calculated delta
        weights_new = self.weights + self.eta * delta
        self.weights = weights_new

        return self.local_gradient
import numpy as np

# Define a class for the sigmoid activation function.
class sigmoid:
    """
    Sigmoid Activation Function

    This class implements the sigmoid activation function and its derivative.

    Methods:
    - forward(input): Compute the sigmoid activation of the input.
    - backward(output): Compute the derivative of the sigmoid function.

    Attributes:
    - output: The output of the sigmoid function during the forward pass.
    - back: The derivative of the sigmoid function during the backward pass.
    """

    def __init__(self, a=1) -> None:
        self.a = a

    def forward(self, input):
        """Compute the sigmoid activation of the input."""
        self.output = 1 / (1 + np.exp(-self.a *input))
        return self.output

    def backward(self, output):
        """Compute the derivative of the sigmoid function."""
        self.back = self.a*np.exp(-self.a*output) / (np.exp(-self.a*output) + 1) ** 2
        return self.back

# Define a class for the hyperbolic tangent (tanh) activation function.
class tanh:
    """
    Hyperbolic Tangent (tanh) Activation Function

    This class implements the hyperbolic tangent (tanh) activation function
    and its derivative.

    Methods:
    - forward(input): Compute the tanh activation of the input.
    - backward(output): Compute the derivative of the tanh function.

    Attributes:
    - back: The derivative of the tanh function during the backward pass.
    """
    def __init__(self,a=1) -> None:
        self.a = a

    def forward(self, input):
        """Compute the hyperbolic tangent (tanh) activation of the input."""
        return np.tanh(self.a*input)

    def backward(self, output):
        """Compute the derivative of the tanh function."""
        self.back = self.a*(1 - np.tanh(self.a*output) ** 2)
        return self.back

# Define a class for a linear activation function.
class linear:
    """
    Linear Activation Function

    This class implements a linear activation function and its gradient.

    Methods:
    - forward(input): Perform a linear transformation on the input.
    - backward(output): Compute the gradient of the linear transformation.

    Attributes:
    - a: Coefficient for the linear transformation.
    - b: Bias for the linear transformation.
    """
    def __init__(self,a=1,b=0) -> None:
        self.a = a  # Coefficient for linear transformation.
        self.b = b  # Bias for linear transformation.

    def forward(self, input):
        """Perform a linear transformation on the input."""
        return np.dot(self.a, input) + self.b

    def backward(self, output):
        """Compute the gradient of the linear transformation."""
        length = int(len(output))
        return np.array([[self.a]] * length)  # Gradient is constant for all inputs.

import numpy as np

class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation Function

    This class implements the ReLU activation function and its derivative.

    Methods:
    - forward(input): Compute the ReLU activation of the input.
    - backward(output): Compute the derivative of the ReLU function.

    Attributes:
    - back: The derivative of the ReLU function during the backward pass.
    """
    def __init__(self, a=1) -> None:
        self.a = a
        pass

    def forward(self, input):
        """Compute the Rectified Linear Unit (ReLU) activation of the input."""
        return np.maximum(0, self.a*input)

    def backward(self, output):
        """Compute the derivative of the ReLU function."""
        return np.where(output > 0, self.a, 0)

import numpy as np
from typing import Tuple, List, Union, Optional

class MLPTwoLayers:

    # DO NOT adjust the constructor params
    def __init__(self, input_size: int = 3072, hidden_size: int = 100, output_size: int = 10, learning_rate: float = 1e-3):
        """
        Initialize a two-layer MLP (Multi-Layer Perceptron).

        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the hidden layer
            output_size: Number of output classes
            learning_rate: Learning rate for gradient descent
            
        Note:
            Weights are initialized using Xavier/Glorot initialization:
            W ~ N(0, sqrt(2/(n_in + n_out)))
            where n_in and n_out are the number of input and output units.
            This maintains variance of activations and gradients across layers.
            Biases are initialized to zeros.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.w1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / (input_size + hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / (hidden_size + output_size))
        self.b1 = np.zeros(hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            features: Input features of shape (batch_size, input_size)

        Returns:
            Softmax probabilities of shape (batch_size, output_size)

        Note:
            We use a numerical stability trick for softmax by subtracting the maximum value
            from each row before applying exp(). This prevents potential overflow issues when
            exponentiating large numbers, which could lead to NaN or Inf values. The subtraction
            doesn't change the final softmax probabilities because the exponential function's
            relative proportions remain the same, but it ensures the calculations stay within
            a numerically stable range.
        """
        self.features = features  # Store input features for use in backpropagation
        self.a1 = 1 / (1 + np.exp(-(np.dot(features, self.w1) + self.b1)))  # Hidden layer activations (sigmoid)
        z2 = np.dot(self.a1, self.w2) + self.b2  # Output layer pre-activation
        z2 -= np.max(z2, axis=1, keepdims=True)  # Numerical stability for softmax
        np.exp(z2, out=z2)
        self.probs = z2 / np.sum(z2, axis=1, keepdims=True)
        return self.probs

    def loss(self, predictions: np.ndarray, label: np.ndarray) -> float:
        """
        Calculate the cross-entropy loss.

        Args:
            predictions: Softmax probabilities from forward pass of shape (batch_size, output_size)
            label: Ground truth labels of shape (batch_size,)
            
        Returns:
            Average cross-entropy loss across the batch

        Note:
            We average the loss across all samples in the batch rather than using the sum.
            This makes the loss value independent of batch size, stabilizes gradient magnitudes, 
            and keeps the learning rate consistent regardless of batch size.
        """
        num_samples = predictions.shape[0]
        correct_log_probs = -np.log(predictions[range(num_samples), label])
        data_loss = np.sum(correct_log_probs)
        return data_loss / num_samples

    def backward(self, label: np.ndarray) -> None:
        """
        Backward pass to update weights and biases using gradient descent.

        Args:
            label: Ground truth labels of shape (batch_size,) with values 0, 1, or 2

        Note:
            Mathematical explanation of gradient descent algorithm:
            1. Compute gradients of loss with respect to all parameters.
            2. For softmax with cross-entropy loss, gradient is (y_hat - y) / batch_size.
            3. Backpropagation:
               - dL/dW2 = a1^T * dL/dz2
               - dL/db2 = sum(dL/dz2)
               - dL/da1 = dL/dz2 * W2^T
               - dL/dz1 = dL/da1 * a1 * (1-a1)  [sigmoid derivative]
               - dL/dW1 = x^T * dL/dz1
               - dL/db1 = sum(dL/dz1)
            4. Parameter update: θ = θ - learning_rate * dL/dθ
        """
        num_samples = self.probs.shape[0]

        dz2 = self.probs.copy()
        dz2[range(num_samples), label] -= 1  # y_hat - y, where y is index-encoded
        dz2 /= num_samples

        dw2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)

        da1 = np.dot(dz2, self.w2.T)
        da1 *= self.a1 * (1 - self.a1)  # Sigmoid derivative

        dw1 = np.dot(self.features.T, da1)
        db1 = np.sum(da1, axis=0)

        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2


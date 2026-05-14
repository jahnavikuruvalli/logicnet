# network/neural_net.py
#
# A neural network built completely from scratch using only numpy.
#
# WHAT IS A NEURAL NETWORK?
# It's a function that takes some inputs and produces an output.
# It has internal numbers called "weights" that it adjusts over time
# until its outputs match the ones you want.
#
# STRUCTURE:
# Input layer → one or more hidden layers → output layer
#
# Each layer is a row of "neurons".
# Each neuron receives values from the previous layer,
# multiplies each by a weight, sums them up, and passes the
# result through an activation function.
#
# TRAINING:
# 1. Forward pass  — feed inputs in, get a prediction out
# 2. Compare       — how wrong was the prediction? (loss)
# 3. Backward pass — figure out which weights caused the error
# 4. Update        — nudge every weight slightly to reduce the error
# Repeat thousands of times.


import numpy as np


# ── Activation functions ───────────────────────────────────────────
#
# An activation function decides how much a neuron "fires".
# Without it, the network is just linear math and can't learn
# complex patterns like XOR.
#
# We use sigmoid: it squashes any number into the range (0, 1).
# Formula: σ(x) = 1 / (1 + e^(-x))

def sigmoid(x):
    """Squash x into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """
    The derivative of sigmoid — used during backpropagation.
    Tells us how much the sigmoid output changes when x changes.
    Formula: σ(x) * (1 - σ(x))
    We pass in the already-computed sigmoid output to save work.
    """
    return x * (1.0 - x)


# ── The NeuralNetwork class ────────────────────────────────────────

class NeuralNetwork:
    """
    A fully connected feedforward neural network.

    Example: NeuralNetwork([2, 4, 1])
    means: 2 input neurons, one hidden layer of 4, 1 output neuron.
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes : list of ints, e.g. [2, 4, 1]
                      First value  = number of inputs
                      Middle values = hidden layer sizes
                      Last value   = number of outputs
        """
        self.layer_sizes = layer_sizes
        self.num_layers  = len(layer_sizes)

        # Learning rate — how big a step we take when adjusting weights.
        # Too high → overshoots, never settles.
        # Too low  → learns very slowly.
        self.learning_rate = 0.1

        # Build weights and biases
        self._init_weights()

        # These are filled during forward/backward pass
        # and read by the visualizer to animate what's happening
        self.activations  = []   # output of each layer after sigmoid
        self.weighted_sums = []  # raw sums before sigmoid (called 'z')
        self.deltas       = []   # error signals during backprop

        # Training history
        self.loss_history = []   # loss value after each training step

    def _init_weights(self):
        """
        Create random starting weights and zero biases.

        self.weights[i] is a 2D matrix connecting layer i to layer i+1.
        Shape: (size of next layer, size of current layer)

        self.biases[i] is a column vector for layer i+1.
        Shape: (size of next layer, 1)

        We use small random values (scaled by layer size) so the
        network doesn't saturate the sigmoid immediately.
        """
        self.weights = []
        self.biases  = []

        for i in range(self.num_layers - 1):
            n_in  = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]

            # Xavier initialization — keeps values from exploding
            scale = np.sqrt(2.0 / n_in)
            w = np.random.randn(n_out, n_in) * scale
            b = np.zeros((n_out, 1))

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, x):
        """
        Run one forward pass through the network.

        x : input as a numpy array, shape (n_inputs, 1)
            e.g. [[0], [1]] for a 2-input network

        Returns the output — a float between 0 and 1.

        Also stores intermediate values in self.activations
        so the visualizer can animate them and backprop can use them.
        """
        self.activations  = []
        self.weighted_sums = []

        current = x
        self.activations.append(current)  # layer 0 = raw input

        for w, b in zip(self.weights, self.biases):
            # Weighted sum: z = W · current + b
            z = np.dot(w, current) + b
            self.weighted_sums.append(z)

            # Activation: a = sigmoid(z)
            a = sigmoid(z)
            self.activations.append(a)
            current = a

        # Final activation is the network's prediction
        return current

    def compute_loss(self, prediction, target):
        """
        Mean squared error loss.
        Measures how wrong the prediction is.

        prediction : network output, float in (0, 1)
        target     : correct answer, 0 or 1

        Loss = (prediction - target)²
        The closer to 0, the better.
        """
        return float(np.mean((prediction - target) ** 2))

    def backward(self, target):
        """
        Backpropagation — compute gradients and update weights.

        target : the correct answer (0 or 1) as a numpy array

        This works by:
        1. Computing the error at the output layer
        2. Propagating that error backward through each layer
        3. Using the error at each layer to nudge the weights

        The key formula at each layer:
            delta = error * sigmoid_derivative(activation)
            weight_update = learning_rate * delta · prev_activation.T
        """
        self.deltas = [None] * (self.num_layers - 1)

        # ── Output layer error ──────────────────────────────────────
        # How wrong was the final output?
        output     = self.activations[-1]
        error      = output - target
        self.deltas[-1] = error * sigmoid_derivative(output)

        # ── Hidden layer errors (going backward) ───────────────────
        # Each hidden layer's error depends on the next layer's error
        # passed back through the weights
        for i in range(self.num_layers - 3, -1, -1):
            next_delta = self.deltas[i + 1]
            w_next     = self.weights[i + 1]

            # Error flowing back: W.T · delta
            error_back = np.dot(w_next.T, next_delta)
            activation = self.activations[i + 1]
            self.deltas[i] = error_back * sigmoid_derivative(activation)

        # ── Update weights and biases ───────────────────────────────
        for i in range(len(self.weights)):
            prev_activation = self.activations[i]
            delta           = self.deltas[i]

            # Gradient descent step:
            # new_weight = old_weight - learning_rate * delta · prev.T
            self.weights[i] -= self.learning_rate * np.dot(delta, prev_activation.T)
            self.biases[i]  -= self.learning_rate * delta

    def train_step(self, x, target):
        """
        Run one complete training step on a single example.

        x      : input, e.g. np.array([[0],[1]])
        target : correct output, e.g. np.array([[1]])

        Returns the loss for this step.
        """
        prediction = self.forward(x)
        loss       = self.compute_loss(prediction, target)
        self.backward(np.array([[target]]))
        self.loss_history.append(loss)
        return loss

    def predict(self, x):
        """
        Run a forward pass and return a clean 0 or 1.
        (rounds the sigmoid output to the nearest integer)
        """
        output = self.forward(x)
        return int(np.round(float(output)))

    def reset(self):
        """Re-randomize all weights — start training from scratch."""
        self._init_weights()
        self.activations   = []
        self.weighted_sums = []
        self.deltas        = []
        self.loss_history  = []

    def get_weight_snapshot(self):
        """
        Return a copy of all current weights and biases.
        Used by history.py to record state for the replay scrubber.
        """
        return {
            "weights": [w.copy() for w in self.weights],
            "biases":  [b.copy() for b in self.biases],
        }

    def load_weight_snapshot(self, snapshot):
        """Restore weights and biases from a saved snapshot."""
        self.weights = [w.copy() for w in snapshot["weights"]]
        self.biases  = [b.copy() for b in snapshot["biases"]]
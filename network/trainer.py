# network/trainer.py
#
# The Trainer sits between the circuit and the neural network.
#
# It takes the truth table generated from the circuit and uses it
# to train the neural network — feeding examples in one at a time,
# recording progress, and controlling the pace of training.
#
# WHY ONE STEP AT A TIME?
# Because we want to animate every single weight update live.
# If we trained all at once the network would just teleport to
# a solution. Stepping lets us watch it think.


import numpy as np
from network.neural_net import NeuralNetwork
from network.history import WeightHistory


class Trainer:
    """
    Controls the training process.

    Given a truth table, it trains a NeuralNetwork step by step,
    recording history and tracking accuracy.
    """

    def __init__(self, layer_sizes=(2, 4, 1)):
        """
        layer_sizes : architecture of the neural network
                      default (2, 4, 1) = 2 inputs, 4 hidden, 1 output
        """
        self.net     = NeuralNetwork(list(layer_sizes))
        self.history = WeightHistory()

        self.truth_table  = []    # list of {"inputs": [...], "output": 0/1}
        self.is_training  = False # whether training is currently running
        self.is_complete  = False # whether network has fully learned the circuit

        # Which row of the truth table we'll train on next
        self.current_row_index = 0

        # How many training steps have been completed
        self.step_count = 0

        # Steps per frame — increase this to train faster
        self.steps_per_frame = 1

        # Accuracy tracking
        self.current_accuracy = 0.0   # 0.0 to 1.0

        # The last input/output we trained on (for the visualizer)
        self.last_input  = None
        self.last_target = None
        self.last_loss   = None

    def load_truth_table(self, truth_table):
        """
        Give the trainer a truth table to learn from.

        truth_table : list of rows from generate_truth_table()
        Each row: {"inputs": [0, 1], "output": 1}

        Also resets the network and history so training
        starts fresh every time you load a new circuit.
        """
        # Filter out rows where output is None (circuit not fully connected)
        self.truth_table = [r for r in truth_table if r["output"] is not None]

        if not self.truth_table:
            return

        # Rebuild the network with the right number of inputs
        n_inputs = len(self.truth_table[0]["inputs"])
        sizes    = list(self.net.layer_sizes)
        sizes[0] = n_inputs   # match input layer to circuit input count
        self.net = NeuralNetwork(sizes)

        self.history.clear()
        self.is_training  = False
        self.is_complete  = False
        self.step_count   = 0
        self.current_row_index = 0
        self.current_accuracy  = 0.0
        self.last_input  = None
        self.last_target = None
        self.last_loss   = None

    def start(self):
        """Begin training."""
        if self.truth_table:
            self.is_training = True
            self.is_complete = False

    def pause(self):
        """Pause training without resetting anything."""
        self.is_training = False

    def reset(self):
        """Stop training and randomize all weights from scratch."""
        self.is_training = False
        self.is_complete = False
        self.step_count  = 0
        self.current_row_index = 0
        self.current_accuracy  = 0.0
        self.last_input  = None
        self.last_target = None
        self.last_loss   = None
        self.net.reset()
        self.history.clear()

    def step(self):
        """
        Run one training step.

        Picks the next row from the truth table (cycling through),
        converts it to numpy arrays, runs train_step on the network,
        records the weight snapshot, and updates accuracy.

        Returns the loss for this step, or None if not ready.
        """
        if not self.truth_table or not self.is_training:
            return None

        # Pick the current row (cycle through all rows repeatedly)
        row = self.truth_table[self.current_row_index]
        self.current_row_index = (self.current_row_index + 1) % len(self.truth_table)

        # Convert inputs list to a numpy column vector
        # e.g. [0, 1] becomes np.array([[0], [1]])
        x = np.array(row["inputs"], dtype=float).reshape(-1, 1)

        # Target is a single value wrapped in a 2D array
        target = float(row["output"])

        # Train the network on this one example
        loss = self.net.train_step(x, target)

        # Record weight state for the replay scrubber
        self.history.record(self.net.get_weight_snapshot())

        # Store for the visualizer to read
        self.last_input  = x
        self.last_target = target
        self.last_loss   = loss

        self.step_count += 1

        # Update accuracy every time we complete a full cycle
        # through the truth table
        if self.current_row_index == 0:
            self._update_accuracy()
            if self.current_accuracy == 1.0:
                self.is_complete = True
                self.is_training = False

        return loss

    def update(self):
        """
        Run one frame's worth of training steps.

        Call this every frame from the main game loop.
        steps_per_frame controls how many steps happen per frame.
        """
        if not self.is_training:
            return

        for _ in range(self.steps_per_frame):
            self.step()
            if self.is_complete:
                break

    def _update_accuracy(self):
        """
        Check the network's predictions against every truth table row
        and compute what fraction it gets right.

        Accuracy = correct predictions / total rows
        """
        if not self.truth_table:
            self.current_accuracy = 0.0
            return

        correct = 0
        for row in self.truth_table:
            x = np.array(row["inputs"], dtype=float).reshape(-1, 1)
            prediction = self.net.predict(x)
            if prediction == row["output"]:
                correct += 1

        self.current_accuracy = correct / len(self.truth_table)

    def get_all_predictions(self):
        """
        Return the network's current prediction for every truth table row.

        Used by the truth table panel to color rows red or green.

        Returns a list of dicts:
        [
            {
                "inputs": [0, 1],
                "expected": 1,
                "predicted": 0,
                "correct": False
            },
            ...
        ]
        """
        results = []
        for row in self.truth_table:
            x = np.array(row["inputs"], dtype=float).reshape(-1, 1)
            predicted = self.net.predict(x)
            results.append({
                "inputs":    row["inputs"],
                "expected":  row["output"],
                "predicted": predicted,
                "correct":   predicted == row["output"]
            })
        return results

    def set_architecture(self, hidden_layers, neurons_per_layer):
        """
        Change the network architecture before training.

        hidden_layers     : how many hidden layers (1, 2, or 3)
        neurons_per_layer : how many neurons in each hidden layer

        Resets everything — call this before starting training.
        """
        if not self.truth_table:
            return

        n_inputs = len(self.truth_table[0]["inputs"])

        # Build layer sizes: input → hidden... → output
        sizes = [n_inputs]
        for _ in range(hidden_layers):
            sizes.append(neurons_per_layer)
        sizes.append(1)  # always 1 output

        self.net = NeuralNetwork(sizes)
        self.reset()
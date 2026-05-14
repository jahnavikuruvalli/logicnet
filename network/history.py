# network/history.py
#
# WeightHistory records a snapshot of all weights and biases
# after every single training step.
#
# Think of it like a DVR for the neural network's memory.
# Every time the network updates its weights, we take a photo.
# The replay scrubber just loads old photos.
#
# MEMORY NOTE:
# For a small network (2,4,1) training 10,000 steps,
# each snapshot is tiny (~10 numbers). Total memory is fine.
# For very large networks or very long training runs this
# would need a smarter approach, but for our purposes it's perfect.


class WeightHistory:
    """
    Records weight snapshots at every training step.

    A snapshot is whatever get_weight_snapshot() returns from
    NeuralNetwork — a dict of numpy arrays for weights and biases.
    """

    def __init__(self):
        self.snapshots = []       # list of snapshots, one per training step
        self.max_snapshots = 50000  # safety cap so we don't run out of memory

    def record(self, snapshot):
        """
        Save a new snapshot.

        snapshot : dict returned by NeuralNetwork.get_weight_snapshot()
                   {"weights": [...], "biases": [...]}

        If we hit the cap, we drop the oldest half of history.
        This keeps memory bounded while preserving recent history.
        """
        self.snapshots.append(snapshot)

        if len(self.snapshots) > self.max_snapshots:
            # Keep only the most recent half
            half = self.max_snapshots // 2
            self.snapshots = self.snapshots[-half:]

    def get(self, index):
        """
        Retrieve the snapshot at a given index.

        index : int, 0 = first recorded state, -1 = most recent
        Returns None if history is empty or index is out of range.
        """
        if not self.snapshots:
            return None

        # Clamp index to valid range
        index = max(0, min(index, len(self.snapshots) - 1))
        return self.snapshots[index]

    def get_at_fraction(self, fraction):
        """
        Retrieve a snapshot by position as a fraction of total history.

        fraction : float from 0.0 to 1.0
                   0.0 = very beginning of training (random weights)
                   0.5 = halfway through training
                   1.0 = most recent state

        This is what the scrubber uses — it passes a 0.0–1.0 value
        based on where the user dragged the slider.
        """
        if not self.snapshots:
            return None

        index = int(fraction * (len(self.snapshots) - 1))
        return self.get(index)

    def clear(self):
        """Wipe all recorded history — called when training resets."""
        self.snapshots = []

    def length(self):
        """How many snapshots are currently stored."""
        return len(self.snapshots)

    def is_empty(self):
        """True if no snapshots have been recorded yet."""
        return len(self.snapshots) == 0

    def get_fraction_for_index(self, index):
        """
        The reverse of get_at_fraction.
        Given a snapshot index, return what fraction of training it is.

        Useful for positioning the scrubber handle when training is live.
        Returns 1.0 (far right) when training is running normally.
        """
        if not self.snapshots:
            return 0.0

        return index / max(1, len(self.snapshots) - 1)
# test_network.py
#
# Tests the neural network layer in isolation.
# No visuals — just math checks in the terminal.
#
# Run with: python3 test_network.py

import numpy as np
from network.neural_net import NeuralNetwork
from network.trainer import Trainer
from network.history import WeightHistory

# ── Test 1: Forward pass produces output in (0, 1) ────────────────
print("=== Test 1: Forward pass ===")

net = NeuralNetwork([2, 4, 1])
x = np.array([[0], [1]])
output = net.forward(x)
print(f"Input: [0, 1]")
output_val = float(output.flatten()[0])
print(f"Output: {output_val:.4f}  (should be between 0 and 1)")
assert 0 < output_val < 1, "Output should be between 0 and 1"
print("✅ Passed\n")

# ── Test 2: Loss decreases over many training steps ───────────────
print("=== Test 2: Network learns AND gate ===")

net2 = NeuralNetwork([2, 4, 1])

# AND gate truth table
and_table = [
    {"inputs": [0, 0], "output": 0},
    {"inputs": [0, 1], "output": 0},
    {"inputs": [1, 0], "output": 0},
    {"inputs": [1, 1], "output": 1},
]

# Train for 2000 steps
for _ in range(2000):
    for row in and_table:
        x = np.array(row["inputs"], dtype=float).reshape(-1, 1)
        net2.train_step(x, float(row["output"]))

# Check predictions
print("AND gate predictions after 2000 steps:")
all_correct = True
for row in and_table:
    x = np.array(row["inputs"], dtype=float).reshape(-1, 1)
    pred = net2.predict(x)
    correct = pred == row["output"]
    if not correct:
        all_correct = False
    status = "✅" if correct else "❌"
    print(f"  {row['inputs']} → predicted {pred}, expected {row['output']} {status}")

print()

# ── Test 3: Trainer with truth table ─────────────────────────────
print("=== Test 3: Trainer learns XOR gate ===")

xor_table = [
    {"inputs": [0, 0], "output": 0},
    {"inputs": [0, 1], "output": 1},
    {"inputs": [1, 0], "output": 1},
    {"inputs": [1, 1], "output": 0},
]

trainer = Trainer(layer_sizes=(2, 6, 1))
trainer.load_truth_table(xor_table)
trainer.start()

# Run 5000 steps
for _ in range(5000):
    trainer.step()

print(f"Accuracy after 5000 steps: {trainer.current_accuracy * 100:.0f}%")
predictions = trainer.get_all_predictions()
for p in predictions:
    status = "✅" if p["correct"] else "❌"
    print(f"  {p['inputs']} → predicted {p['predicted']}, expected {p['expected']} {status}")

print()

# ── Test 4: History recording ─────────────────────────────────────
print("=== Test 4: Weight history ===")

history = WeightHistory()
print(f"Empty history length: {history.length()}  (should be 0)")
assert history.is_empty()

net3 = NeuralNetwork([2, 4, 1])
for _ in range(10):
    x = np.array([[1], [0]])
    net3.train_step(x, 1.0)
    history.record(net3.get_weight_snapshot())

print(f"After 10 steps, history length: {history.length()}  (should be 10)")
assert history.length() == 10

snap_start = history.get_at_fraction(0.0)
snap_end   = history.get_at_fraction(1.0)
print(f"Got start snapshot: {snap_start is not None}")
print(f"Got end snapshot:   {snap_end is not None}")
assert snap_start is not None
assert snap_end is not None
print("✅ Passed\n")

print("✅ All network tests passed!")
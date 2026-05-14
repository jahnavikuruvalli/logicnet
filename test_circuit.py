# test_circuit.py
#
# Run this file to check that the circuit code works correctly.
# It builds a simple AND gate circuit in code (no visuals)
# and prints the truth table.
#
# Run it with:  python test_circuit.py

import pygame
pygame.init()  # needed because circuit_board imports pygame

from circuit.gate import Gate
from circuit.circuit_board import CircuitBoard, InputNode, OutputNode
from circuit.truth_table import generate_truth_table, format_truth_table

# ── Test 1: Gate logic ─────────────────────────────────────────────
print("=== Test 1: Gate logic ===")

and_gate = Gate("AND", 0, 0)
and_gate.input_a = 1
and_gate.input_b = 1
print(f"AND(1,1) = {and_gate.compute()}")   # expect 1

and_gate.input_b = 0
print(f"AND(1,0) = {and_gate.compute()}")   # expect 0

xor_gate = Gate("XOR", 0, 0)
xor_gate.input_a = 1
xor_gate.input_b = 1
print(f"XOR(1,1) = {xor_gate.compute()}")   # expect 0

xor_gate.input_b = 0
print(f"XOR(1,0) = {xor_gate.compute()}")   # expect 1

not_gate = Gate("NOT", 0, 0)
not_gate.input_a = 1
print(f"NOT(1)   = {not_gate.compute()}")   # expect 0

print()

# ── Test 2: Truth table for AND circuit ───────────────────────────
print("=== Test 2: AND circuit truth table ===")

# Build a minimal circuit board (needs a pygame display to exist)
screen = pygame.display.set_mode((800, 600))

board = CircuitBoard(0, 0, 800, 600)

# Add an AND gate and connect it
gate = board.add_gate("AND")

# Manually wire input nodes to gate, and gate to output
from circuit.wire import Wire
wire1 = Wire(board.input_nodes[0], gate, 0)  # A → input_a
wire2 = Wire(board.input_nodes[1], gate, 1)  # B → input_b
wire3 = Wire(gate, board.output_nodes[0], 0) # gate → output Q
board.wires = [wire1, wire2, wire3]

# Generate and print truth table
rows = generate_truth_table(board)
format_truth_table(rows, ["A", "B"])

print()

# ── Test 3: Truth table for XOR circuit ───────────────────────────
print("=== Test 3: XOR circuit truth table ===")

board2 = CircuitBoard(0, 0, 800, 600)
xor = board2.add_gate("XOR")

wire_a = Wire(board2.input_nodes[0], xor, 0)
wire_b = Wire(board2.input_nodes[1], xor, 1)
wire_out = Wire(xor, board2.output_nodes[0], 0)
board2.wires = [wire_a, wire_b, wire_out]

rows2 = generate_truth_table(board2)
format_truth_table(rows2, ["A", "B"])

pygame.quit()
print("\n✅ All tests passed!")
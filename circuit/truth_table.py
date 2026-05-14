# circuit/truth_table.py
#
# This file automatically generates a truth table from whatever
# circuit is currently built on the board.
#
# A truth table lists every possible combination of inputs
# and the output the circuit produces for each one.
#
# Example for a 2-input AND gate:
#
#   A | B | Output
#   0 | 0 |   0
#   0 | 1 |   0
#   1 | 0 |   0
#   1 | 1 |   1
#
# We generate this by temporarily setting the input nodes to
# every combination, running propagation, and recording the result.


def generate_truth_table(circuit_board):
    """
    Given a CircuitBoard, return a truth table as a list of rows.

    Each row is a dictionary:
    {
        "inputs": [0, 1, ...],   # one value per input node
        "output": 0 or 1 or None # what the circuit produces
    }

    Returns an empty list if there are no input or output nodes.
    """

    input_nodes  = circuit_board.input_nodes
    output_nodes = circuit_board.output_nodes

    # Need at least one input and one output to make a table
    if not input_nodes or not output_nodes:
        return []

    n = len(input_nodes)          # number of input nodes
    num_rows = 2 ** n             # e.g. 2 inputs → 4 rows, 3 inputs → 8 rows

    # Save the current input values so we can restore them afterward
    original_values = [node.value for node in input_nodes]

    rows = []

    for i in range(num_rows):
        # Convert the row number into a binary combination of inputs.
        #
        # Example with n=2:
        #   i=0 → binary "00" → [0, 0]
        #   i=1 → binary "01" → [0, 1]
        #   i=2 → binary "10" → [1, 0]
        #   i=3 → binary "11" → [1, 1]
        #
        # (i >> bit) & 1 extracts one bit at a time from the number i.
        # We reverse it so the leftmost input is the most significant bit.

        input_combo = []
        for bit in range(n - 1, -1, -1):
            input_combo.append((i >> bit) & 1)

        # Set each input node to its value for this row
        for node, val in zip(input_nodes, input_combo):
            node.value = val

        # Run signal propagation through the circuit
        circuit_board._propagate()

        # Read the output — we use the first output node
        output_val = output_nodes[0].value

        rows.append({
            "inputs": input_combo,
            "output": output_val
        })

    # Restore the original input values so the circuit looks unchanged
    for node, val in zip(input_nodes, original_values):
        node.value = val
    circuit_board._propagate()

    return rows


def format_truth_table(rows, input_labels=None):
    """
    Print a truth table nicely to the terminal — useful for debugging.

    rows         : the list returned by generate_truth_table()
    input_labels : list of strings like ["A", "B"] — defaults to A, B, C...
    """
    if not rows:
        print("No truth table — check that your circuit has inputs and outputs connected.")
        return

    n = len(rows[0]["inputs"])

    # Default labels A, B, C, D...
    if input_labels is None:
        input_labels = [chr(65 + i) for i in range(n)]  # 65 = ASCII 'A'

    # Header row
    header = " | ".join(input_labels) + " | Output"
    print(header)
    print("-" * len(header))

    # Data rows
    for row in rows:
        input_str = " | ".join(str(v) for v in row["inputs"])
        output_str = str(row["output"]) if row["output"] is not None else "?"
        print(f"{input_str} | {output_str}")
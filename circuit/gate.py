# circuit/gate.py
#
# A logic gate takes one or two boolean inputs (True/False, which we treat as 1 or 0)
# and produces one boolean output based on a rule.
#
# For example:
#   AND gate: output is True only if BOTH inputs are True
#   OR gate:  output is True if AT LEAST ONE input is True
#   NOT gate: flips the input — True becomes False, False becomes True


# This dictionary maps a gate's type name to its logic function.
# Each function takes inputs and returns the output.
#
# 'a' and 'b' are the two inputs. NOT only uses 'a'.
# 'int(...)' converts True/False to 1/0 so the math works cleanly.

GATE_FUNCTIONS = {
    "AND":  lambda a, b: int(a and b),
    "OR":   lambda a, b: int(a or b),
    "NOT":  lambda a, _: int(not a),       # ignores second input
    "XOR":  lambda a, b: int(a ^ b),       # True only if inputs DIFFER
    "NAND": lambda a, b: int(not (a and b)),
    "NOR":  lambda a, b: int(not (a or b)),
}


class Gate:
    """
    Represents a single logic gate on the circuit board.

    Each gate has:
    - a type (like "AND" or "XOR")
    - a position on screen (x, y)
    - up to two input slots
    - one output
    """

    # How wide and tall to draw each gate on screen (in pixels)
    WIDTH = 80
    HEIGHT = 50

    def __init__(self, gate_type, x, y):
        """
        Create a new gate.

        gate_type : string, e.g. "AND", "OR", "NOT"
        x, y      : where on the canvas to place it
        """
        if gate_type not in GATE_FUNCTIONS:
            raise ValueError(f"Unknown gate type: {gate_type}. Choose from {list(GATE_FUNCTIONS.keys())}")

        self.gate_type = gate_type
        self.x = x
        self.y = y

        # Input values — None means "nothing connected yet"
        self.input_a = None
        self.input_b = None

        # Whether the user is currently dragging this gate around
        self.dragging = False

        # Offset used during dragging so the gate doesn't snap to your cursor
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def compute(self):
        """
        Run the gate's logic and return its output (0 or 1).

        If any required input is missing (None), return None.
        NOT gates only need input_a.
        All others need both.
        """
        if self.gate_type == "NOT":
            if self.input_a is None:
                return None
            return GATE_FUNCTIONS["NOT"](self.input_a, None)
        else:
            if self.input_a is None or self.input_b is None:
                return None
            return GATE_FUNCTIONS[self.gate_type](self.input_a, self.input_b)

    def get_input_pin_positions(self):
        """
        Returns the screen (x, y) positions of this gate's input pins.
        These are the dots on the LEFT side of the gate where wires connect in.

        NOT gates have one input pin.
        All others have two.
        """
        if self.gate_type == "NOT":
            # One input pin, centered vertically
            return [(self.x, self.y + self.HEIGHT // 2)]
        else:
            # Two input pins, spaced vertically
            return [
                (self.x, self.y + self.HEIGHT // 4),       # top input
                (self.x, self.y + 3 * self.HEIGHT // 4),   # bottom input
            ]

    def get_output_pin_position(self):
        """
        Returns the screen (x, y) of this gate's single output pin.
        This is the dot on the RIGHT side of the gate where a wire goes out.
        """
        return (self.x + self.WIDTH, self.y + self.HEIGHT // 2)

    def move(self, new_x, new_y):
        """Update the gate's position (used while dragging)."""
        self.x = new_x
        self.y = new_y

    def __repr__(self):
        """A readable description when you print a gate — useful for debugging."""
        return f"Gate({self.gate_type} at ({self.x}, {self.y}) | a={self.input_a}, b={self.input_b} → {self.compute()})"
# circuit/wire.py
#
# A Wire connects one gate's output pin to another gate's input pin.
#
# Visually, a wire is just a line drawn between two (x, y) points.
# When a signal travels through it, we animate a small "pulse" dot
# that moves from the start point to the end point.


class Wire:
    """
    Represents a connection between two pins on the circuit board.

    A wire knows:
    - where it starts (the output pin of one gate)
    - where it ends (the input pin of another gate)
    - what signal value (0 or 1) is currently flowing through it
    - whether a pulse animation is currently traveling along it
    """

    # How fast the pulse dot moves along the wire (0.0 to 1.0 per frame)
    # 0.02 means it takes 50 frames to travel the full length
    PULSE_SPEED = 0.02

    def __init__(self, start_gate, end_gate, end_input_index):
        """
        Create a wire.

        start_gate      : the Gate whose output this wire comes from
        end_gate        : the Gate whose input this wire goes into
        end_input_index : 0 = top input (input_a), 1 = bottom input (input_b)
        """
        self.start_gate = start_gate
        self.end_gate = end_gate
        self.end_input_index = end_input_index  # which input slot on the end gate

        # The current signal value flowing through this wire (0 or 1)
        self.signal = 0

        # Pulse animation state
        # 'progress' goes from 0.0 (at start) to 1.0 (at end)
        # When it reaches 1.0, the animation resets
        self.pulse_active = False
        self.pulse_progress = 0.0

    def get_start_pos(self):
        """Where the wire begins — the output pin of the start gate."""
        return self.start_gate.get_output_pin_position()

    def get_end_pos(self):
        """Where the wire ends — the correct input pin of the end gate."""
        pins = self.end_gate.get_input_pin_positions()
        return pins[self.end_input_index]

    def get_pulse_pos(self):
        """
        Calculate where the pulse dot is right now.

        We interpolate between start and end based on pulse_progress.
        At progress=0.0, the dot is at the start.
        At progress=1.0, the dot is at the end.

        Interpolation formula:
            current = start + (end - start) * progress
        """
        if not self.pulse_active:
            return None

        x1, y1 = self.get_start_pos()
        x2, y2 = self.get_end_pos()

        # Linear interpolation (lerp)
        pulse_x = x1 + (x2 - x1) * self.pulse_progress
        pulse_y = y1 + (y2 - y1) * self.pulse_progress

        return (int(pulse_x), int(pulse_y))

    def trigger_pulse(self, signal_value):
        """
        Start a pulse animation traveling down this wire.

        signal_value : 0 or 1 — what value is being sent
        """
        self.signal = signal_value
        self.pulse_active = True
        self.pulse_progress = 0.0

    def update(self):
        """
        Advance the pulse animation by one frame.

        Call this every frame from the main game loop.
        When the pulse reaches the end, it stops and
        delivers its signal to the destination gate's input.
        """
        if not self.pulse_active:
            return

        self.pulse_progress += self.PULSE_SPEED

        if self.pulse_progress >= 1.0:
            # Pulse has arrived — deliver the signal
            self.pulse_active = False
            self.pulse_progress = 0.0
            self._deliver_signal()

    def _deliver_signal(self):
        """
        Write the signal value into the correct input slot of the end gate.

        end_input_index 0 → input_a
        end_input_index 1 → input_b
        """
        if self.end_input_index == 0:
            self.end_gate.input_a = self.signal
        else:
            self.end_gate.input_b = self.signal

    def get_color(self):
        """
        Returns the color this wire should be drawn in.

        Signal 1 (high) → bright electric blue
        Signal 0 (low)  → dim grey
        """
        if self.signal == 1:
            return (100, 180, 255)   # electric blue
        else:
            return (80, 80, 100)     # dim grey

    def __repr__(self):
        return (f"Wire({self.start_gate.gate_type} → "
                f"{self.end_gate.gate_type} input {self.end_input_index} "
                f"| signal={self.signal})")
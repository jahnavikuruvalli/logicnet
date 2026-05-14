# circuit/circuit_board.py
#
# The CircuitBoard is the main canvas where everything lives.
#
# It is responsible for:
#   - Storing all gates and wires
#   - Handling mouse clicks and drags
#   - Connecting pins with wires when you click them
#   - Propagating signals through the circuit every frame
#   - Drawing everything onto the screen


import pygame
from circuit.gate import Gate
from circuit.wire import Wire


# Colors used when drawing gates and pins
COLOR_GATE_BODY    = (30, 35, 55)       # dark blue-grey box
COLOR_GATE_BORDER  = (80, 120, 200)     # blue outline
COLOR_GATE_TEXT    = (200, 220, 255)    # light text
COLOR_PIN_DEFAULT  = (100, 100, 140)    # unconnected pin
COLOR_PIN_HOVER    = (255, 220, 80)     # pin you're hovering over
COLOR_PIN_SELECTED = (80, 255, 180)     # pin you've clicked and are wiring from
COLOR_WIRE_PULSE   = (255, 255, 255)    # the moving dot on a wire
COLOR_INPUT_ON     = (0, 255, 136)      # green — input node is 1
COLOR_INPUT_OFF    = (60, 60, 80)       # dark — input node is 0
COLOR_OUTPUT       = (100, 180, 255)    # blue — output node

PIN_RADIUS = 6          # how big the pin dots are drawn
INPUT_NODE_RADIUS = 14  # how big the clickable input toggles are


class InputNode:
    """
    A toggleable input on the left side of the canvas.
    Clicking it flips between 0 and 1.
    This is how you feed values into the circuit.
    """
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label   # "A", "B", "C" etc.
        self.value = 0       # starts at 0 (off)

    def toggle(self):
        self.value = 1 - self.value   # flips 0→1 or 1→0

    def contains(self, mx, my):
        """Returns True if mouse position (mx, my) is inside this node."""
        dx = mx - self.x
        dy = my - self.y
        return (dx * dx + dy * dy) <= INPUT_NODE_RADIUS ** 2

    def get_output_pin_position(self):
        """Input nodes have one output pin on their right side."""
        return (self.x + INPUT_NODE_RADIUS, self.y)


class OutputNode:
    """
    A read-only output display on the right side of the canvas.
    It shows the final result of the circuit.
    """
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label
        self.value = None   # None until something is connected

    def get_input_pin_positions(self):
        """Output nodes accept one wire into their left side."""
        return [(self.x - INPUT_NODE_RADIUS, self.y)]

    def get_input_pin_position(self):
        return self.get_input_pin_positions()[0]


class CircuitBoard:
    """
    The full circuit canvas.

    Holds all gates, wires, input nodes, and output nodes.
    Handles all mouse interaction and drawing.
    """

    def __init__(self, x, y, width, height):
        """
        x, y          : top-left corner of the panel on screen
        width, height : size of the canvas area
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.gates = []       # list of Gate objects
        self.wires = []       # list of Wire objects
        self.input_nodes = [] # list of InputNode objects
        self.output_nodes = []# list of OutputNode objects

        # Wiring state — when you click an output pin, we remember it
        # and wait for you to click an input pin to complete the wire
        self.pending_wire_start = None   # (gate_or_node, pin_type)
        self.hovered_pin = None          # pin your mouse is currently over

        # Dragging state
        self.dragging_gate = None

        # Add some default input/output nodes to start with
        self._setup_default_nodes()

    def _setup_default_nodes(self):
        """Create two input nodes and one output node by default."""
        self.input_nodes = [
            InputNode(self.x + 40, self.y + self.height // 3, "A"),
            InputNode(self.x + 40, self.y + 2 * self.height // 3, "B"),
        ]
        self.output_nodes = [
            OutputNode(self.x + self.width - 40, self.y + self.height // 2, "Q"),
        ]

    def add_gate(self, gate_type):
        """
        Drop a new gate onto the canvas near the center.
        Called when the user clicks a gate button in the toolbar.
        """
        cx = self.x + self.width // 2
        cy = self.y + self.height // 2
        # Offset slightly so stacked gates don't overlap exactly
        offset = len(self.gates) * 20
        gate = Gate(gate_type, cx + offset, cy + offset)
        self.gates.append(gate)
        return gate

    # ------------------------------------------------------------------
    # Mouse handling
    # ------------------------------------------------------------------

    def handle_event(self, event):
        """
        Process a single pygame event.
        Call this from the main game loop for every event.
        """
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._handle_click(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_gate = None   # stop dragging

        elif event.type == pygame.MOUSEMOTION:
            self._handle_mouse_move(event.pos)

    def _handle_click(self, pos):
        mx, my = pos

        # 1. Check if clicking an input node toggle
        for node in self.input_nodes:
            if node.contains(mx, my):
                node.toggle()
                self._propagate()
                return

        # 2. Check if clicking an output pin (start a wire)
        pin = self._get_pin_at(mx, my)
        if pin:
            pin_owner, pin_type, pin_index = pin
            if pin_type == "output":
                # Start drawing a wire from this pin
                self.pending_wire_start = (pin_owner, pin_type)
                return
            elif pin_type == "input" and self.pending_wire_start is not None:
                # Complete the wire
                self._complete_wire(pin_owner, pin_index)
                return

        # 3. Cancel pending wire if clicking empty space
        if self.pending_wire_start is not None:
            self.pending_wire_start = None
            return

        # 4. Check if clicking a gate body to drag it
        for gate in reversed(self.gates):  # reversed so top gate gets priority
            gx, gy = gate.x, gate.y
            gw, gh = Gate.WIDTH, Gate.HEIGHT
            if gx <= mx <= gx + gw and gy <= my <= gy + gh:
                self.dragging_gate = gate
                gate.drag_offset_x = mx - gate.x
                gate.drag_offset_y = my - gate.y
                return

    def _handle_mouse_move(self, pos):
        mx, my = pos

        # Move the gate being dragged
        if self.dragging_gate:
            self.dragging_gate.move(
                mx - self.dragging_gate.drag_offset_x,
                my - self.dragging_gate.drag_offset_y
            )

        # Track which pin the mouse is hovering over
        pin = self._get_pin_at(mx, my)
        self.hovered_pin = pin

    def _get_pin_at(self, mx, my, radius=PIN_RADIUS + 4):
        """
        Check if (mx, my) is close to any pin on any gate or node.

        Returns (owner, pin_type, index) or None.
        pin_type is "output" or "input"
        index is which input pin (0 or 1) for input pins
        """
        # Check gate output pins
        for gate in self.gates:
            ox, oy = gate.get_output_pin_position()
            if abs(mx - ox) <= radius and abs(my - oy) <= radius:
                return (gate, "output", 0)

        # Check gate input pins
        for gate in self.gates:
            for i, (px, py) in enumerate(gate.get_input_pin_positions()):
                if abs(mx - px) <= radius and abs(my - py) <= radius:
                    return (gate, "input", i)

        # Check input node output pins
        for node in self.input_nodes:
            ox, oy = node.get_output_pin_position()
            if abs(mx - ox) <= radius and abs(my - oy) <= radius:
                return (node, "output", 0)

        # Check output node input pins
        for node in self.output_nodes:
            px, py = node.get_input_pin_position()
            if abs(mx - px) <= radius and abs(my - py) <= radius:
                return (node, "input", 0)

        return None

    def _complete_wire(self, end_owner, end_input_index):
        """
        Finish drawing a wire from the pending start pin to end_owner's input.
        """
        start_owner, _ = self.pending_wire_start
        self.pending_wire_start = None

        # Don't connect a gate to itself
        if start_owner is end_owner:
            return

        # Handle wiring to an OutputNode separately
        if isinstance(end_owner, OutputNode):
            wire = Wire(start_owner, end_owner, 0)
            # Patch: OutputNode doesn't have input_a/b, we handle it in propagate
            wire._is_output_wire = True
            self.wires.append(wire)
            self._propagate()
            return

        wire = Wire(start_owner, end_owner, end_input_index)
        self.wires.append(wire)
        self._propagate()

    # ------------------------------------------------------------------
    # Signal propagation
    # ------------------------------------------------------------------

    def _propagate(self):
        """
        Push signal values through the entire circuit.

        We do multiple passes to handle chains of gates.
        Each pass reads gate outputs and writes them to connected inputs.
        """
        # First, set all gate inputs to None (disconnected)
        for gate in self.gates:
            gate.input_a = None
            gate.input_b = None

        # Also reset output nodes
        for node in self.output_nodes:
            node.value = None

        # Run several passes so signals travel through chains
        for _ in range(len(self.gates) + 2):
            for wire in self.wires:
                # Get the signal from the wire's source
                source = wire.start_gate
                if isinstance(source, InputNode):
                    signal = source.value
                elif isinstance(source, Gate):
                    signal = source.compute()
                else:
                    signal = None

                if signal is None:
                    continue

                wire.signal = signal

                # Deliver to destination
                dest = wire.end_gate
                if isinstance(dest, OutputNode):
                    dest.value = signal
                elif isinstance(dest, Gate):
                    if wire.end_input_index == 0:
                        dest.input_a = signal
                    else:
                        dest.input_b = signal

    def trigger_all_pulses(self):
        """Trigger visual pulse animations on all wires."""
        for wire in self.wires:
            wire.trigger_pulse(wire.signal)

    # ------------------------------------------------------------------
    # Update (called every frame)
    # ------------------------------------------------------------------

    def update(self):
        """Advance all wire pulse animations."""
        for wire in self.wires:
            wire.update()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw(self, surface):
        """Draw everything on the circuit board."""

        # Panel background
        pygame.draw.rect(surface, (10, 12, 20),
                         (self.x, self.y, self.width, self.height))
        # Panel border
        pygame.draw.rect(surface, (40, 50, 80),
                         (self.x, self.y, self.width, self.height), 2)

        # Draw wires first (so gates appear on top)
        self._draw_wires(surface)

        # Draw gates
        for gate in self.gates:
            self._draw_gate(surface, gate)

        # Draw input/output nodes
        self._draw_nodes(surface)

        # Draw the wire currently being dragged (follows mouse)
        self._draw_pending_wire(surface)

    def _draw_wires(self, surface):
        for wire in self.wires:
            x1, y1 = wire.get_start_pos()
            x2, y2 = wire.get_end_pos()
            color = wire.get_color()

            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 2)

            # Draw pulse dot if active
            pulse_pos = wire.get_pulse_pos()
            if pulse_pos:
                pygame.draw.circle(surface, COLOR_WIRE_PULSE, pulse_pos, 4)

    def _draw_gate(self, surface, gate):
        gx, gy = gate.x, gate.y
        gw, gh = Gate.WIDTH, Gate.HEIGHT

        # Gate body
        pygame.draw.rect(surface, COLOR_GATE_BODY, (gx, gy, gw, gh), border_radius=6)
        pygame.draw.rect(surface, COLOR_GATE_BORDER, (gx, gy, gw, gh), 2, border_radius=6)

        # Gate label
        font = pygame.font.SysFont("monospace", 14, bold=True)
        label = font.render(gate.gate_type, True, COLOR_GATE_TEXT)
        lx = gx + (gw - label.get_width()) // 2
        ly = gy + (gh - label.get_height()) // 2
        surface.blit(label, (lx, ly))

        # Input pins
        for i, (px, py) in enumerate(gate.get_input_pin_positions()):
            color = self._pin_color((gate, "input", i))
            pygame.draw.circle(surface, color, (px, py), PIN_RADIUS)

        # Output pin
        ox, oy = gate.get_output_pin_position()
        color = self._pin_color((gate, "output", 0))
        pygame.draw.circle(surface, color, (ox, oy), PIN_RADIUS)

        # Show computed value near output pin
        val = gate.compute()
        if val is not None:
            font_small = pygame.font.SysFont("monospace", 11)
            val_surf = font_small.render(str(val), True, (180, 220, 180))
            surface.blit(val_surf, (ox + 8, oy - 8))

    def _draw_nodes(self, surface):
        font = pygame.font.SysFont("monospace", 13, bold=True)

        for node in self.input_nodes:
            color = COLOR_INPUT_ON if node.value == 1 else COLOR_INPUT_OFF
            pygame.draw.circle(surface, color, (node.x, node.y), INPUT_NODE_RADIUS)
            pygame.draw.circle(surface, (150, 200, 255), (node.x, node.y),
                               INPUT_NODE_RADIUS, 2)
            label = font.render(node.label, True, (220, 220, 255))
            surface.blit(label, (node.x - label.get_width() // 2 - 18, node.y - 8))
            # Output pin dot
            ox, oy = node.get_output_pin_position()
            pygame.draw.circle(surface, COLOR_PIN_DEFAULT, (ox, oy), PIN_RADIUS)

        for node in self.output_nodes:
            color = COLOR_INPUT_ON if node.value == 1 else (
                    COLOR_INPUT_OFF if node.value == 0 else (50, 50, 70))
            pygame.draw.circle(surface, color, (node.x, node.y), INPUT_NODE_RADIUS)
            pygame.draw.circle(surface, COLOR_OUTPUT, (node.x, node.y),
                               INPUT_NODE_RADIUS, 2)
            label = font.render(node.label, True, (220, 220, 255))
            surface.blit(label, (node.x + INPUT_NODE_RADIUS + 6, node.y - 8))
            # Input pin dot
            px, py = node.get_input_pin_position()
            pygame.draw.circle(surface, COLOR_PIN_DEFAULT, (px, py), PIN_RADIUS)

    def _draw_pending_wire(self, surface):
        """Draw a line from the selected output pin to the current mouse position."""
        if self.pending_wire_start is None:
            return
        owner, _ = self.pending_wire_start
        if isinstance(owner, InputNode):
            start_pos = owner.get_output_pin_position()
        else:
            start_pos = owner.get_output_pin_position()

        mouse_pos = pygame.mouse.get_pos()
        pygame.draw.line(surface, COLOR_PIN_SELECTED, start_pos, mouse_pos, 2)

    def _pin_color(self, pin_tuple):
        """Return the color for a pin based on hover/selected state."""
        if self.pending_wire_start and pin_tuple == self.pending_wire_start:
            return COLOR_PIN_SELECTED
        if self.hovered_pin and pin_tuple[:2] == self.hovered_pin[:2]:
            return COLOR_PIN_HOVER
        return COLOR_PIN_DEFAULT
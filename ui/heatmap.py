# ui/heatmap.py
#
# After training, this draws a color overlay on top of each gate
# on the circuit board.
#
# Gates that were hardest for the network to learn
# (contributed most to backprop error) glow red.
# Gates that were easy glow green.
#
# HOW WE MEASURE "DIFFICULTY":
# During training, the network's loss tells us how wrong it was.
# We run the truth table through the network and check which
# input combinations still produce the most error.
# We then trace those inputs back through the circuit to see
# which gates they passed through — those gates get blamed.
#
# This is an approximation, not exact backprop attribution,
# but it produces a meaningful and visually interesting result.


import pygame
import numpy as np


COLOR_EASY   = (0, 200, 80,  120)   # green with transparency
COLOR_HARD   = (220, 40, 40, 120)   # red with transparency
COLOR_MEDIUM = (200, 180, 0, 100)   # yellow for middle ground


class GateHeatmap:
    """
    Computes and draws a difficulty heatmap over the circuit gates.

    Call compute() once after training finishes.
    Call draw() every frame to render the overlay.
    """

    def __init__(self):
        # Maps gate object → difficulty score (0.0 = easy, 1.0 = hard)
        self.gate_scores = {}
        self.visible     = False   # only show after training

    def compute(self, circuit_board, trainer):
        """
        Compute a difficulty score for each gate.

        We measure how much error the network still produces
        for each truth table row, then attribute that error
        to the gates whose outputs fed into that row's inputs.

        circuit_board : CircuitBoard instance
        trainer       : Trainer instance (after training)
        """
        self.gate_scores = {}

        if not trainer.truth_table or not trainer.net.activations:
            return

        import numpy as np

        # Get per-row errors from the network
        row_errors = []
        for row in trainer.truth_table:
            x = np.array(row["inputs"], dtype=float).reshape(-1, 1)
            prediction = trainer.net.forward(x)
            error = abs(float(prediction.flatten()[0]) - float(row["output"]))
            row_errors.append(error)

        # Normalize errors to 0–1
        max_error = max(row_errors) if row_errors else 1.0
        if max_error < 1e-8:
            max_error = 1.0
        row_errors = [e / max_error for e in row_errors]

        # For each gate, find which truth table rows it participates in.
        # A gate participates in a row if any of its inputs come
        # from an input node that is active (1) in that row.
        #
        # Simple heuristic: gates deeper in the circuit (more wires
        # between them and inputs) get weighted by downstream error.

        gates = circuit_board.gates
        if not gates:
            return

        # Build a map of which gates connect to which input nodes
        # by tracing wires backward
        gate_input_map = {gate: set() for gate in gates}

        for wire in circuit_board.wires:
            from circuit.circuit_board import InputNode
            if isinstance(wire.start_gate, InputNode):
                if wire.end_gate in gate_input_map:
                    gate_input_map[wire.end_gate].add(
                        circuit_board.input_nodes.index(wire.start_gate)
                    )

        # Score each gate based on average error of rows
        # where its connected inputs are active
        for gate in gates:
            connected_inputs = gate_input_map[gate]

            if not connected_inputs:
                # Gate not connected to any input — medium difficulty
                self.gate_scores[gate] = 0.5
                continue

            relevant_errors = []
            for row_i, row in enumerate(trainer.truth_table):
                # Check if any of this gate's inputs are active in this row
                active = any(
                    row["inputs"][inp_i] == 1
                    for inp_i in connected_inputs
                    if inp_i < len(row["inputs"])
                )
                if active:
                    relevant_errors.append(row_errors[row_i])

            if relevant_errors:
                self.gate_scores[gate] = sum(relevant_errors) / len(relevant_errors)
            else:
                self.gate_scores[gate] = 0.0

        self.visible = True

    def draw(self, surface, circuit_board):
        """
        Draw colored overlays on each gate.

        Called every frame after compute() has been run.
        """
        if not self.visible or not self.gate_scores:
            return

        from circuit.gate import Gate

        for gate, score in self.gate_scores.items():
            # Interpolate between green (easy) and red (hard)
            if score < 0.5:
                # Easy to medium — green to yellow
                t = score * 2   # 0→1 as score goes 0→0.5
                r = int(COLOR_EASY[0] + t * (COLOR_MEDIUM[0] - COLOR_EASY[0]))
                g = int(COLOR_EASY[1] + t * (COLOR_MEDIUM[1] - COLOR_EASY[1]))
                b = int(COLOR_EASY[2] + t * (COLOR_MEDIUM[2] - COLOR_EASY[2]))
                a = int(COLOR_EASY[3] + t * (COLOR_MEDIUM[3] - COLOR_EASY[3]))
            else:
                # Medium to hard — yellow to red
                t = (score - 0.5) * 2   # 0→1 as score goes 0.5→1.0
                r = int(COLOR_MEDIUM[0] + t * (COLOR_HARD[0] - COLOR_MEDIUM[0]))
                g = int(COLOR_MEDIUM[1] + t * (COLOR_HARD[1] - COLOR_MEDIUM[1]))
                b = int(COLOR_MEDIUM[2] + t * (COLOR_HARD[2] - COLOR_MEDIUM[2]))
                a = int(COLOR_MEDIUM[3] + t * (COLOR_HARD[3] - COLOR_MEDIUM[3]))

            color = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b)),
                max(0, min(255, a))
            )

            # Draw a transparent rectangle over the gate
            overlay = pygame.Surface(
                (Gate.WIDTH, Gate.HEIGHT), pygame.SRCALPHA
            )
            overlay.fill(color)
            surface.blit(overlay, (gate.x, gate.y))

            # Draw a colored border ring around the gate
            border_color = (
                max(0, min(255, r)),
                max(0, min(255, g)),
                max(0, min(255, b))
            )
            pygame.draw.rect(
                surface, border_color,
                (gate.x, gate.y, Gate.WIDTH, Gate.HEIGHT),
                2, border_radius=6
            )

            # Score label
            font  = pygame.font.SysFont("monospace", 9)
            label = f"{score:.2f}"
            lsurf = font.render(label, True, (220, 220, 220))
            surface.blit(lsurf, (
                gate.x + Gate.WIDTH  // 2 - lsurf.get_width()  // 2,
                gate.y + Gate.HEIGHT - 12
            ))

    def hide(self):
        """Turn off the heatmap overlay."""
        self.visible = False

    def clear(self):
        """Wipe scores and hide."""
        self.gate_scores = {}
        self.visible     = False
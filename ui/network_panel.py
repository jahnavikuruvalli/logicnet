# ui/network_panel.py
#
# Draws the right panel — the live neural network visualization.
#
# What gets drawn:
#   - Neurons as glowing circles, brightness = activation value
#   - Connection lines colored and sized by weight value
#     (thick blue = strong positive, thick red = strong negative)
#   - Floating numbers showing input × weight during forward pass
#   - Orange error signals flowing right-to-left during backprop
#   - A loss curve graph at the bottom like a heartbeat monitor
#
# This panel reads directly from a Trainer instance every frame.


import pygame
import math
import numpy as np


# ── Colors ─────────────────────────────────────────────────────────
COLOR_BG           = (10, 10, 18)
COLOR_NEURON_BASE  = (30, 35, 60)
COLOR_NEURON_GLOW  = (80, 160, 255)     # activated neuron color
COLOR_WEIGHT_POS   = (60, 140, 255)     # positive weight — blue
COLOR_WEIGHT_NEG   = (255, 60, 80)      # negative weight — red
COLOR_WEIGHT_ZERO  = (40, 40, 60)       # near-zero weight — dark
COLOR_BACKPROP     = (255, 140, 0)      # backprop error — orange
COLOR_BORDER       = (35, 45, 70)
COLOR_TEXT         = (160, 190, 230)
COLOR_TITLE        = (100, 160, 255)
COLOR_LOSS_LINE    = (80, 220, 160)     # loss curve color
COLOR_LOSS_BG      = (12, 18, 30)

NEURON_RADIUS = 18
LOSS_GRAPH_HEIGHT = 120


class NetworkPanel:
    """
    Draws the neural network visualization in the right panel.
    """

    def __init__(self, rect):
        self.rect    = rect
        self.trainer = None

        self.font_title  = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_small  = pygame.font.SysFont("monospace", 10)
        self.font_weight = pygame.font.SysFont("monospace", 9)

        # Animation state
        # forward_alpha: 0→1, controls how visible forward pass numbers are
        # backprop_alpha: 0→1, controls orange backprop glow
        self.forward_alpha  = 0.0
        self.backprop_alpha = 0.0

        # Cached neuron positions so we don't recompute every frame
        self._neuron_positions = []   # list of lists: [layer][neuron] = (x, y)
        self._last_layer_sizes = []

    def set_trainer(self, trainer):
        self.trainer = trainer

    # ── Main draw entry point ──────────────────────────────────────

    def draw(self, surface):
        """Draw the entire network panel."""
        pygame.draw.rect(surface, COLOR_BG, self.rect)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1)

        # Title
        title = self.font_title.render("NEURAL NETWORK", True, COLOR_TITLE)
        surface.blit(title, (self.rect.x + 10, self.rect.y + 8))

        if self.trainer is None or not self.trainer.net.weights:
            self._draw_empty(surface)
            return

        net = self.trainer.net

        # Compute neuron layout positions
        self._compute_positions(net)

        # Network drawing area (above the loss graph)
        net_area_bottom = self.rect.bottom - LOSS_GRAPH_HEIGHT - 10

        # Draw in order: connections first, then neurons on top
        self._draw_connections(surface, net, net_area_bottom)
        self._draw_neurons(surface, net, net_area_bottom)

        # Draw loss curve at the bottom
        self._draw_loss_curve(surface)

        # Animate forward/backprop overlays
        self._update_animations()

    # ── Layout computation ─────────────────────────────────────────

    def _compute_positions(self, net):
        """
        Compute the (x, y) center position of every neuron.

        Neurons are spread evenly across the panel width (layers)
        and panel height (neurons within a layer).

        Results cached in self._neuron_positions.
        """
        if net.layer_sizes == self._last_layer_sizes:
            return  # no need to recompute if architecture hasn't changed

        self._last_layer_sizes = list(net.layer_sizes)
        self._neuron_positions = []

        # Available drawing area
        x_start = self.rect.x + 40
        x_end   = self.rect.right - 40
        y_start = self.rect.y + 40
        y_end   = self.rect.bottom - LOSS_GRAPH_HEIGHT - 30

        n_layers = len(net.layer_sizes)
        x_gap    = (x_end - x_start) / max(n_layers - 1, 1)

        for layer_i, n_neurons in enumerate(net.layer_sizes):
            x = x_start + layer_i * x_gap
            positions = []

            # Cap display to 8 neurons per layer so they fit on screen
            display_count = min(n_neurons, 8)
            y_gap = (y_end - y_start) / max(display_count - 1, 1) \
                    if display_count > 1 else 0

            for neuron_i in range(display_count):
                if display_count == 1:
                    y = (y_start + y_end) / 2
                else:
                    y = y_start + neuron_i * y_gap
                positions.append((int(x), int(y)))

            self._neuron_positions.append(positions)

    # ── Drawing connections ────────────────────────────────────────

    def _draw_connections(self, surface, net, net_area_bottom):
        """
        Draw all weight connections between layers.

        Line thickness = how strong the weight is (abs value)
        Line color     = blue for positive, red for negative
        """
        for layer_i in range(len(net.weights)):
            w_matrix = net.weights[layer_i]   # shape: (n_out, n_in)

            src_positions = self._neuron_positions[layer_i]
            dst_positions = self._neuron_positions[layer_i + 1]

            # Find max weight for normalizing thickness
            max_w = np.max(np.abs(w_matrix)) + 1e-8

            for dst_i, dst_pos in enumerate(dst_positions):
                if dst_i >= w_matrix.shape[0]:
                    break
                for src_i, src_pos in enumerate(src_positions):
                    if src_i >= w_matrix.shape[1]:
                        break

                    w = float(w_matrix[dst_i, src_i])

                    # Normalize weight to 0–1 range
                    strength = abs(w) / max_w

                    # Skip very weak connections (cleaner visual)
                    if strength < 0.05:
                        continue

                    # Color: interpolate between zero-color and pos/neg color
                    if w > 0:
                        r = int(COLOR_WEIGHT_ZERO[0] + strength *
                                (COLOR_WEIGHT_POS[0] - COLOR_WEIGHT_ZERO[0]))
                        g = int(COLOR_WEIGHT_ZERO[1] + strength *
                                (COLOR_WEIGHT_POS[1] - COLOR_WEIGHT_ZERO[1]))
                        b = int(COLOR_WEIGHT_ZERO[2] + strength *
                                (COLOR_WEIGHT_POS[2] - COLOR_WEIGHT_ZERO[2]))
                    else:
                        r = int(COLOR_WEIGHT_ZERO[0] + strength *
                                (COLOR_WEIGHT_NEG[0] - COLOR_WEIGHT_ZERO[0]))
                        g = int(COLOR_WEIGHT_ZERO[1] + strength *
                                (COLOR_WEIGHT_NEG[1] - COLOR_WEIGHT_ZERO[1]))
                        b = int(COLOR_WEIGHT_ZERO[2] + strength *
                                (COLOR_WEIGHT_NEG[2] - COLOR_WEIGHT_ZERO[2]))

                    color = (
                        max(0, min(255, r)),
                        max(0, min(255, g)),
                        max(0, min(255, b))
                    )

                    # Thickness 1–4 pixels based on strength
                    thickness = max(1, int(strength * 4))

                    pygame.draw.line(surface, color, src_pos, dst_pos, thickness)

                    # Backprop overlay — orange tint on connections
                    if self.backprop_alpha > 0.1 and layer_i > 0:
                        alpha = int(self.backprop_alpha * 180)
                        overlay_color = (
                            min(255, color[0] + alpha),
                            max(0, color[1] - alpha // 2),
                            max(0, color[2] - alpha // 2)
                        )
                        if strength > 0.2:
                            pygame.draw.line(surface, overlay_color,
                                             src_pos, dst_pos, max(1, thickness - 1))

    # ── Drawing neurons ────────────────────────────────────────────

    def _draw_neurons(self, surface, net, net_area_bottom):
        """
        Draw each neuron as a glowing circle.

        Brightness scales with the neuron's activation value.
        """
        activations = net.activations  # filled after last forward pass

        for layer_i, positions in enumerate(self._neuron_positions):
            for neuron_i, (x, y) in enumerate(positions):

                # Get activation value if available
                act_val = 0.5   # default if no forward pass yet
                if activations and layer_i < len(activations):
                    layer_act = activations[layer_i]
                    if neuron_i < layer_act.shape[0]:
                        act_val = float(layer_act[neuron_i, 0])

                # Base circle (dark)
                pygame.draw.circle(surface, COLOR_NEURON_BASE,
                                   (x, y), NEURON_RADIUS)

                # Glow circle — brighter when activation is higher
                glow_intensity = max(0.0, min(1.0, act_val))
                glow_r = int(COLOR_NEURON_BASE[0] +
                             glow_intensity * (COLOR_NEURON_GLOW[0] - COLOR_NEURON_BASE[0]))
                glow_g = int(COLOR_NEURON_BASE[1] +
                             glow_intensity * (COLOR_NEURON_GLOW[1] - COLOR_NEURON_BASE[1]))
                glow_b = int(COLOR_NEURON_BASE[2] +
                             glow_intensity * (COLOR_NEURON_GLOW[2] - COLOR_NEURON_BASE[2]))
                glow_color = (
                    max(0, min(255, glow_r)),
                    max(0, min(255, glow_g)),
                    max(0, min(255, glow_b))
                )
                pygame.draw.circle(surface, glow_color,
                                   (x, y), NEURON_RADIUS)

                # Outer ring
                pygame.draw.circle(surface, COLOR_BORDER,
                                   (x, y), NEURON_RADIUS, 2)

                # Activation value label inside neuron
                label = f"{act_val:.1f}"
                surf  = self.font_weight.render(label, True, (200, 220, 255))
                surface.blit(surf, (x - surf.get_width() // 2,
                                    y - surf.get_height() // 2))

                # Layer label above first neuron in each layer
                if neuron_i == 0:
                    if layer_i == 0:
                        layer_name = "INPUT"
                    elif layer_i == len(self._neuron_positions) - 1:
                        layer_name = "OUTPUT"
                    else:
                        layer_name = f"H{layer_i}"

                    lsurf = self.font_small.render(layer_name, True,
                                                   (80, 100, 140))
                    surface.blit(lsurf, (x - lsurf.get_width() // 2,
                                         positions[0][1] - NEURON_RADIUS - 18))

    # ── Loss curve ─────────────────────────────────────────────────

    def _draw_loss_curve(self, surface):
        """
        Draw the loss curve as a live graph at the bottom of the panel.

        Like a heartbeat monitor — starts high and jagged,
        smooths and falls as the network improves.
        """
        graph_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.bottom - LOSS_GRAPH_HEIGHT,
            self.rect.width - 20,
            LOSS_GRAPH_HEIGHT - 10
        )

        # Background
        pygame.draw.rect(surface, COLOR_LOSS_BG, graph_rect, border_radius=4)
        pygame.draw.rect(surface, COLOR_BORDER, graph_rect, 1, border_radius=4)

        # Label
        label = self.font_small.render("LOSS", True, (80, 110, 150))
        surface.blit(label, (graph_rect.x + 6, graph_rect.y + 4))

        if self.trainer is None:
            return

        loss_history = self.trainer.net.loss_history
        if len(loss_history) < 2:
            return

        # Sample at most 300 points so it's not too dense
        history = loss_history
        if len(history) > 300:
            step = len(history) // 300
            history = history[::step]

        max_loss = max(history) + 1e-8
        min_loss = min(history)

        # Draw the line
        points = []
        for i, loss in enumerate(history):
            x = graph_rect.x + int(i / (len(history) - 1) * graph_rect.width)
            # Normalize loss to graph height
            normalized = (loss - min_loss) / (max_loss - min_loss + 1e-8)
            y = graph_rect.bottom - int(normalized * (graph_rect.height - 20)) - 5
            points.append((x, y))

        if len(points) >= 2:
            pygame.draw.lines(surface, COLOR_LOSS_LINE, False, points, 2)

        # Current loss value
        current_loss = loss_history[-1]
        loss_text    = f"{current_loss:.4f}"
        lsurf = self.font_small.render(loss_text, True, COLOR_LOSS_LINE)
        surface.blit(lsurf, (graph_rect.right - lsurf.get_width() - 6,
                              graph_rect.y + 4))

    # ── Animation updates ──────────────────────────────────────────

    def _update_animations(self):
        """
        Fade in/out the forward pass and backprop animation overlays.

        Called every frame. The trainer sets flags we read here.
        """
        if self.trainer is None:
            return

        # If training is running, pulse both alphas
        if self.trainer.is_training:
            # Forward pass brightens then fades
            self.forward_alpha = min(1.0, self.forward_alpha + 0.05)

            # Backprop lags behind forward pass
            if self.forward_alpha > 0.5:
                self.backprop_alpha = min(1.0, self.backprop_alpha + 0.04)
            else:
                self.backprop_alpha = max(0.0, self.backprop_alpha - 0.06)
        else:
            # Fade everything out when not training
            self.forward_alpha  = max(0.0, self.forward_alpha  - 0.03)
            self.backprop_alpha = max(0.0, self.backprop_alpha - 0.03)

    # ── Empty state ────────────────────────────────────────────────

    def _draw_empty(self, surface):
        lines = ["Connect a circuit", "and hit Train", "to see the network", "learn live."]
        font  = pygame.font.SysFont("monospace", 13)
        y     = self.rect.centery - len(lines) * 22 // 2
        for line in lines:
            surf = font.render(line, True, (60, 80, 110))
            x    = self.rect.centerx - surf.get_width() // 2
            surface.blit(surf, (x, y))
            y += 22
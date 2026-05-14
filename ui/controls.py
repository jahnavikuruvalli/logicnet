# ui/controls.py
#
# Draws the toolbar at the top of the window.
#
# Contains:
#   - Gate buttons (AND, OR, NOT, XOR, NAND, NOR) — click to add a gate
#   - Train / Pause / Reset buttons
#   - Speed slider — controls steps_per_frame
#   - Architecture controls — hidden layers and neurons per layer
#   - A "Generate Truth Table" button


import pygame


# Colors
COLOR_BTN_GATE     = (25, 35, 65)
COLOR_BTN_GATE_HOV = (40, 60, 110)
COLOR_BTN_GATE_BDR = (60, 100, 200)

COLOR_BTN_TRAIN    = (20, 60, 35)
COLOR_BTN_TRAIN_HOV= (30, 90, 50)
COLOR_BTN_TRAIN_BDR= (0, 180, 80)

COLOR_BTN_PAUSE    = (60, 55, 20)
COLOR_BTN_PAUSE_HOV= (90, 80, 25)
COLOR_BTN_PAUSE_BDR= (200, 180, 0)

COLOR_BTN_RESET    = (60, 20, 20)
COLOR_BTN_RESET_HOV= (90, 30, 30)
COLOR_BTN_RESET_BDR= (200, 60, 60)

COLOR_BTN_GEN      = (25, 50, 65)
COLOR_BTN_GEN_HOV  = (35, 75, 95)
COLOR_BTN_GEN_BDR  = (60, 160, 220)

COLOR_TEXT         = (200, 220, 255)
COLOR_LABEL        = (100, 120, 160)
COLOR_SLIDER_BG    = (25, 30, 50)
COLOR_SLIDER_FILL  = (60, 120, 220)
COLOR_SLIDER_KNOB  = (140, 180, 255)


class Button:
    """A simple clickable button."""

    def __init__(self, x, y, w, h, label,
                 color, hover_color, border_color):
        self.rect         = pygame.Rect(x, y, w, h)
        self.label        = label
        self.color        = color
        self.hover_color  = hover_color
        self.border_color = border_color
        self.hovered      = False
        self.font = pygame.font.SysFont("monospace", 11, bold=True)

    def handle_event(self, event):
        """Returns True if this button was clicked."""
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface):
        color = self.hover_color if self.hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=5)
        pygame.draw.rect(surface, self.border_color, self.rect, 1,
                         border_radius=5)
        surf = self.font.render(self.label, True, COLOR_TEXT)
        cx = self.rect.x + (self.rect.width  - surf.get_width())  // 2
        cy = self.rect.y + (self.rect.height - surf.get_height()) // 2
        surface.blit(surf, (cx, cy))


class Slider:
    """
    A horizontal slider that returns a value between min_val and max_val.

    Drag the knob left and right to change the value.
    """

    def __init__(self, x, y, w, h, label, min_val, max_val,
                 initial, integer=False):
        self.rect     = pygame.Rect(x, y, w, h)
        self.label    = label
        self.min_val  = min_val
        self.max_val  = max_val
        self.value    = initial
        self.integer  = integer   # if True, snap to whole numbers
        self.dragging = False
        self.font     = pygame.font.SysFont("monospace", 10)

    def handle_event(self, event):
        """Update value when user drags the knob. Returns True if value changed."""
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging = False

        if event.type == pygame.MOUSEMOTION and self.dragging:
            # Map mouse x position to slider value
            relative_x = event.pos[0] - self.rect.x
            fraction   = max(0.0, min(1.0, relative_x / self.rect.width))
            new_val    = self.min_val + fraction * (self.max_val - self.min_val)
            if self.integer:
                new_val = round(new_val)
            if new_val != self.value:
                self.value = new_val
                return True

        return False

    def draw(self, surface):
        # Track background
        pygame.draw.rect(surface, COLOR_SLIDER_BG, self.rect,
                         border_radius=3)

        # Filled portion
        fraction  = (self.value - self.min_val) / (self.max_val - self.min_val)
        fill_w    = int(fraction * self.rect.width)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y,
                                fill_w, self.rect.height)
        pygame.draw.rect(surface, COLOR_SLIDER_FILL, fill_rect,
                         border_radius=3)

        # Knob
        knob_x = self.rect.x + fill_w
        knob_y = self.rect.centery
        pygame.draw.circle(surface, COLOR_SLIDER_KNOB, (knob_x, knob_y), 7)

        # Label and value
        display_val = int(self.value) if self.integer else f"{self.value:.1f}"
        text = f"{self.label}: {display_val}"
        surf = self.font.render(text, True, COLOR_LABEL)
        surface.blit(surf, (self.rect.x, self.rect.y - 13))


class Controls:
    """
    The full toolbar — all buttons and sliders together.

    Call handle_event() with every pygame event.
    Check the returned action string to know what the user did.
    """

    def __init__(self, toolbar_rect):
        self.rect = toolbar_rect

        x  = toolbar_rect.x + 8
        y  = toolbar_rect.y
        h  = toolbar_rect.height
        bh = h - 10   # button height
        by = y + 5    # button y

        # ── Gate buttons ──────────────────────────────────────────
        gate_w      = 48
        gate_labels = ["AND", "OR", "NOT", "XOR", "NAND", "NOR"]
        self.gate_buttons = []
        for i, label in enumerate(gate_labels):
            btn = Button(x + i * (gate_w + 4), by, gate_w, bh,
                         label,
                         COLOR_BTN_GATE, COLOR_BTN_GATE_HOV, COLOR_BTN_GATE_BDR)
            self.gate_buttons.append(btn)

        # ── Separator position ────────────────────────────────────
        sep_x = x + len(gate_labels) * (gate_w + 4) + 8

        # ── Generate truth table button ───────────────────────────
        self.btn_generate = Button(
            sep_x, by, 80, bh, "GET TABLE",
            COLOR_BTN_GEN, COLOR_BTN_GEN_HOV, COLOR_BTN_GEN_BDR
        )
        sep_x += 88

        # ── Train / Pause / Reset buttons ─────────────────────────
        self.btn_train = Button(
            sep_x, by, 58, bh, "TRAIN",
            COLOR_BTN_TRAIN, COLOR_BTN_TRAIN_HOV, COLOR_BTN_TRAIN_BDR
        )
        self.btn_pause = Button(
            sep_x + 62, by, 58, bh, "PAUSE",
            COLOR_BTN_PAUSE, COLOR_BTN_PAUSE_HOV, COLOR_BTN_PAUSE_BDR
        )
        self.btn_reset = Button(
            sep_x + 124, by, 58, bh, "RESET",
            COLOR_BTN_RESET, COLOR_BTN_RESET_HOV, COLOR_BTN_RESET_BDR
        )
        sep_x += 190

        # ── Speed slider ──────────────────────────────────────────
        self.slider_speed = Slider(
            sep_x + 10, by + 16, 80, 8,
            "Speed", 1, 50, 5, integer=True
        )
        sep_x += 100

        # ── Architecture sliders ──────────────────────────────────
        self.slider_layers = Slider(
            sep_x + 10, by + 16, 70, 8,
            "Layers", 1, 3, 1, integer=True
        )
        self.slider_neurons = Slider(
            sep_x + 100, by + 16, 70, 8,
            "Neurons", 2, 16, 4, integer=True
        )

        # ── Add Input / Output node buttons ──────────────────────
        self.btn_add_input = Button(
            sep_x + 185, by, 58, bh, "+INPUT",
            COLOR_BTN_GATE, COLOR_BTN_GATE_HOV, COLOR_BTN_GATE_BDR
        )
        self.btn_add_output = Button(
            sep_x + 248, by, 62, bh, "+OUTPUT",
            COLOR_BTN_GATE, COLOR_BTN_GATE_HOV, COLOR_BTN_GATE_BDR
        )

    def handle_event(self, event):
        """
        Process one pygame event across all controls.

        Returns an action string describing what happened, or None.

        Possible return values:
            "add_AND", "add_OR", "add_NOT", "add_XOR", "add_NAND", "add_NOR"
            "generate_table"
            "train", "pause", "reset"
            "speed_changed"
            "architecture_changed"
            "add_input", "add_output"
            None  (nothing happened)
        """
        # Gate buttons
        for btn in self.gate_buttons:
            if btn.handle_event(event):
                return f"add_{btn.label}"

        # Generate table
        if self.btn_generate.handle_event(event):
            return "generate_table"

        # Train controls
        if self.btn_train.handle_event(event):
            return "train"
        if self.btn_pause.handle_event(event):
            return "pause"
        if self.btn_reset.handle_event(event):
            return "reset"

        # Speed slider
        if self.slider_speed.handle_event(event):
            return "speed_changed"

        # Architecture sliders
        arch_changed = self.slider_layers.handle_event(event)
        arch_changed = self.slider_neurons.handle_event(event) or arch_changed
        if arch_changed:
            return "architecture_changed"

        # Node buttons
        if self.btn_add_input.handle_event(event):
            return "add_input"
        if self.btn_add_output.handle_event(event):
            return "add_output"

        return None

    def get_speed(self):
        """Current steps per frame from the speed slider."""
        return int(self.slider_speed.value)

    def get_architecture(self):
        """
        Returns (hidden_layers, neurons_per_layer) from the sliders.
        """
        return (int(self.slider_layers.value),
                int(self.slider_neurons.value))

    def draw(self, surface):
        """Draw all controls onto the toolbar."""
        for btn in self.gate_buttons:
            btn.draw(surface)

        self.btn_generate.draw(surface)
        self.btn_train.draw(surface)
        self.btn_pause.draw(surface)
        self.btn_reset.draw(surface)

        self.slider_speed.draw(surface)
        self.slider_layers.draw(surface)
        self.slider_neurons.draw(surface)

        self.btn_add_input.draw(surface)
        self.btn_add_output.draw(surface)
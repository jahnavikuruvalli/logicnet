# ui/truth_table_panel.py
#
# Draws the middle panel — the live truth table.
#
# Each row shows:
#   - the input combination (e.g. 0 | 1)
#   - the correct expected output
#   - the network's current prediction
#   - a green background if correct, red if wrong
#
# This panel reads from the Trainer every frame and redraws.
# Watching red rows flip to green one by one is the most
# satisfying part of the whole app.


import pygame


# Colors
COLOR_BG          = (12, 14, 22)
COLOR_CORRECT     = (0, 60, 35)       # dark green row background
COLOR_WRONG       = (60, 15, 15)      # dark red row background
COLOR_CORRECT_TEXT = (0, 255, 136)    # bright green text
COLOR_WRONG_TEXT   = (255, 68, 68)    # bright red text
COLOR_HEADER_BG   = (20, 25, 45)
COLOR_HEADER_TEXT = (120, 160, 255)
COLOR_INPUT_TEXT  = (180, 200, 230)
COLOR_BORDER      = (35, 45, 70)
COLOR_DIVIDER     = (25, 32, 55)

ROW_HEIGHT    = 36
HEADER_HEIGHT = 40
PADDING       = 10


class TruthTablePanel:
    """
    Draws the live truth table in the middle panel.

    Reads prediction data from a Trainer instance every frame.
    """

    def __init__(self, rect):
        """
        rect : pygame.Rect defining this panel's screen area
        """
        self.rect = rect

        self.font_header = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_row    = pygame.font.SysFont("monospace", 13)
        self.font_title  = pygame.font.SysFont("monospace", 12, bold=True)
        self.font_status = pygame.font.SysFont("monospace", 12)

        # Trainer reference — set this before drawing
        self.trainer = None

    def set_trainer(self, trainer):
        """Connect this panel to a Trainer so it can read predictions."""
        self.trainer = trainer

    def draw(self, surface):
        """Draw the entire truth table panel."""

        # Panel background
        pygame.draw.rect(surface, COLOR_BG, self.rect)
        pygame.draw.rect(surface, COLOR_BORDER, self.rect, 1)

        # Panel title
        title = self.font_title.render("TRUTH TABLE", True, COLOR_HEADER_TEXT)
        surface.blit(title, (self.rect.x + PADDING, self.rect.y + 8))

        # If no trainer or no truth table yet, show a placeholder
        if self.trainer is None or not self.trainer.truth_table:
            self._draw_empty(surface)
            return

        predictions = self.trainer.get_all_predictions()
        if not predictions:
            self._draw_empty(surface)
            return

        n_inputs = len(predictions[0]["inputs"])

        # Starting y position for the table (below title)
        table_y = self.rect.y + 30

        # Draw column headers
        table_y = self._draw_header(surface, table_y, n_inputs)

        # Draw each row
        for i, row in enumerate(predictions):
            table_y = self._draw_row(surface, table_y, row, i)
            if table_y > self.rect.bottom - ROW_HEIGHT:
                break  # don't draw outside the panel

        # Draw accuracy and step count at the bottom
        self._draw_footer(surface)

    def _draw_header(self, surface, y, n_inputs):
        """Draw the column header row."""
        header_rect = pygame.Rect(self.rect.x, y, self.rect.width, HEADER_HEIGHT)
        pygame.draw.rect(surface, COLOR_HEADER_BG, header_rect)

        # Build column labels
        input_labels = [chr(65 + i) for i in range(n_inputs)]  # A, B, C...
        all_labels   = input_labels + ["Expected", "Got"]

        col_width = self.rect.width // len(all_labels)

        for i, label in enumerate(all_labels):
            surf = self.font_header.render(label, True, COLOR_HEADER_TEXT)
            cx = self.rect.x + i * col_width + col_width // 2 - surf.get_width() // 2
            cy = y + HEADER_HEIGHT // 2 - surf.get_height() // 2
            surface.blit(surf, (cx, cy))

        # Divider line below header
        pygame.draw.line(surface, COLOR_BORDER,
                         (self.rect.x, y + HEADER_HEIGHT),
                         (self.rect.right, y + HEADER_HEIGHT), 1)

        return y + HEADER_HEIGHT

    def _draw_row(self, surface, y, row, row_index):
        """
        Draw one truth table row.

        Correct rows get a green tint, wrong rows get red.
        """
        row_rect = pygame.Rect(self.rect.x, y, self.rect.width, ROW_HEIGHT)

        # Background color based on correctness
        bg_color = COLOR_CORRECT if row["correct"] else COLOR_WRONG
        pygame.draw.rect(surface, bg_color, row_rect)

        # Alternating slight shade for readability
        if row_index % 2 == 0:
            shade = pygame.Surface((self.rect.width, ROW_HEIGHT), pygame.SRCALPHA)
            shade.fill((255, 255, 255, 8))
            surface.blit(shade, (self.rect.x, y))

        # Text color
        text_color = COLOR_CORRECT_TEXT if row["correct"] else COLOR_WRONG_TEXT

        n_inputs   = len(row["inputs"])
        all_values = row["inputs"] + [row["expected"], row["predicted"]]
        col_width  = self.rect.width // (n_inputs + 2)

        for i, val in enumerate(all_values):
            # Input columns use normal text color
            # Expected and predicted columns use red/green
            if i < n_inputs:
                color = COLOR_INPUT_TEXT
            else:
                color = text_color

            text = str(val)
            surf = self.font_row.render(text, True, color)
            cx = self.rect.x + i * col_width + col_width // 2 - surf.get_width() // 2
            cy = y + ROW_HEIGHT // 2 - surf.get_height() // 2
            surface.blit(surf, (cx, cy))

        # Subtle divider between rows
        pygame.draw.line(surface, COLOR_DIVIDER,
                         (self.rect.x, y + ROW_HEIGHT),
                         (self.rect.right, y + ROW_HEIGHT), 1)

        return y + ROW_HEIGHT

    def _draw_footer(self, surface):
        """Draw accuracy and step count at the bottom of the panel."""
        if self.trainer is None:
            return

        footer_y = self.rect.bottom - 70

        # Divider above footer
        pygame.draw.line(surface, COLOR_BORDER,
                         (self.rect.x, footer_y),
                         (self.rect.right, footer_y), 1)

        # Accuracy
        accuracy_pct = int(self.trainer.current_accuracy * 100)
        acc_color = COLOR_CORRECT_TEXT if accuracy_pct == 100 else COLOR_WRONG_TEXT
        acc_text  = f"Accuracy: {accuracy_pct}%"
        acc_surf  = self.font_status.render(acc_text, True, acc_color)
        surface.blit(acc_surf, (self.rect.x + PADDING, footer_y + 10))

        # Step count
        step_text = f"Steps: {self.trainer.step_count:,}"
        step_surf = self.font_status.render(step_text, True, (120, 140, 180))
        surface.blit(step_surf, (self.rect.x + PADDING, footer_y + 28))

        # Status message
        if self.trainer.is_complete:
            msg   = "LEARNED ✓"
            color = COLOR_CORRECT_TEXT
        elif self.trainer.is_training:
            msg   = "TRAINING..."
            color = (180, 180, 80)
        elif self.trainer.step_count > 0:
            msg   = "PAUSED"
            color = (120, 140, 180)
        else:
            msg   = "READY"
            color = (100, 120, 160)

        msg_surf = self.font_status.render(msg, True, color)
        surface.blit(msg_surf,
                     (self.rect.right - msg_surf.get_width() - PADDING,
                      footer_y + 10))

    def _draw_empty(self, surface):
        """Show a placeholder when no circuit is connected yet."""
        lines = [
            "Build a circuit",
            "on the left,",
            "then hit Train.",
        ]
        font = pygame.font.SysFont("monospace", 13)
        y = self.rect.centery - len(lines) * 20 // 2
        for line in lines:
            surf = font.render(line, True, (60, 80, 110))
            x = self.rect.centerx - surf.get_width() // 2
            surface.blit(surf, (x, y))
            y += 22
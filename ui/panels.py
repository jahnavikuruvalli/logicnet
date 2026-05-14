# ui/panels.py
#
# Sets up the three-panel layout of the app.
#
# The window is divided into three vertical sections:
#
#   ┌─────────────────┬──────────────┬─────────────────┐
#   │                 │              │                  │
#   │  Circuit Board  │ Truth Table  │ Neural Network   │
#   │    (left)       │  (middle)    │    (right)       │
#   │                 │              │                  │
#   └─────────────────┴──────────────┴─────────────────┘
#
# Each panel is just a rectangle — a region of the screen.
# The other ui files draw into these rectangles.


import pygame


# ── Window settings ────────────────────────────────────────────────
WINDOW_WIDTH  = 1400
WINDOW_HEIGHT = 800
WINDOW_TITLE  = "LogicNet — Neural Network Logic Circuit Learner"
FPS           = 60

# ── Panel proportions (fractions of window width) ─────────────────
LEFT_FRACTION   = 0.38   # circuit board gets 38% of width
MIDDLE_FRACTION = 0.22   # truth table gets 22%
RIGHT_FRACTION  = 0.40   # network panel gets 40%

# ── Colors ────────────────────────────────────────────────────────
COLOR_BACKGROUND = (10, 10, 15)      # near-black
COLOR_PANEL_BG   = (12, 14, 22)      # slightly lighter panel bg
COLOR_BORDER     = (35, 45, 70)      # subtle panel border
COLOR_TEXT       = (180, 200, 230)   # general text
COLOR_TITLE      = (100, 160, 255)   # panel title text


class PanelLayout:
    """
    Owns the pygame window and defines the three panel rectangles.

    Other classes receive their panel rect and draw into it.
    """

    def __init__(self):
        pygame.init()
        pygame.display.set_caption(WINDOW_TITLE)

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock  = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("monospace", 13, bold=True)
        self.font_small = pygame.font.SysFont("monospace", 11)

        # Compute panel rectangles
        # A rect is (x, y, width, height)
        toolbar_height = 50   # strip at the top for buttons

        left_w   = int(WINDOW_WIDTH * LEFT_FRACTION)
        middle_w = int(WINDOW_WIDTH * MIDDLE_FRACTION)
        right_w  = WINDOW_WIDTH - left_w - middle_w

        panel_y = toolbar_height
        panel_h = WINDOW_HEIGHT - toolbar_height

        self.toolbar_rect = pygame.Rect(0, 0, WINDOW_WIDTH, toolbar_height)

        self.left_rect   = pygame.Rect(0,              panel_y, left_w,   panel_h)
        self.middle_rect = pygame.Rect(left_w,         panel_y, middle_w, panel_h)
        self.right_rect  = pygame.Rect(left_w+middle_w, panel_y, right_w,  panel_h)

    def begin_frame(self):
        """Clear the screen at the start of each frame."""
        self.screen.fill(COLOR_BACKGROUND)

    def draw_panel_borders(self):
        """Draw subtle borders between panels."""
        pygame.draw.rect(self.screen, COLOR_BORDER, self.left_rect,   1)
        pygame.draw.rect(self.screen, COLOR_BORDER, self.middle_rect, 1)
        pygame.draw.rect(self.screen, COLOR_BORDER, self.right_rect,  1)

    def draw_toolbar_bg(self):
        """Draw the toolbar background."""
        pygame.draw.rect(self.screen, (15, 18, 30), self.toolbar_rect)
        pygame.draw.line(self.screen, COLOR_BORDER,
                         (0, self.toolbar_rect.bottom),
                         (WINDOW_WIDTH, self.toolbar_rect.bottom), 1)

    def draw_panel_title(self, text, rect, color=COLOR_TITLE):
        """Draw a small label at the top of a panel."""
        surf = self.font_title.render(text, True, color)
        self.screen.blit(surf, (rect.x + 10, rect.y + 6))

    def end_frame(self):
        """Flip the display buffer and tick the clock."""
        pygame.display.flip()
        self.clock.tick(FPS)

    def get_screen(self):
        return self.screen
"""
Author: Martin Zůbek
Date: 2026-04-10
File: constants.py
Description: This file contains the constants used in the gui module.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

# Minimum width of the application window in pixels
WINDOW_MIN_WIDTH: int = 700
# Minimum height of the application window in pixels
WINDOW_MIN_HEIGHT: int = 600

# Spacing between widgets in the main layout in pixels
LAYOUT_SPACING: int = 16
# Margin on each side of the main layout in pixels
LAYOUT_MARGIN: int = 30

# Minimum font size of the title label in pixels
TITLE_FONT_MIN_SIZE: int = 18
# Maximum font size of the title label in pixels
TITLE_FONT_MAX_SIZE: int = 48
# Divisor used to derive the title font size from the window width
TITLE_FONT_WIDTH_DIVISOR: int = 28

# Maximum height of the identity document image preview in pixels
MAX_PREVIEW_HEIGHT: int = 400

# Horizontal margin subtracted from the scroll-area viewport width when computing
# the available width for image previews
PREVIEW_HORIZONTAL_MARGIN: int = 76

# Height of the result header drawn above the fraud map preview in pixels
FRAUD_MAP_HEADER_HEIGHT: int = 56

# Font size used in the fraud map result header
FRAUD_MAP_HEADER_FONT_SIZE: int = 20

# Path to the bold TrueType font used in the fraud map result header
FRAUD_MAP_HEADER_FONT_PATH: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Background colour of the fraud map header area as an RGB tuple
FRAUD_MAP_HEADER_BACKGROUND: tuple[int, int, int] = (254, 242, 242)

# Red channel value of the fraud map overlay colour
FRAUD_MAP_OVERLAY_RED: int = 220
# Green channel value of the fraud map overlay colour
FRAUD_MAP_OVERLAY_GREEN: int = 30
# Blue channel value of the fraud map overlay colour
FRAUD_MAP_OVERLAY_BLUE: int = 30
# Maximum alpha value of the fraud map overlay
FRAUD_MAP_OVERLAY_MAX_ALPHA: int = 190

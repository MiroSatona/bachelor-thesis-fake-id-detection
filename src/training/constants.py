"""
Author: Martin Zůbek
Date: 2026-04-03
File: constants.py
Description: This file contains constants used for the training utilities.
"""

# String constants for metric names
LOSS: str = "loss"
ACCURACY: str = "accuracy"
PRECISION: str = "precision"
RECALL: str = "recall"
F1_SCORE: str = "f1_score"

# List of all supported metrics
METRICS: list[str] = [LOSS, ACCURACY, PRECISION, RECALL, F1_SCORE]
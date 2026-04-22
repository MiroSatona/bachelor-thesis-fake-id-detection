"""
Author: Martin Zůbek
Date: 2026-04-03
File: constants.py
Description: This file contains constants used for the training utilities.
"""

from pathlib import Path

# String constants for metric names
LOSS: str = "loss"
ACCURACY: str = "accuracy"
PRECISION: str = "precision"
RECALL: str = "recall"
F1_SCORE: str = "f1_score"

# String constants for confusion matrix components
TP: str = "TP"
TN: str = "TN"
FP: str = "FP"
FN: str = "FN"

# List of all supported metrics
METRICS: list[str] = [LOSS, ACCURACY, PRECISION, RECALL, F1_SCORE]

# Titles for training and validation phases
TRAIN_TITLE: str = "train"
VALIDATION_TITLE: str = "validation"

# Name for the directory where all experiments are stored
EXPERIMENTS_DIR: str = "experiments"
# Path to the directory where all experiments are stored
EXPERIMENTS_PATH: Path = (
    Path(__file__).parent.parent.parent / EXPERIMENTS_DIR
)
# Names for subdirectories and files within each experiment
GRAPHS_DIR: str = "graphs"
CHECKPOINTS_DIR: str = "checkpoints"
TEST_DIR: str = "test"

# Names for configuration, metrics and summary files and checkpoint file
CONFIG_FILE_NAME: str = "config.json"
METRICS_FILE_NAME: str = "metrics.json"
SUMMARY_FILE_NAME: str = "summary.json"
CHECKPOINT_FILE_NAME: str = "checkpoint.pth"

# Files from the test phase
TEST_METRICS_FILE_NAME: str = "test_metrics.json"
TEST_CONFUSION_MATRIX_FILE_VALUES_NAME: str = "test_confusion_matrix_values.json"
TEST_CONFUSION_MATRIX_FILE_MATRIX_NAME: str = "test_confusion_matrix.png"

# Constants for early stopping
EARLY_STOPPING_PATIENCE: int = 5
EARLY_STOPPING_DELTA: float = 0.001 

# Constants for learning rate optimization
OPTIM_PATIENCE: int = 2
OPTIM_FACTOR: float = 0.1
OPTIM_MIN_LR: float = 1e-6

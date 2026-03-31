"""
Author: Martin Zůbek
Date: 2026-03-28
File: constants.py
Description: This file contains constant values for the proccessing provided data for training or 
             for application.
"""

# Subset of the provided dataset with positive samples
POSITIVE_SUBSET: str = "positive"
# Subset of the provided dataset with negative samples
ATTACK_SUBSET: str = "attack"

# Supproted image extensisons for the provided image of the identity document
IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Percentage of the dataset to be used for training
TRAIN_PERCENTAGE: float = 0.7
# Percentage of the dataset to be used for validation
VAL_PERCENTAGE: float = 0.15
# Percentage of the dataset to be used for testing
TEST_PERCENTAGE: float = 0.15

# Seed for random splitting provided dataset into training, validation and testing sets
DEFAULT_SPLIT_SEED: int = 28
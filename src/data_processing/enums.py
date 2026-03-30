"""
Author: Martin Zůbek
Date: 2026-03-28
File: enums.py
Description: This file contains enum values for the proccessing provided data for training or for application.
"""

from enum import Enum, IntEnum

class Label(IntEnum):
    """
    Enum class for labels in the internal representation of the dataset, indicating which identity document is real and which is fake.

    Attributes:    
        REAL = 0 - Positive/real identity document
        FAKE = 1 - Fake/edited identity document
    """
    REAL: int = 0
    FAKE: int = 1

class AttackType(str, Enum):
    """
    Enum class for attack types in the internal representation of the dataset, indicating which type of attack is present in the provided image of the identity document.
    Attributes:
        BOTH = "both" - Attack with changed photo and text
        PHOTO_ONLY = "photo_only" - Attack with changed photo only
        TEXT_ONLY = "text_only" - Attack with changed text only
    """
    BOTH: str = "both"
    PHOTO_ONLY: str = "photo_only"
    TEXT_ONLY: str = "text_only"

class DatasetSubset(str, Enum):
    """
    Enum class for dataset subsets in the internal representation of the dataset, indicating which subset of the dataset is being used for training, validation, or testing.
    Attributes:
         TRAIN = "train" - Subset of the dataset used for training the model
         VAL = "val" - Subset of the dataset used for validating the model during training
         TEST = "test" - Subset of the dataset used for testing after training
    """
    TRAIN: str = "train"
    VAL: str = "val"
    TEST: str = "test"
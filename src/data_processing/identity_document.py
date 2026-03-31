"""
Author: Martin Zůbek
Date: 2026-03-28
File: identity_document.py
Description: This file contains the definition of the IdentityDocumentImage dataclass which is used 
for internal representation of the every image of the identity document in the dataset 
or for the application.
"""

from dataclasses import dataclass
from pathlib import Path

from src.data_processing.enums import AttackType, DatasetSubset, Label


@dataclass(frozen=True)
class IdentityDocumentImage:
    """
    Dataclass for internal representation of the every image of the identity document in the dataset
    or for the application.
    
    Attributes:
        path (Path): The path to the image file.
        label (Label | None): The label indicating whether the document is real or fake.
        attack_type (AttackType | None): The type of attack present in the image.
        dataset_part (str | None): The part of the dataset the image belongs to.
        dataset_subset (DatasetSubset | None): The subset of the dataset the image belongs to.
    """
    path: Path
    label: Label | None
    attack_type: AttackType | None
    dataset_part: str | None
    dataset_subset: DatasetSubset | None
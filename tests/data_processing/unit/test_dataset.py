"""
Atuhor: Martin Zůbek
Date: 2026-03-28
File: test_dataset.py
Description: Unit tests for the Dataset class.
"""

from pathlib import Path

from src.data_processing.dataset import Dataset
from src.data_processing.enums import Label
from src.data_processing.identity_document import IdentityDocumentImage


def create_identity_document(path_str: str) -> IdentityDocumentImage:
    """
    Function for creating an identity document with given path string for testing.

    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.
        
    Returns:
        None
    """
    return IdentityDocumentImage(
        path=Path(path_str),
        label=Label.REAL,
        attack_type=None,
        dataset_part="midv_2020",
        dataset_subset=None,
    )


def test_dataset_can_be_created_correctly() -> None:
    """
    Test for checking that Dataset can be created correctly with given training, validation, 
    and testing data.
    
    Arguments:
        None
    
    Returns:
        None
    """
    dataset: Dataset = Dataset(
        train_data=[create_identity_document("train1.png"), create_identity_document("train2.jpg")],
        val_data=[create_identity_document("val1.tiff")],
        test_data=[create_identity_document("test1.webp"), create_identity_document("test2.jpg")],
    )

    assert dataset.dataset_len == 5
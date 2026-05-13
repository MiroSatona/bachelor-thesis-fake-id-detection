"""
Author: Martin Zůbek
Date: 2026-03-28
File: test_data_handler_integration.py
Description: Integration tests for the DataHandler class.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

from pathlib import Path

from src.data_processing.data_handler import DataHandler
from src.data_processing.dataset import Dataset, IdentityDocumentImage


def test_data_handler_loads_and_splits_dataset(tmp_path: Path) -> None:
    """
    Test for checking that DataHandler can load and split the dataset correctly.

    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.

    Returns:
        None
    """

    positive_dataset_part: Path = tmp_path / "positive" / "midv_2020"
    positive_dataset_part.mkdir(parents=True)

    attack_dataset_part_text_directory: Path = tmp_path / "attack" / "midv_2020" / "text_only"
    attack_dataset_part_photo_directory: Path = tmp_path / "attack" / "midv_2020" / "photo_only"
    attack_dataset_part_both_directory: Path = tmp_path / "attack" / "midv_2020" / "both"

    attack_dataset_part_text_directory.mkdir(parents=True)
    attack_dataset_part_photo_directory.mkdir(parents=True)
    attack_dataset_part_both_directory.mkdir(parents=True)

    for name in range(10):
        (positive_dataset_part / f"real_{name}.jpg").write_text("img")
        (attack_dataset_part_text_directory / f"text_{name}.jpg").write_text("img")
        (attack_dataset_part_photo_directory / f"photo_{name}.jpg").write_text("img")
        (attack_dataset_part_both_directory / f"both_{name}.jpg").write_text("img")

    handler: DataHandler = DataHandler(tmp_path)
    handler.load_dataset()
    dataset: Dataset = handler.get_dataset_for_training()

    assert handler.identity_documents is not None
    assert isinstance(dataset, Dataset)
    assert dataset.dataset_len == 40


def test_data_handler_load_single_identity_document(tmp_path: Path) -> None:
    """
    Test for checking that DataHandler can load a single identity document correctly.

    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.

    Returns:
        None
    """
    
    identity_document_path: Path = tmp_path / "single.jpg"
    identity_document_path.write_text("single image")

    handler: DataHandler  = DataHandler(identity_document_path)
    identity_document: IdentityDocumentImage = ( 
        handler.load_single_identity_document()
    )
    

    assert identity_document.path == identity_document_path
    assert identity_document.label is None
    assert identity_document.attack_type is None
    assert identity_document.dataset_part is None
    assert identity_document.dataset_subset is None
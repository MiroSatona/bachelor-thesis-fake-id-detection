"""
Author: Martin Zůbek
Date: 2026-03-28
File: test_data_splitter.py
Description: Unit tests for the DatasetSplitter class for splitting identity documents 
of the dataset.
"""

from pathlib import Path

import pytest

from src.data_processing.data_splitter import DatasetSplitter
from src.data_processing.dataset import Dataset
from src.data_processing.enums import AttackType, Label
from src.data_processing.identity_document import IdentityDocumentImage


def create_positive_identity_document(index: int) -> IdentityDocumentImage:
    """
    Function for creating a positive identity document with given index for testing.
    
    Arguments:
        index (int): The index of the identity document to create.
        
    Returns:
        IdentityDocumentImage: The created positive identity document.
    """
    return IdentityDocumentImage(
        path=Path(f"real_{index}.jpg"),
        label=Label.REAL,
        attack_type=None,
        dataset_part="midv_2020",
        dataset_subset=None,
    )


def create_attack_identity_document(index: int, attack_type: AttackType) -> IdentityDocumentImage:
    """
    Function for creating an attack identity document with given index and attack type for testing.
    
    Arguments:
        index (int): The index of the identity document to create.
        attack_type (AttackType): The type of the attack.
        
    Returns:
        IdentityDocumentImage: The created attack identity document.
    """
    return IdentityDocumentImage(
        path=Path(f"{attack_type.value}_{index}.jpg"),
        label=Label.FAKE,
        attack_type=attack_type,
        dataset_part="midv_2020",
        dataset_subset=None,
    )


def create_valid_list_with_identity_documents() -> list[IdentityDocumentImage]:
    """
    Function for creating a valid list with identity documents <=> dataset containing 10 positive 
    and 30 attack identity documents for testing.
    
    Arguments:
        None
    
    Returns:
        list[IdentityDocumentImage]: The created list with identity documents.
    """
    documents: list[IdentityDocumentImage] = []
    documents.extend(create_positive_identity_document(i) for i in range(10))
    documents.extend(create_attack_identity_document(i, AttackType.TEXT_ONLY) for i in range(10))
    documents.extend(create_attack_identity_document(i, AttackType.PHOTO_ONLY) for i in range(10))
    documents.extend(create_attack_identity_document(i, AttackType.BOTH) for i in range(10))

    return documents

def test_grouped_identity_documents_valid() -> None:
    """
    Test for grouping identity documents into groups based on their labels and attack types.

    Arguments:
        None

    Returns:
        None
    """
    splitter: DatasetSplitter = DatasetSplitter()
    identity_documents_list: list[IdentityDocumentImage] = (
        create_valid_list_with_identity_documents()
    )

    grouped_identity_documents: dict[str, list[IdentityDocumentImage]] = (
        splitter.grouped_identity_documents(identity_documents_list)
    )

    assert len(grouped_identity_documents) == 4
    assert len(grouped_identity_documents["positive"]) == 10
    assert len(grouped_identity_documents["attack_text"]) == 10
    assert len(grouped_identity_documents["attack_photo"]) == 10
    assert len(grouped_identity_documents["attack_both"]) == 10


def test_validate_grouped_identity_documents() -> None:
    """
    Test for validating the grouped identity documents with all required groups present.
    
    Arguments:
        None
        
    Returns:
        None
    """
    splitter: DatasetSplitter = DatasetSplitter()
    grouped: dict[str, list[IdentityDocumentImage]] = splitter.grouped_identity_documents(
        create_valid_list_with_identity_documents()
    )
    splitter.validate_grouped_identity_documents(grouped)


def test_validate_grouped_identity_documents_missing_group() -> None:
    """
    Test for validating grouped identity documents with a missing required group,
    which should raise a ValueError.
    
    Arguments:
        None
    
    Returns:
        None
    """
    splitter: DatasetSplitter = DatasetSplitter()
    documents: list[IdentityDocumentImage] = create_valid_list_with_identity_documents()
    identity_documents: list[IdentityDocumentImage] = [
        identity_document
        for identity_document in documents
        if identity_document.attack_type != AttackType.BOTH
    ]
    grouped_identity_documents: dict[str, list[IdentityDocumentImage]] = (
        splitter.grouped_identity_documents(identity_documents)
    )

    with pytest.raises(ValueError):
        splitter.validate_grouped_identity_documents(grouped_identity_documents)


def test_split_identity_documents_for_dataset() -> None:
    """
    Test for splitting identity documents into training, validation, and test sets for the dataset.

    Arguments:
        None

    Returns:
        None
    """
    splitter: DatasetSplitter = DatasetSplitter()
    identity_documents: list[IdentityDocumentImage] = create_valid_list_with_identity_documents()

    dataset: Dataset = splitter.split_identity_documents_for_dataset(identity_documents)
    dataset_size: int = (len(dataset.train_data) + len(dataset.val_data) + len(dataset.test_data))

    assert dataset_size == len(identity_documents)
    assert dataset.dataset_len == len(identity_documents)


def test_split_identity_documents_for_dataset_deterministic() -> None:
    """
    Test for checking that splitting identity documents into training, validation,
    and test sets for the dataset is deterministic.

    Arguments:
        None

    Returns:
        None
    """
    splitter: DatasetSplitter = DatasetSplitter()
    identity_documents: list[IdentityDocumentImage] = create_valid_list_with_identity_documents()

    first_dataset: Dataset = splitter.split_identity_documents_for_dataset(identity_documents)
    second_dataset: Dataset = splitter.split_identity_documents_for_dataset(identity_documents)

    assert [identity_document.path for identity_document in first_dataset.train_data] == [
        identity_document.path for identity_document in second_dataset.train_data
    ]
    assert [identity_document.path for identity_document in first_dataset.val_data] == [
        identity_document.path for identity_document in second_dataset.val_data
    ]
    assert [identity_document.path for identity_document in first_dataset.test_data] == [
        identity_document.path for identity_document in second_dataset.test_data
    ]
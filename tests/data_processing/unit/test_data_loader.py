"""
Author: Martin Zůbek
Date: 2026-03-28
File: test_data_loader.py
Description: Unit tests for the DataLoader class for loading the dataset of identity documents 
or single identity document images.
"""

from pathlib import Path

import pytest

from src.data_processing.data_loader import DataLoader
from src.data_processing.enums import AttackType, Label
from src.data_processing.identity_document import IdentityDocumentImage


def test_load_single_identity_document(tmp_path: Path) -> None:
    """
    Test for loading single identity document image and creating an IdentityDocumentImage instance 
    for its internal representation.
    
    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.
        
        
    Returns:
        None
    """
    identity_document_path: Path  = tmp_path / "identity_document.jpg"
    identity_document_path.write_text("id")

    loader: DataLoader = DataLoader(identity_document_path) 
    loaded_identity_document: IdentityDocumentImage = loader.load_single_identity_document()

    assert loaded_identity_document.path == identity_document_path
    assert loaded_identity_document.label is None
    assert loaded_identity_document.attack_type is None
    assert loaded_identity_document.dataset_part is None
    assert loaded_identity_document.dataset_subset is None

def test_load_not_existing_identity_document(tmp_path: Path) -> None:
    """
    Test for loading non-existing identity document image and raising FileNotFoundError.
    
    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.
        
    Returns:
        None
    """
    identity_document_path: Path  = tmp_path / "non_existent.jpg"
    loader: DataLoader = DataLoader(identity_document_path) 

    with pytest.raises(FileNotFoundError):
        loader.load_single_identity_document()

def test_load_unsupported_image_extension(tmp_path: Path) -> None:
    """
    Test for loading identity document image with unsupported image extension
    and raising ValueError.

    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.

    Returns:
        None
    """
    unsupported_image_extension_path: Path = tmp_path / "document.txt"
    unsupported_image_extension_path.write_text("not an image")

    loader: DataLoader = DataLoader(unsupported_image_extension_path)
    with pytest.raises(ValueError):
        loader.load_single_identity_document()

def test_load_dataset_with_missing_subsets(tmp_path: Path) -> None:
    loader: DataLoader = DataLoader(tmp_path)

    with pytest.raises(FileNotFoundError):
        loader.load_dataset()

def test_load_positive_subset_only_images(tmp_path: Path) -> None:
    """
    Test for loading only images from the positive subset of the dataset and creating 
    IdentityDocumentImage instances for their internal representation.

    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.

    Returns:
        None
    """
    positive_subset: Path = tmp_path / "positive"
    dataset_part_positive: Path = positive_subset / "midv_2020"
    dataset_part_positive.mkdir(parents=True)

    attack_subset: Path = tmp_path / "attack"
    dataset_part_attack: Path = attack_subset / "midv_2020"
    dataset_part_attack.mkdir(parents=True)

    (dataset_part_positive/"real1.jpg").write_text("real1")
    (dataset_part_positive/"real2.png").write_text("real2")
    (dataset_part_positive/"real3.tiff").write_text("real3")
    (dataset_part_positive/"fake_image.txt").write_text("fake_image")

    loader: DataLoader = DataLoader(tmp_path)
    identity_documents: list[IdentityDocumentImage] = loader.load_dataset()

    assert len(identity_documents) == 3
    assert all(document.label == Label.REAL for document in identity_documents)
    assert all(document.attack_type is None for document in identity_documents)
    assert all(document.dataset_part == "midv_2020" for document in identity_documents)


def test_load_only_supported_attack_types(tmp_path: Path) -> None:
    """
    Test for loading only images with supported attack types from the attack subset of the dataset 
    and creating IdentityDocumentImage instances for their internal representation.
    
    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.
        
    Returns:
        None
    """
    positive_subset: Path = tmp_path / "positive"
    dataset_part_positive: Path = positive_subset / "midv_2020"
    dataset_part_positive.mkdir(parents=True)

    attack_subset: Path = tmp_path / "attack"
    dataset_part_attack: Path = attack_subset / "midv_2020"
    dataset_part_attack.mkdir(parents=True)

    text_only_dir: Path = dataset_part_attack / "text_only"
    photo_only_dir: Path = dataset_part_attack / "photo_only"
    both_dir: Path = dataset_part_attack / "both"
    unknown_dir: Path = dataset_part_attack / "unknown"

    text_only_dir.mkdir(parents=True)
    photo_only_dir.mkdir(parents=True)
    both_dir.mkdir(parents=True)
    unknown_dir.mkdir(parents=True)

    (text_only_dir / "img1.jpg").write_text("img")
    (photo_only_dir / "img2.png").write_text("img")
    (both_dir / "img3.webp").write_text("img")
    (unknown_dir / "img4.jpg").write_text("img")
    (text_only_dir / "fake_image.txt").write_text("fake image")

    loader: DataLoader = DataLoader(tmp_path)
    identity_documents: list[IdentityDocumentImage] = loader.load_dataset()

    assert len(identity_documents) == 3

    attack_types = {identity_document.attack_type for identity_document in identity_documents}
    assert attack_types == {
        AttackType.TEXT_ONLY,
        AttackType.PHOTO_ONLY,
        AttackType.BOTH,
    }

    assert all(identity_document.label == Label.FAKE for identity_document in identity_documents)
    assert all(identity_document.dataset_part == "midv_2020" 
        for identity_document in identity_documents
    )

def test_load_all_dataset(tmp_path: Path) -> None:
    """
    Test for loading all dataset with both positive and attack identity document images and creating
    IdentityDocumentImage instances for their internal representation.

    Arguments:
        tmp_path (Path): Temporary directory provided by pytest for creating test files.

    Returns:
        None
    """
    positive_subset: Path = tmp_path / "positive"
    dataset_part_positive: Path = positive_subset / "midv_2020"
    dataset_part_positive.mkdir(parents=True)

    (dataset_part_positive/"real1.jpg").write_text("real1")
    (dataset_part_positive/"real2.png").write_text("real2")

    attack_subset: Path = tmp_path / "attack"
    dataset_part_attack: Path = attack_subset / "midv_2020"
    dataset_part_attack.mkdir(parents=True)

    text_only_dir: Path = dataset_part_attack / "text_only"
    photo_only_dir: Path = dataset_part_attack / "photo_only"
    both_dir: Path = dataset_part_attack / "both"

    text_only_dir.mkdir(parents=True)
    photo_only_dir.mkdir(parents=True)
    both_dir.mkdir(parents=True)

    (text_only_dir / "img1.jpg").write_text("img")
    (photo_only_dir / "img2.png").write_text("img")
    (both_dir / "img3.webp").write_text("img")

    loader: DataLoader = DataLoader(tmp_path)
    identity_documents: list[IdentityDocumentImage] = loader.load_dataset()

    assert len(identity_documents) == 5
    assert sum(document.label == Label.REAL for document in identity_documents) == 2
    assert sum(document.label == Label.FAKE for document in identity_documents) == 3
    assert all(document.dataset_part == "midv_2020" for document in identity_documents)
    attack_types: set[AttackType] = {
        document.attack_type 
        for document in identity_documents 
            if document.label == Label.FAKE and document.attack_type is not None
    }
    assert attack_types == {
        AttackType.TEXT_ONLY,
        AttackType.PHOTO_ONLY,
        AttackType.BOTH,
    }


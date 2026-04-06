"""
Author: Martin Zůbek
Date: 2026-03-28
File: test_data_utils.py
Description: Unit tests for the DataUtils class for utility functions.
"""

from pathlib import Path

from src.data_processing.data_utils import DataUtils
from src.data_processing.enums import AttackType


def test_is_image_returns_true_for_supported_image(tmp_path: Path) -> None:
    """
    Test for checking that is_image function returns True for supported image file.
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.
        
    Returns:
        None
    """

    image_path: Path = tmp_path / "image.jpg"
    image_path.write_text("image")

    assert DataUtils.is_image(image_path) is True


def test_is_image_returns_false_for_unsupported_file(tmp_path: Path) -> None:
    """
    Test for checking that is_image function returns False for unsupported file.
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.
        
    Returns:
        None
    """
    unsupported_path: Path = tmp_path / "unsupported.txt"
    unsupported_path.write_text("unsupported")

    assert DataUtils.is_image(unsupported_path) is False


def test_is_image_returns_false_for_directory(tmp_path: Path) -> None:
    """
    Test for checking that is_image function returns False for a directory.
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for creating test files.
        
    
    Returns:
        None
    """
    directory = tmp_path / "directory"
    directory.mkdir()

    assert DataUtils.is_image(directory) is False


def test_parse_attack_type_both_for_exact_name() -> None:
    """
    Test for checking that parse_attack_type function returns AttackType.BOTH for exact name "both".
    
    Arguments:
        None
    
    Returns:
        None
    """
    assert DataUtils.parse_attack_type("both") == AttackType.BOTH


def test_parse_attack_type_both_for_prefixed_name() -> None:
    """
    Test for checking that parse_attack_type function returns AttackType.BOTH 
    for prefixed name "both_v1".
    
    Arguments:
        None
    
    Returns:
        None
    """
    assert DataUtils.parse_attack_type("both_v1") == AttackType.BOTH


def test_parse_attack_type_photo_only() -> None:
    """
    Test for checking that parse_attack_type function returns AttackType.PHOTO_ONLY 
    for name "photo_only".
    
    Arguments:
        None
    
    Returns:
        None
    """
    assert DataUtils.parse_attack_type("photo_only") == AttackType.PHOTO_ONLY


def test_parse_attack_type_text_only() -> None:
    """
    Test for checking that parse_attack_type function returns AttackType.TEXT_ONLY 
    for name "text_only".
    
    Arguments:
        None
    
    Returns:
        None
    """
    assert DataUtils.parse_attack_type("text_only") == AttackType.TEXT_ONLY


def test_parse_attack_type_returns_none_for_unsupported_value() -> None:
    """
    Test for checking that parse_attack_type function returns None for unsupported value.
    
    Arguments:
        None
    
    Returns:
        None
    """
    assert DataUtils.parse_attack_type("unsupported_attack") is None
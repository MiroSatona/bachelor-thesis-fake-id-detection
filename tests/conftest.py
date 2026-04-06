"""
Author: Martin Zůbek
Date: 2026-03-28
File: conftest.py
Description: This file is used for pytest configuration of Python import paths.
"""

import sys
from pathlib import Path


def add_to_syspath(path: Path) -> None:
    """
    Function for adding the given path to the Python import paths.

    Arguments:
        path (Path): The path to be added to the Python import paths.

    Returns:
        None
    """
    path_str = str(path.resolve())
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


################ DATA PROCESSING TESTS ################
ROOT: Path = Path(__file__).resolve().parent.parent
add_to_syspath(ROOT)

################ EXTRACTORS TESTS #####################
EXTERNAL_ROOT: Path = ROOT / "external"
TRUFOR_ROOT: Path = EXTERNAL_ROOT / "TruFor" / "TruFor_train_test"
IML_VIT_ROOT: Path = EXTERNAL_ROOT / "IML_VIT"
MMFUSION_ROOT: Path = EXTERNAL_ROOT / "MMFusion"
SWINT_ROOT: Path = EXTERNAL_ROOT / "SwinT" / "models"

if TRUFOR_ROOT.exists():
    add_to_syspath(TRUFOR_ROOT)

if IML_VIT_ROOT.exists():
    add_to_syspath(IML_VIT_ROOT)

if MMFUSION_ROOT.exists():
    add_to_syspath(MMFUSION_ROOT)

if SWINT_ROOT.exists():
    add_to_syspath(SWINT_ROOT)
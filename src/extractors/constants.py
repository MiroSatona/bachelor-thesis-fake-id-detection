"""
Author: Martin Zůbek
Date: 2026-03-28
File: constants.py
Description: This file contains the constants for the Extractor classes.
"""
from pathlib import Path

# Default flag for using GPU for feature extraction
DEFAULT_USE_GPU: bool = True
# String for the directory with all the code for extractors
EXTRACTORS_DIR_STR: str = "external"

####################### TruFor specific constants #######################
# String for the directory with the code for TruFor extractor
TRUFOR_NAME: str = "TruFor"
# Name of the directory with the source code of the TruFor extractor
TRUFOR_SRC: str = "TruFor_train_test"
# Path to TruFor configuration file trufor_ph3.yaml
TRUFOR_CONFIG: Path = (
	Path(__file__).resolve().parents[2]
	/ EXTRACTORS_DIR_STR
	/ TRUFOR_NAME
	/ TRUFOR_SRC
	/ "lib"
	/ "config"
	/ "trufor_ph3.yaml"
)

###################### IML-Vit specific constants #######################
IML_VIT_NAME: str = "IML_VIT"
IML_VIT_INPUT_SIZE: int = 1024
IML_VIT_CONFIG: Path = (
	Path(__file__).resolve().parents[2]
	/ EXTRACTORS_DIR_STR
	/ IML_VIT_NAME
	/ "checkpoints" 
    / "iml_vit.pth"
)
IML_VIT_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IML_VIT_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

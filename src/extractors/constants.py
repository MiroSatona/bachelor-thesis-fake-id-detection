"""
Author: Martin Zůbek
Date: 2026-03-28
File: constants.py
Description: This file contains the constants for the Extractor classes.
"""
from pathlib import Path

# String for use CPU for feature extraction
CPU: str = "cpu"
# String for using GPU for feature extraction
CUDA: str = "cuda"
# Default flag for using GPU for feature extraction
DEFAULT_USE_GPU: bool = True
# Name of the directory with all the code for extractors
EXTRACTORS_DIR_STR: str = "external"

# ImageNet mean for normalizing input images 
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
# ImageNet standard deviation for normalizing input images
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

####################### TruFor specific constants #######################
# Name of the directory with the code for TruFor extractor
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
# Name of the directory with the code for IML-ViT extractor
IML_VIT_NAME: str = "IML_VIT"
# Input size for the IML-ViT model
IML_VIT_INPUT_SIZE: int = 1024
# Path to the  trained IML-ViT checkpoint
IML_VIT_CHECKPOINT: Path = (
	Path(__file__).resolve().parents[2]
	/ EXTRACTORS_DIR_STR
	/ IML_VIT_NAME
	/ "checkpoints"
	/ "iml-vit_checkpoint.pth"
)

#################### MMFusion specific constants ######################
# Name of the directory with the code for MMFusion extractor
MMFUSION_NAME: str = "MMFusion"
# Name of the yaml config file for MMFusion
MMFUSION_CONFIG: Path = (
    Path(__file__).resolve().parents[2]
    / EXTRACTORS_DIR_STR
    / MMFUSION_NAME
    / "experiments"
    / "ec_example_phase2.yaml"
)
# Path to the trained MMFusion checkpoint
MMFUSION_CHECKPOINT: Path = (
    Path(__file__).resolve().parents[2]
    / EXTRACTORS_DIR_STR
    / MMFUSION_NAME
    / "ckpt"
    / "early_fusion_detection.pth"
)    
# Path to the pretrained Noiseprint++ weights
MMFUSION_NP_WEIGHTS: Path = (
    Path(__file__).resolve().parents[2]
    / EXTRACTORS_DIR_STR
    / MMFUSION_NAME
    / "pretrained"
    / "noiseprint"
    / "np++.pth"
)

####################### SwinT specific constants #######################
# Name of the directory with the code for SwinT extractor
SWINT_NAME: str = "SwinT"
# Name of the directory with the models of the SwinT extractor  
SWINT_DIR_MODELS: str = "models"
# Path to the yaml config file for SwinT
SWINT_CONFIG: Path = (
    Path(__file__).resolve().parents[2]
    / EXTRACTORS_DIR_STR
    / SWINT_NAME
    / "configs"
    / "swin"
    / "swin_tiny_patch4_window7_224.yaml"
)
# Path to the pretrained SwinT checkpoint
SWINT_PRETRAINED_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / EXTRACTORS_DIR_STR
    / SWINT_NAME
    / "pretrained"
    / "swin_tiny_patch4_window7_224.pth"
)

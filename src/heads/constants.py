"""
Author: Martin Zůbek
Date: 2026-04-01
File: constants.py
Description: This file contains the constants used in the heads module.
"""

# Count of classes for classification. Default is two for this project, 
# positive identity document or fraudulent identity document.
CLASSES_COUNT: int = 2
# Dropout percentage for the MLP head.
DROPOUT_PRECENTAGE: float = 0.2

# Key for the TruFor extractor.
TRUFOR_KEY: str = "trufor"
# Key for the Swin Transformer extractor.
SWINT_KEY: str = "swint"
# Key for the MMFusion extractor.    
MMFUSION_KEY: str = "mmfusion"
# Key for the IMLViT extractor.
IMLVIT_KEY: str = "imlvit"

# Dimension of the embedding vector extracted by each extractor.
EXTRACTORS_DIMENSION: dict[str, int] = {
    TRUFOR_KEY: 512,
    SWINT_KEY: 768,
    MMFUSION_KEY: 512,
    IMLVIT_KEY: 768,
}

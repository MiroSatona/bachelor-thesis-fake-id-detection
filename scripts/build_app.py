"""
Author: Martin Zůbek
Date: 2026-04-10
File: build_app.py
Description: This script is responsible for cloning the repository of the extractor used
in the application.
"""

from scripts.download_extractors import EXTRACTORS, EXTRACTORS_DIR, MMFUSION_NAME, get_extractor

if __name__ == "__main__":
    # Create the external directory if it does not exist
    EXTRACTORS_DIR.mkdir(exist_ok=True)
    # Clone the repository of the extractor if it does not exist
    if not (EXTRACTORS_DIR / MMFUSION_NAME).exists():
        get_extractor(EXTRACTORS_DIR, MMFUSION_NAME, EXTRACTORS[MMFUSION_NAME])
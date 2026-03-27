"""
Atuhor: Martin Zůbek
Date: 2026-03-26
File: download_extractors.py
Description: This script is used for downloading pretrained models used as feature extractors. The code of the extractors is downloaded to the directory with name "external".
"""

import sys
import subprocess
from pathlib import Path


# Path to the directory where the extractors will be downloaded
EXTRACTORS_DIR: Path = Path("external")
# Name of the TruFor
TRUFOR_NAME: str = "TruFor"

# Dictionary with the names of the extractors and with URLs for downloading
EXTRACTORS: dict[str, str] = {
    TRUFOR_NAME: "https://github.com/grip-unina/TruFor.git",
}


def get_extractor(extractors_directory: Path, extractor_name: str, extractor_url: str,) -> None: 
    """
    Function for downloading the extractor.

    Args:
        extractors_directory (Path): Path to the directory where the extractor will be downloaded.
        extractor_name (str): Name of the extractor.
        extractor_url (str): URL for downloading the extractor.

    Returns:
        None
    """

    # Get path for the code of the dowloaded extractor
    extractor_path = extractors_directory / extractor_name

    # Check if the extractor is already downloaded
    if extractor_path.exists():
        print(f"[INFO] {extractor_name} at {extractor_path} is already downloaded, skipping.")
        return

    # TruFor download
    if extractor_name == TRUFOR_NAME:
        print(f"[INFO] Downloading {extractor_name} from {extractor_url}...")
        subprocess.run(["git", "clone", extractor_url, str(extractor_path)], check=True)
        return


def download_extractors() -> None:
    """
    Function for downloading all the extractors.

    Args:
        None

    Returns:
        None
    """
    EXTRACTORS_DIR.mkdir(exist_ok=True)

    # Download all the extractors from the EXTRACTORS dictionary
    for extractor_name, extractor_url in EXTRACTORS.items():
        try:
            get_extractor(EXTRACTORS_DIR, extractor_name, extractor_url)
        except subprocess.CalledProcessError as exception:
            print( f"[ERROR] Failed to download {extractor_name} from {extractor_url}: {exception}", file=sys.stderr)
            sys.exit(1)

    print("[INFO] All extractors have been successfully downloaded.")

if __name__ == "__main__":
    download_extractors()
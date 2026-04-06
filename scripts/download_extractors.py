"""
Atuhor: Martin Zůbek
Date: 2026-03-26
File: download_extractors.py
Description: This script is used for downloading pretrained models used as feature extractors. 
The code of the extractors is downloaded to the directory with name "external".
"""

import shutil
import subprocess
import sys
from pathlib import Path

import gdown

# Path to the directory where the extractors will be downloaded
EXTRACTORS_DIR: Path = Path("external")
# Name of the TruFor
TRUFOR_NAME: str = "TruFor"
# Name of the MMFusion
MMFUSION_NAME: str = "MMFusion"
# Name of the SwinT
SWINT_NAME: str = "SwinT"
# Name of the IML-ViT
IML_VIT_NAME: str = "IML_VIT"

# Dictionary with the names of the extractors and with URLs for downloading
EXTRACTORS: dict[str, str] = {
    TRUFOR_NAME: "https://github.com/grip-unina/TruFor.git",
    MMFUSION_NAME: "https://github.com/IDT-ITI/MMFusion-IML.git",
    IML_VIT_NAME: "https://github.com/SunnyHaze/IML-ViT.git",
    SWINT_NAME: "https://github.com/microsoft/Swin-Transformer.git"
    
}

################################ IML-ViT specific constants ########################################
# String with the URL for downloading the weights of the IML-ViT
IML_VIT_WEIGHTS_URL: str = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
# Path where the weights of the IML-ViT will be downloaded
IML_VIT_DOWNLOAD_WEIGHTS_PATH: Path = Path(
    "external/IML_VIT/pretrained-weights/mae_pretrain_vit_base.pth"
)
# Google Drive ID for downloading the checkpoint of trained IML-ViT model
IML_VIT_CHECKPOINT_GDRIVE_ID: str = "1xXJGJPW1i5j9Pc1JKd7fJmIAQkvt9jY7"
# Path where the checkpoint of trained IML-ViT model will be downloaded
IML_VIT_CHECKPOINT_DOWNLOAD_PATH: Path = Path("external/IML_VIT/checkpoints/iml-vit_checkpoint.pth")

############################# MMFusion specific constants ##########################################
# URL for downloading the MMFusion checkpoints
MMFUSION_CKPT_URL: str = "https://drive.google.com/drive/folders/1P1efXjVmYltZ7zWT-m6WoIlZgh9R_kAg"
# Path where the MMFusion checkpoints will be downloaded
MMFUSION_CKPT_DOWNLOAD_PATH: Path = Path("external/MMFusion/ckpt")
# URL for downloading the MMFusion pretrained models
MMFUSION_PRETRAINED_URL: str = "https://drive.google.com/drive/folders/1eTLcCZWqF-VFKayrbl0HPKzvEkThUcok"
# Path where the MMFusion pretrained models will be downloaded
MMFUSION_PRETRAINED_DOWNLOAD_PATH: Path = Path("external/MMFusion/pretrained")

############################# SwinT specific constants ##########################################
SWINT_PRETRAINED_URL: str = (  
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth"
)

def reorganize_download(goal_directory: Path, downloaded_directory_name: str) -> None:
    """
    Function for reorganizing the downloaded files. The function moves all the files from the 
    directory with name downloaded_directory_name to the goal_directory and then removes 
    the directory with name downloaded_directory_name.
    
    Arguments:
        goal_directory (Path): Path to the directory where the files should be moved.
        downloaded_directory_name (str): Name of the directory where the files are 
        currently located.

    Raises:
        RuntimeError: If the directory where the files are currently located is not a directory or
        if the file with the same name as the file being moved already exists 
        in the goal directory.

    Returns:
        None
    """
    
    # Get the path to the directory where the files are currently located
    removed_directory: Path = goal_directory / downloaded_directory_name

    # Check if the directory where the files are currently located exists
    if not removed_directory.exists():
        return
    
    # Check if the directory where the files are currently located is a directory
    if not removed_directory.is_dir():
        raise RuntimeError(
            f"Error while reorganizing download: {removed_directory} is not a directory"
        )
    
    # Move each file from the directory where the files are currently located to the goal directory
    for file in removed_directory.iterdir():
        new_file_path: Path = goal_directory / file.name

        if new_file_path.exists():
            raise RuntimeError(
                f"Error while reorganizing download: {new_file_path} already exists."
            )

        shutil.move(str(file), str(new_file_path))

    removed_directory.rmdir()
    
    
def get_extractor(extractors_directory: Path, extractor_name: str, extractor_url: str,) -> None: 
    """
    Function for downloading the extractor.

    Arguments:
        extractors_directory (Path): Path to the directory where the extractor 
        will be downloaded.
        extractor_name (str): Name of the extractor.
        extractor_url (str): URL for downloading the extractor.

    Returns:
        None
    """
    
    # Get path for the code of the dowloaded extractor
    extractor_path = extractors_directory / extractor_name

    # Get the code of the extractor by cloning the repository from the provided URL
    print(f"[INFO] Downloading {extractor_name} from {extractor_url}...")
    subprocess.run(["git", "clone", extractor_url, str(extractor_path)], check=True)

    # Download IML-ViT weights and checkpoint
    if extractor_name == IML_VIT_NAME:

        print(f"[INFO] Downloading IML-ViT checkpoint (id={IML_VIT_CHECKPOINT_GDRIVE_ID}).")
        IML_VIT_CHECKPOINT_DOWNLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(
            id=IML_VIT_CHECKPOINT_GDRIVE_ID,
            output=str(IML_VIT_CHECKPOINT_DOWNLOAD_PATH),
            quiet=False,
        )

        print(f"[INFO] Downloading IML-ViT weights from {IML_VIT_WEIGHTS_URL}.")
        subprocess.run(
            ["wget", IML_VIT_WEIGHTS_URL, "-O", str(IML_VIT_DOWNLOAD_WEIGHTS_PATH)], check=True
        )
        return
    
    # Download MMFusion weights and checkpoint
    if extractor_name == MMFUSION_NAME:
        MMFUSION_CKPT_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
        MMFUSION_PRETRAINED_DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Downloading MMFusion checkpoint from {MMFUSION_CKPT_URL}.")
        gdown.download_folder(
            url=MMFUSION_CKPT_URL,
            output=str(MMFUSION_CKPT_DOWNLOAD_PATH),
            quiet=False,
        )

        print(f"[INFO] Downloading MMFusion pretrained weights from {MMFUSION_PRETRAINED_URL}.")
        gdown.download_folder(
            url=MMFUSION_PRETRAINED_URL,
            output=str(MMFUSION_PRETRAINED_DOWNLOAD_PATH),
            quiet=False,
        )

        print("[INFO] Reorganizing MMFusion checkpoint directory.")
        reorganize_download(MMFUSION_CKPT_DOWNLOAD_PATH, "model_zoo")

        print("[INFO] Reorganizing MMFusion pretrained directory.")
        reorganize_download(MMFUSION_PRETRAINED_DOWNLOAD_PATH, "pretrained")

        return

    # Download SwinT pretrained weights
    if extractor_name == SWINT_NAME:
        print(f"[INFO] Downloading SwinT pretrained weights from {SWINT_PRETRAINED_URL}.")
        subprocess.run(
            ["mkdir", "-p", str(extractor_path / "pretrained")], check=True
        )
        subprocess.run(
            ["wget", SWINT_PRETRAINED_URL, "-O", str(
                extractor_path 
                / "pretrained" 
                / "swin_tiny_patch4_window7_224.pth"
            )],
            check=True,
        )

        return

    return

def download_extractors() -> None:
    """
    Function for downloading all the extractors.

    Args:
        None

    Returns:
        None
    """
    # Create the directory for the extractors if it does not exist
    EXTRACTORS_DIR.mkdir(exist_ok=True)

    # Download all the extractors from the EXTRACTORS dictionary
    for extractor_name, extractor_url in EXTRACTORS.items():
        try:
            get_extractor(EXTRACTORS_DIR, extractor_name, extractor_url)
        except subprocess.CalledProcessError as exception:
            print(
                (
                    f"[ERROR] Failed to download {extractor_name} "
                    f"from {extractor_url}: {exception}"
                ),
                file=sys.stderr,
            )
            sys.exit(1)

    print("[INFO] All extractors have been successfully downloaded.")

if __name__ == "__main__":
    download_extractors()
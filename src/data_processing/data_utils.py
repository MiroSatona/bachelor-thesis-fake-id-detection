"""
Author: Martin Zůbek
Date: 2026-03-28
File: data_utils.py
Description: This file defines DataUtils class with static methods for data processing, while they 
are loading from provided paths.
"""

from pathlib import Path

from .constants import IMAGE_EXTENSIONS
from .enums import AttackType


class DataUtils:
    """
    DataUtils class that provides static method for data processing, while they are loading 
    from provided paths.

    Attributes:
        None
    """

    @staticmethod
    def is_image(path: Path) -> bool:
        """
        Static method for checking if the provided path is an image file and if it has 
        a valid image extension.

        Arguments:
            path (Path): The path to the image file.

        Returns:
            bool: True if the path is a valid image file, False otherwise.
        """
        return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS

    @staticmethod
    def parse_attack_type(folder_name: str) -> AttackType | None:
        """
        Static method for parsing the attack type from the folder name. In detaset for this project,
        the folder for attack looks like:

        .
        ├── attack
        │   ├── fantasy_id
        │   │   ├── both_detp
        │   │   │   ├── arabic-NF-1003.jpg
        │   │   │   ├── arabic-NF-1012.jpg
        │   │   │   ├── arabic-NF-1030.jpg
        │   │   │   ├── arabic-NF-1037.jpg
        │   │   │   ├── arabic-NF-1040.jpg
        │   │   ├── photo_only
        │   │   │   ├── chinese2-flickr_F_16745558933_a3f9fe2bc9_k.jpg
        │   │   │   ├── chinese2-flickr_F_17178039488_60828ef8de_k.jpg
        │   │   │   ├── chinese2-flickr_F_17619703551_9705002b22_b.jpg
        │   │   │   ├── chinese2-flickr_F_17773666675_1bdbd28a62_c.jpg
        │   │   └── text_only
        │   │       ├── chinese2-flickr_F_16858102101_d4c5c1e7a2_c.jpg
        │   │       ├── chinese2-flickr_F_21696826019_7407a3bea0_k.jpg
        │   │       ├── chinese2-flickr_F_25164805428_423bc49285_o.jpg
        │   │       ├── chinese2-flickr_F_31196390972_be24a6067c_k.jpg
        │   │       ├── chinese2-flickr_F_31998401683_5272a75c42_c.jpg
        │   │       ├── chinese2-flickr_F_31998955273_5516f4db28_h.jpg
        │   │       ├── chinese2-flickr_F_32689517821_bfacb0e1ba_h.jpg

        The method will parse the attack type from the folder name, which is one of the following: 
        both, both_(everything), photo_only, text_only. For eliminating the case sensitivity 
        problem, the method will convert the folder name to lower case before parsing.

        Arguments:
            folder_name (str): The name of the folder to parse the attack type from.

        Returns:
            AttackType | None: The parsed attack type, or None if the folder name does not 
            supported attack type.
        """
        name: str = folder_name.lower()

        if name.startswith(AttackType.BOTH.value) or name == AttackType.BOTH.value:
            return AttackType.BOTH
        if name == AttackType.PHOTO_ONLY.value:
            return AttackType.PHOTO_ONLY
        if name == AttackType.TEXT_ONLY.value:
            return AttackType.TEXT_ONLY

        return None
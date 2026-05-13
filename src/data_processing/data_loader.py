"""
Author: Martin Zůbek
Date: 2026-03-28
File: data_loader.py
Description: This file contains the DataLoader class for loading identity document images from the 
dataset and creating IdentityDocumentImage instances for their internal representation.
"""

from pathlib import Path

from src.data_processing.constants import ATTACK_SUBSET, POSITIVE_SUBSET
from src.data_processing.data_utils import DataUtils
from src.data_processing.enums import AttackType, Label
from src.data_processing.identity_document import IdentityDocumentImage


class DataLoader:
    """
    Class responsible for loading identity document images from the dataset and creating 
    IdentityDocumentImage instances for their internal representation.

    Attributes:
        data_path (Path): Path to the dataset or to a single identity document image.
    """
    def __init__(self, data_path: Path) -> None:
        """
        Constructor of the DataLoader class.

        Arguments:
            data_path (Path): Path to the dataset or to a single identity document image.

        Returns:
            None
        """
        self.data_path = data_path

    def load_single_identity_document(self) -> IdentityDocumentImage:
        """
        Method for loading single identity document image and creating an IdentityDocumentImage 
        instance for internal representation.

        Arguments:
            None

        Returns:
            IdentityDocumentImage: An instance of the IdentityDocumentImage class representing 
            the loaded identity document image.
        """
        
        # Check if the provided path exists
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"The provided identity document {self.data_path} does not exist."
            )
        
        # Check if the provided path is a valid image file
        if not DataUtils.is_image(self.data_path):
            raise ValueError(f"The provided file {self.data_path} is not supported.")
        
        return IdentityDocumentImage(
            path=self.data_path,
            label=None,
            attack_type=None,
            dataset_part=None,
            dataset_subset=None
        )
        
    def load_dataset(self) -> list[IdentityDocumentImage]:
        """
        Method for loading identity document images from the dataset and creating 
        IdentityDocumentImage instances for their internal representation.

        Arguments:
            None

        Returns:
            list[IdentityDocumentImage]: A list of IdentityDocumentImage instances representing 
            the loaded identity document images from the dataset.
        
        """

        # Path to the positive subset
        positive_subset: Path = self.data_path / POSITIVE_SUBSET
        # Path to the attack subset
        attack_subset: Path = self.data_path / ATTACK_SUBSET

        # Check if the subsets of the dataset exist
        if not positive_subset.exists() or not attack_subset.exists():
            raise FileNotFoundError(f"Subsets of the dataset not found in {self.data_path}.")

        # Load identity documents from both subsets into a single internal list.
        dataset: list[IdentityDocumentImage] = []
        dataset.extend(self.load_positive_identity_documents(positive_subset))
        dataset.extend(self.load_attack_identity_documents(attack_subset))

        return dataset

    def load_positive_identity_documents(
        self,
        positive_subset: Path,
    ) -> list[IdentityDocumentImage]:
        """
        Method for loading positive identity document images and creating IdentityDocumentImage 
        instances for their internal representation.

        positive/                      # positive_subset
        ├── fantasy_id                 # dataset_part
        │   ├── fantasyid_1.jpg
        │   ├── fantasyid_10.jpg
        │   ├── fantasyid_100.jpg
        │   ├── fantasyid_101.jpg
        │   ├── fantasyid_102.jpg
        │   ├── fantasyid_103.jpg
        │   ├── fantasyid_104.jpg
        ├── midv_holo
        │   ├── midvholo_1.png
        │   ├── midvholo_10.png
        │   ├── midvholo_11.png
        │   ├── midvholo_12.png
        │   ├── midvholo_13.png

        Arguments:
            positive_subset (Path): Path to the positive subset of the dataset.

        Returns:
            list[IdentityDocumentImage]: A list of IdentityDocumentImage instances representing 
            the loaded positive identity document images.
        """

        # List for storing the loaded positive identity document images
        positive_id_documents: list[IdentityDocumentImage] = []
        
        for dataset_part in positive_subset.iterdir():

            # Check if the dataset part is a directory in the positive subset, if not, skip it
            if not dataset_part.is_dir():
                continue
            
            # Save dataset part name
            dataset_part_name: str = dataset_part.name

            # Get identity document images from the dataset part
            for identity_document in dataset_part.rglob("*"):

                # Check if the file is a valid image file, if not, skip it
                if not DataUtils.is_image(identity_document):
                    continue

                positive_id_documents.append(
                    IdentityDocumentImage(
                        path=identity_document,
                        label=Label.REAL,
                        attack_type=None,
                        dataset_part=dataset_part_name,
                        dataset_subset=None
                    )
                )
        
        return positive_id_documents

                
    def load_attack_identity_documents(self, attack_subset: Path) -> list[IdentityDocumentImage]:
        """
        Method for loading attack identity document images and creating IdentityDocumentImage 
        instances for their internal representation.


        ├── attack                                                        # attack_subset
        │   ├── fantasy_id                                                # dataset_part
        │   │   ├── both_detp                                             # attack_type
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
        
        Arguments:
            attack_subset (Path): Path to the attack subset of the dataset.

        Returns:
                list[IdentityDocumentImage]: A list of IdentityDocumentImage instances representing 
                the loaded attack identity document images.
        """

        # List for storing the loaded attack identity document images        
        attack_id_document: list[IdentityDocumentImage] = []

        for dataset_part in attack_subset.iterdir():
            
            # Check if the dataset part is a directory in the attack subset, if not, skip it
            if not dataset_part.is_dir():
                continue
            
            # Save dataset part name
            dataset_part_name: str = dataset_part.name

            for attack_type_dir in dataset_part.iterdir():

                # Check if the attack type directory is a directory, if not, skip it
                if not attack_type_dir.is_dir():
                    continue
                
                # Parse attack type from the attack type directory name
                attack_type: AttackType | None = DataUtils.parse_attack_type(attack_type_dir.name)

                if attack_type is None:
                    print(f"Warning: Unknown attack type '{attack_type_dir.name}' " 
                          f"in {attack_type_dir}. Skipping.")
                    continue

                # Get identity document images from the attack type directory
                for identity_document in attack_type_dir.rglob("*"):

                    # Check if the file is a valid image file, if not, skip it
                    if not DataUtils.is_image(identity_document):
                        continue

                    attack_id_document.append(
                        IdentityDocumentImage(
                            path=identity_document,
                            label=Label.FAKE,
                            attack_type=attack_type,
                            dataset_part=dataset_part_name,
                            dataset_subset=None
                        )
                    )

        return attack_id_document

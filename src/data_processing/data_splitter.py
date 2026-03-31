"""
Author: Martin Zůbek
Date: 2026-03-28
File: data_splitter.py
Description: This file defines DatasetSplitter class with methods for splitting datasets
into training, validation, and test sets.
"""

import random
from collections import defaultdict

from src.data_processing.constants import DEFAULT_SPLIT_SEED, TRAIN_PERCENTAGE, VAL_PERCENTAGE
from src.data_processing.dataset import Dataset
from src.data_processing.enums import AttackType, Label
from src.data_processing.identity_document import IdentityDocumentImage


class DatasetSplitter:
    """
    Class responsible for splitting the dataset of identity document images into training, 
    validation, and test sets and validating that no part of the dataset is missing.

    Attributes:
        None
    """
    def split_identity_documents_for_dataset(
            self, 
            identity_documents: list[IdentityDocumentImage]
    ) -> Dataset:
        """
        Method to split the dataset of identity document images into training, validation, 
        and test sets.

        Arguments:
            identity_documents: (list[IdentityDocumentImage]): list of IdentityDocumentImage 
            objects to be split into training, validation, and test sets.

        Returns:
            A Dataset object containing the training, validation, and test sets.
        """

        # Group identity documents by their label and attack type
        grouped_identity_documents: dict[str, list[IdentityDocumentImage]] = (
            self.grouped_identity_documents(identity_documents)
        )
        # Validate that no part of the dataset is missing
        self.validate_grouped_identity_documents(grouped_identity_documents)

        # Train data
        train_data: list[IdentityDocumentImage] = []
        # Validation data
        val_data: list[IdentityDocumentImage] = []
        # Test data
        test_data: list[IdentityDocumentImage] = []

        # Create a random shuffler
        shuffler = random.Random(DEFAULT_SPLIT_SEED)

        # Split each group of identity documents into training, validation, and test sets
        for group in grouped_identity_documents.values():

            group_copy = group.copy()
            shuffler.shuffle(group_copy)

            train_end = int(len(group_copy) * TRAIN_PERCENTAGE)
            val_end = train_end + int(len(group_copy) * VAL_PERCENTAGE)

            train_data.extend(group_copy[:train_end])
            val_data.extend(group_copy[train_end:val_end])
            test_data.extend(group_copy[val_end:])

        # Shuffle splits to avoid group ordering in the final dataset
        shuffler.shuffle(train_data)
        shuffler.shuffle(val_data)
        shuffler.shuffle(test_data)

        # Return the dataset containing the training, validation, and test sets
        return Dataset(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
        )
    
    def grouped_identity_documents(
            self, 
            identity_documents: list[IdentityDocumentImage]
        ) -> dict[str, list[IdentityDocumentImage]]:
        """
        Method to group identity documents by their label and attack type. The groups are:
        - "positive": real identity documents
        - "attack_text": identity documents with text-only attacks
        - "attack_photo": identity documents with photo-only attacks
        - "attack_both": identity documents with both text and photo attacks

        Arguments:
            identity_documents: (list[IdentityDocumentImage]): list of IdentityDocumentImage objects
            to be grouped
        
        Returns:
            A dictionary where the keys are the group names ("positive", "attack_text", 
            "attack_photo", "attack_both") and the values are lists of 
            IdentityDocumentImage objects that belong to each group.
        """
        
        # Use defaultdict to automatically create an empty list for each new key
        grouped_identity_documents: dict[str, list[IdentityDocumentImage]] = defaultdict(list)

        # Grouped identity documents by their label and attack type
        for identity_document in identity_documents:
            if identity_document.label == Label.REAL:
                grouped_identity_documents["positive"].append(identity_document)
            elif identity_document.attack_type == AttackType.TEXT_ONLY:
                grouped_identity_documents["attack_text"].append(identity_document)
            elif identity_document.attack_type == AttackType.PHOTO_ONLY:
                grouped_identity_documents["attack_photo"].append(identity_document)
            elif identity_document.attack_type == AttackType.BOTH:
                grouped_identity_documents["attack_both"].append(identity_document)

        return grouped_identity_documents

    def validate_grouped_identity_documents(self, 
        grouped_documents: dict[str, list[IdentityDocumentImage]]
    ) -> None:
        """
        Method to validate that no part of the dataset is missing. It checks that there are identity
        documents in each of the following groups:
        - "positive": real identity documents
        - "attack_text": identity documents with text-only attacks
        - "attack_photo": identity documents with photo-only attacks
        - "attack_both": identity documents with both text and photo attacks

        Arguments:
            grouped_documents: (dict[str, list[IdentityDocumentImage]]): A dictionary where the keys
            are the group names and the values are lists of IdentityDocumentImage objects 
            that belong to each group.

        Raises:
            ValueError: If any of the supported groups are empty.

        Returns:
            None
        """

        # Supported groups to check for emptiness
        supported_groups = ["positive", "attack_text", "attack_photo", "attack_both"]
        # Empty groups that are missing identity documents
        empty_groups = [
            group_name 
            for group_name in supported_groups 
            if len(grouped_documents[group_name]) == 0
        ]

        # Check if there are any empty groups
        if empty_groups:
            raise ValueError(
                f"No identity documents loaded for this groups: {', '.join(empty_groups)}"
            )

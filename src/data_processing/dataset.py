"""
Author: Martin Zůbek
Date: 2026-03-28
File: dataset.py
Description: This file defines the Dataset dataclass, which encapsulates the training, validation, 
and testing data for splitted identity document images.
"""

from dataclasses import dataclass

from src.data_processing.identity_document import IdentityDocumentImage


@dataclass(frozen=True)
class Dataset:
    """
    Dataset dataclass that holds the training, validation, and testing data
    for identity document images.
    
    Attributes:
        train_data (list[IdentityDocumentImage]): A list of identity document images for training.
        val_data (list[IdentityDocumentImage]): A list of identity document images for validation.
        test_data (list[IdentityDocumentImage]): A list of identity document images for testing.
    """
    train_data: list[IdentityDocumentImage]
    val_data: list[IdentityDocumentImage]
    test_data: list[IdentityDocumentImage]

    @property
    def dataset_len(self) -> int:
        """ 
        Property for returning the count of samples in the dataset. 
        
        Arguments:
            None

        Returns:
            int: The count of samples in the dataset (training + validation + testing).
        """
        return len(self.train_data) + len(self.val_data) + len(self.test_data)
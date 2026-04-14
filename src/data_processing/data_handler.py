"""
Author: Martin Zůbek
Date: 2026-03-28
File: data_handler.py
Description: This file defines DataHandler which is responsible for handling and operating with 
provided data for creation of dataset for training, validation, and test sets or for loading single 
identity document image for application.
"""
from pathlib import Path

from src.data_processing.data_loader import DataLoader
from src.data_processing.data_splitter import DatasetSplitter
from src.data_processing.dataset import Dataset
from src.data_processing.identity_document import IdentityDocumentImage


class DataHandler:
    """
    Class responsible for handling and operating with provided data for creation of dataset 
    for training, validation and test sets or for loading single identity document image 
    for application.
    
    Attributes:
        loader (DataLoader): An instance of DataLoader class for loading data from provided paths.
        splitter (DatasetSplitter): An instance of DatasetSplitter class for splitting the dataset 
        into training,validation, and test sets.
        identity_documents (list[IdentityDocumentImage] | None): A list of IdentityDocumentImage 
        objects representing the loaded dataset, or None if the dataset is not loaded.
        dataset (Dataset | None): A Dataset object containing the training, validation, and test 
        sets, or None if the dataset is not created.
    """
    def __init__(self, data_path: Path) -> None:
        """
        Constructor for DataHandler class.

        Arguments:
            data_path (Path): A Path object representing the path to the dataset directory.
        
        Returns:
            None
        """
        self.loader = DataLoader(data_path)
        self.splitter = DatasetSplitter()
        self.identity_documents: list[IdentityDocumentImage] | None = None
        self.dataset: Dataset | None = None

    def load_dataset(self) -> None:
        """
        Method to load the dataset of identity document images using the DataLoader instance.

        Arguments:
            None
        
        Returns:
            None
        """
        self.identity_documents = self.loader.load_dataset()

    def load_single_identity_document(self) -> IdentityDocumentImage:
        """
        Method to load a single identity document image using the DataLoader instance.
        Arguments:
            None
            
        Returns:
            IdentityDocumentImage: An instance of IdentityDocumentImage representing 
            the loaded identity document image.
        """
        return self.loader.load_single_identity_document()

    def get_dataset_for_training(self) -> Dataset:
        """
        Method to get the dataset for training, validation, and test sets. If the dataset is not 
        created, it will be created by splitting the loaded identity document images 
        using the DatasetSplitter instance.

        Arguments:
            None

        Raise:
            ValueError: If the dataset is not loaded.
        
        Returns:
            Dataset: An instance of Dataset containing the training, validation, and test sets.
        """

        # Check if the dataset is loaded, if not raise an error
        if self.identity_documents is None:
            raise ValueError("Dataset is not loaded.")

        # Create the dataset by splitting the loaded identity document images if it is not created
        if self.dataset is None:
            self.dataset = (
                self.splitter.split_identity_documents_for_dataset(self.identity_documents)
            )

        return self.dataset
    
    def shuffle_epoch_training_data(
            self, 
            training_data: list[IdentityDocumentImage]
    ) -> list[IdentityDocumentImage]:
        """
        Method to shuffle the training data for each epoch.

        Arguments:
            training_data (list[IdentityDocumentImage]): A list of IdentityDocumentImage objects
            representing the training data to be shuffled.

        Returns:
            list[IdentityDocumentImage]: A list of IdentityDocumentImage objects representing the 
            shuffled training data.
        """
        return self.splitter.training_shuffle(training_data)
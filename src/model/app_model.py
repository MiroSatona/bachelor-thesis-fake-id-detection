"""
Author: Martin Zůbek
Date: 2026-04-10
File: app_model.py
Description: This file contains the implementation of the AppModel class which is responsible 
for classifying a single identity document for the application.
"""
from pathlib import Path

import torch

from src.data_processing.data_loader import DataLoader
from src.data_processing.enums import Label
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.mmfusion import MMFusionExtractor
from src.heads.base_head import BaseHead
from src.heads.constants import (
    DROPOUT_PRECENTAGE,
    EXTRACTORS_DIMENSION,
    MMFUSION_KEY,
)
from src.heads.mlp_head import MLPHead

# Path to the checkpoint file containing the weights of the head for classification
CHECKPOINT_PATH: Path = Path(__file__).resolve().parents[2] / "checkpoint" / "checkpoint.pth"

class AppModel:
    """
    Class responsible for classifying a single identity document for the application.
    
    Attributes:
        loader (DataLoader | None): An instance of DataLoader class for loading data
        from provided paths.
        extractor (BaseExtractor): An extractor for extracting features from identity documents.
        head (BaseHead): A head for classifying the extracted features.
        """
    def __init__(self) -> None:
        """
        Constructor for AppModel class.
        
        Arguments:
            None
            
        Returns:
            None
        """
        self.loader: DataLoader | None = None
        # Initialize and load the extractor
        self.extractor: BaseExtractor = MMFusionExtractor(gpu=False)
        self.extractor.load_extractor()

        # Initialize the head
        self.head: BaseHead = MLPHead(
            input_dimension=EXTRACTORS_DIMENSION[MMFUSION_KEY],
            dropout_precentage=DROPOUT_PRECENTAGE,
        )
        # Load the weights of the head
        self.head.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location=self.extractor.device, weights_only=True)
        )
        # Move the head to the same device as the extractor and set it to evaluation mode
        self.head.to(self.extractor.device)
        self.head.eval()

    def classify(self, data_path: Path) -> bool:
        """
        Method to classify a single identity document for the application.
        
        Arguments:
            data_path (Path): A Path object of the path to the identity document.

        Raises:
            RuntimeError: If the loader cannot be created.
        
        Returns:
            bool: True if the identity document is classified as fraudulent, otherwise False.
        """
        # Create a loader with the provided data path
        self.load_new_identity_document(data_path)
        # Load the single identity document image
        if self.loader is not None:
            identity_document: IdentityDocumentImage = self.loader.load_single_identity_document()
        else:
            raise RuntimeError("Cannot create the loader for loading the identity document image.")
                    
        with torch.no_grad():
            # Extract the embedding from the identity document
            embedding: torch.Tensor = self.extractor.extract(identity_document)
            # Classify the embedding by the head
            predictions: torch.Tensor = self.head(embedding.unsqueeze(0))
            # Get the predicted class -> 0 for real identity document, 1 for fake
            predicted_class: int = int(
                torch.argmax(predictions, dim=1).item()
            )
            # Return True if the predicted class is fraudulent identity document, otherwise False
            return predicted_class == Label.FAKE.value

    def load_new_identity_document(self, data_path: Path) -> None:
        """
        Method to load a new identity document by creating a new DataLoader instance
        with the provided data path.
        
        Arguments:
            data_path (Path): A Path object representing the path to the new identity document.
        
        Returns:
            None
        """
        self.loader = DataLoader(data_path)

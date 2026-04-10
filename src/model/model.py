"""
Author: Martin Zůbek
Date: 2026-04-05
File: model.py
Description: This file contains the implementation of the Model class, which represents the model
for identity document forgery detection.
"""
from pathlib import Path

import torch

from src.data_processing.data_handler import DataHandler
from src.data_processing.dataset import Dataset
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.iml_vit import IMLViTExtractor
from src.extractors.mmfusion import MMFusionExtractor
from src.extractors.swint import SwinExtractor
from src.extractors.trufor import TruForExtractor
from src.heads.base_head import BaseHead
from src.heads.constants import (
    DROPOUT_PRECENTAGE,
    EXTRACTORS_DIMENSIONS,
    IMLVIT_KEY,
    MMFUSION_KEY,
    SWINT_KEY,
    TRUFOR_KEY,
)
from src.heads.mlp_head import MLPHead


class Model():
    """
    Class representing the model for identity document forgery detection. This class is responsible
    for initializing the extractor and head, loading the dataset, and training 
    and validating the model.

    Attributes:
        batch_size (int): The batch size for training the model.
        number_of_epochs (int): The number of epochs for training the model.
        learning_rate (float): The learning rate for training the model.
        data_handler (DataHandler): The data handler for loading the data for the model.
        data (Dataset | IdentityDocumentImage | None): The dataset for training and validating.
        or a single identity document image.
        extractor (BaseExtractor | None): The feature extractor for the model.
        head (BaseHead | None): The head for the model.
    """
    def __init__(
            self, 
            extractor_name:str, 
            device:bool,
            data_path: str,
            batch_size: int | None = None,
            number_of_epochs: int | None = None,
            learning_rate: float | None = None,
            dropout_percentage: float = DROPOUT_PRECENTAGE
    ) -> None:
        """
        Constructor of the Model class.
        
        Arguments:
            extractor_name (str): The name of the extractor to be used in the model.
            device (bool): Flag for using GPU or CPU for feature extractors.
            data_path (str): The path to the data.
            batch_size (int | None): The batch size for training the model.
            number_of_epochs (int | None): The number of epochs for training the model.
            learning_rate (float | None): The learning rate for training the model.
            dropout_percentage (float): The dropout percentage for the head of the model. 
            Default is 0.2.

        Raise:
            ValueError: If the provided extractor name is not supported.

        Returns:
            None
        """     
        self.batch_size: int | None = batch_size
        self.number_of_epochs: int | None = number_of_epochs
        self.learning_rate: float | None = learning_rate
        self.data_handler: DataHandler = DataHandler(Path(data_path))
        self.data: Dataset | IdentityDocumentImage | None = None
        self.extractor: BaseExtractor | None = None
        self.head: BaseHead | None = None

        # Initialize the extractor and head based on the provided extractor name
        if extractor_name == TRUFOR_KEY:
            self.extractor = TruForExtractor(device=device)
            self.head = MLPHead(
                input_dimension=EXTRACTORS_DIMENSIONS[TRUFOR_KEY],
                dropout_percentage=dropout_percentage
            )
        elif extractor_name == MMFUSION_KEY:
            self.extractor = MMFusionExtractor(device=device)
            self.head = MLPHead(
                input_dimension=EXTRACTORS_DIMENSIONS[MMFUSION_KEY],
                dropout_percentage=dropout_percentage
            )
        elif extractor_name == SWINT_KEY:
            self.extractor = SwinExtractor(device=device)
            self.head = MLPHead(
                input_dimension=EXTRACTORS_DIMENSIONS[SWINT_KEY],
                dropout_percentage=dropout_percentage
            )
        elif extractor_name == IMLVIT_KEY:
            self.extractor = IMLViTExtractor(device=device)
            self.head = MLPHead(
                input_dimension=EXTRACTORS_DIMENSIONS[IMLVIT_KEY],
                dropout_percentage=dropout_percentage
            )
        
        # Unsupported extractor name was provided 
        if self.extractor is None or self.head is None:
            raise ValueError(
                f"Invalid extractor name was provided: {extractor_name}"
            )
        
        # Load the extractor model
        self.extractor.load_extractor()

    def train(self) -> None:
        """
        Method for training the classification head of the model. 
        """

        # Get the dataset for training and validation
        self.data_handler.load_dataset()
        self.data = self.data_handler.get_dataset_for_training()
        
        # Move the head to the same device as the extractor
        self.head = self.head.to(self.extractor.device)

        # Set up the optimizer
        optimizer = torch.optim.Adam(
            self.head.parameters(),
            lr=self.learning_rate
        )

        # Set up the loss function
        loss_function = torch.nn.CrossEntropyLoss()
        # Get the training data from the dataset
        train_data: list[IdentityDocumentImage] = self.data.train_data

        for epoch in range(self.number_of_epochs):
            # Set the head to training mode
            self.head.train()
            # Loss for the current epoch
            epoch_loss: float = 0.0
            # Number of correct classifications
            correct_classifications: int = 0
            # Total number of identity documents processed
            total_identity_documents: int = 0

            # Iterate over the training data in batches
            for batch_iterator in range(0, len(train_data), self.batch_size):
                # Get the current batch of data
                batch_data = train_data[batch_iterator:batch_iterator + self.batch_size]

                # List of embeddings for the current batch
                batch_embeddings: list[torch.Tensor] = []
                # List of labels for the current batch 
                # -> 0 is real identity document, 1 is fake identity document
                # REF: src/data_processing/enums.py -> Label Enum class
                batch_labels: list[int] = []

                
                # Extract embeddings for the current batch and collect labels
                for identity_document in batch_data:
                    embedding: torch.Tensor = self.extractor.extract(identity_document)
                    batch_embeddings.append(embedding)
                    batch_labels.append(identity_document.label.value)

                # Collect list of the tensors of the embedings to matrix with batch size
                # and move to the same device as the extractor and head
                batch_embeddings_tensor: torch.Tensor = torch.stack(batch_embeddings).to(
                    self.extractor.device
                )
                # Convert labels to tensor and move to the extractors and heads device
                batch_labels_tensor: torch.Tensor = torch.tensor(
                    batch_labels, dtype=torch.long, device=self.extractor.device
                )  
                
                
                # Zero the gradients because gradients are accumulated by default in PyTorch
                optimizer.zero_grad()
                # Forward pass through the head to get predictions
                predictions: torch.Tensor = self.head(batch_embeddings_tensor)
                # Calculate the loss 
                loss: torch.Tensor = loss_function(predictions, batch_labels_tensor)

                # Backpropagation of the loss
                loss.backward()
                # Update the weights of the head using the optimizer
                optimizer.step()

                # Update the epoch loss
                epoch_loss += loss.item() * batch_embeddings_tensor.size(0)
                # Update the number of correct classifications
                correct_classifications += (
                    predictions.argmax(dim=1) == batch_labels_tensor
                ).sum().item()
                # Update the total number of identity documents processed
                total_identity_documents += len(batch_data)

            train_loss: float = epoch_loss / total_identity_documents
            train_accuracy: float = correct_classifications / total_identity_documents


    def validate(self, loss_function: torch.nn.CrossEntropyLoss) -> tuple[float, float]:
        
        # Set the head to evaluation mode
        self.head.eval()

        # Loss for the validation
        validation_loss: float = 0.0
        # Number of correct classifications
        correct_classifications: int = 0
        # Total number of identity documents processed
        total_identity_documents: int = 0
        # Get the validation data from the dataset
        validation_data: list[IdentityDocumentImage] = self.data.val_data

        with torch.no_grad():
            # Iterate over the validation data in batches
            for batch_iterator in range(0, len(validation_data), self.batch_size):
                # Get the current batch of data
                batch_data: list[IdentityDocumentImage] = validation_data[
                    batch_iterator:batch_iterator + self.batch_size
                ]
                
                # List of tensors of the embeddings for the current batch
                embeddings_tensor: list[torch.Tensor] = []
                # List of tensors of the labels for the current batch
                labels_tensor: list[torch.Tensor] = []

                # Extract embeddings for the current batch and collect labels
                for identity_document in batch_data:
                    embedding_tensor: torch.Tensor = self.extractor.extract(identity_document)
                    embeddings_tensor.append(embedding_tensor)
                    labels_tensor.append(identity_document.label.value)

                # Collect list of the tensors of the embedings to matrix with batch size
                # and move to the same device as the extractor and head
                batch_embedding_tensor: torch.Tensor = torch.stack(embeddings_tensor).to(
                    self.extractor.device
                )
                # Convert labels to tensor and move to the extractors and heads device
                batch_labels_tensor: torch.Tensor = torch.tensor(
                    labels_tensor, dtype=torch.long, device=self.extractor.device
                )
                
                # Forward pass through the head to get predictions
                predictions: torch.Tensor = self.head(batch_embedding_tensor)
                # Calculate the loss
                loss: torch.Tensor = loss_function(predictions, batch_labels_tensor)

                # Update the validation loss
                validation_loss += loss.item() * batch_embedding_tensor.size(0)
                # Update the number of correct classifications
                correct_classifications += (
                    predictions.argmax(dim=1) == batch_labels_tensor
                ).sum().item()
                # Update the total number of identity documents processed for validation
                total_identity_documents += len(batch_data)

            return( 
                validation_loss / total_identity_documents, 
                correct_classifications / total_identity_documents
            )

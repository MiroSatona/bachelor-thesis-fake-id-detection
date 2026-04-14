"""
Author: Martin Zůbek
Date: 2026-04-05
File: model.py
Description: 
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
from src.training.constants import LOSS
from src.training.metrics import MetricsHandler
from src.training.train_logger import TrainLogger


class Model():
    """
    
    Attributes:
        batch_size (int): The batch size for training the model.
        number_of_epochs (int): The number of epochs for training the model.
        learning_rate (float): The learning rate for training the model.
        data_handler (DataHandler): The data handler for loading the data for the model.
        data (Dataset | IdentityDocumentImage | None): The dataset for training and validating.
        or a single identity document image.
        extractor (BaseExtractor | None): The feature extractor for the model.
        head (BaseHead | None): The head for the model.
        metrics_handler_train (MetricsHandler): The metrics handler for the training phase.
        metrics_handler_val (MetricsHandler): The metrics handler for the validation phase.
        metrics_handler_test (MetricsHandler): The metrics handler for the test phase.
        train_logger (TrainLogger): The train logger for saving information about the training process.
    """
    def __init__(
            self, 
            extractor_name:str, 
            device:bool,
            data_path: str,
            batch_size: int,
            number_of_epochs: int,
            learning_rate: float,
            dropout_percentage: float = DROPOUT_PRECENTAGE
    ) -> None:
        """
        Constructor of the Model class.
        
        Arguments:
            extractor_name (str): The name of the extractor to be used in the model.
            device (bool): Flag for using GPU or CPU for feature extractors.
            data_path (str): The path to the data.
            batch_size (int): The batch size for training the model.
            number_of_epochs (int): The number of epochs for training the model.
            learning_rate (float): The learning rate for training the model.
            dropout_percentage (float): The dropout percentage for the head of the model. 
            Default is 0.2.

        Raise:
            ValueError: If the provided extractor name is not supported.

        Returns:
            None
        """     
        self.batch_size: int = batch_size
        self.number_of_epochs: int = number_of_epochs
        self.learning_rate: float = learning_rate
        self.data_handler: DataHandler = DataHandler(Path(data_path))
        self.data: Dataset | IdentityDocumentImage | None = None
        self.extractor: BaseExtractor | None = None
        self.head: BaseHead | None = None
        self.metrics_handler_train: MetricsHandler = MetricsHandler("train")
        self.metrics_handler_val: MetricsHandler = MetricsHandler("val")
        self.metrics_handler_test: MetricsHandler = MetricsHandler("test")
        self.train_logger: TrainLogger = TrainLogger(extractor_name)


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
        
        # Load the extractor model
        self.extractor.load_extractor()

        # Save the configuration of the model for the training phase
        self.train_logger.save_configuration(
            {
                "extractor_name": extractor_name,
                "device": "GPU" if device else "CPU",
                "data_path": data_path,
                "batch_size": batch_size,
                "number_of_epochs": number_of_epochs,
                "learning_rate": learning_rate,
                "dropout_percentage": dropout_percentage
            }
        )

    def train(self) -> None:
        """
        Arguments:
            None

        Returns:
            None
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
        # Variable for tracking the best validation loss for early stopping
        best_val_loss: float = float("inf")
        # Varible for tracking the number of epochs without improvement in validation loss for early stopping
        val_loss_not_improved: int = 0
        # Index of the best epoch
        best_epoch_index: int = -1

        for epoch in range(self.number_of_epochs):
            # Set the head to training mode
            self.head.train()
            # Loss for the current epoch
            epoch_loss: float = 0.0
            # Number of correct classifications
            correct_classifications: int = 0
            # Total number of identity documents processed
            total_identity_documents: int = 0
            # List of all predictions for the current epoch
            epoch_all_predictions: list[torch.Tensor] = []
            # List of all labels for the current epoch
            epoch_all_labels: list[torch.Tensor] = []

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

            # Calculate the training loss for the current epoch
            train_loss: float = epoch_loss / total_identity_documents

            # Concatenate all batch predictions and labels into a single tensor
            epoch_predictions_tensor: torch.Tensor = torch.cat(epoch_all_predictions)
            epoch_labels_tensor: torch.Tensor = torch.cat(epoch_all_labels)

            # Save the metrics for the current epoch
            self.metrics_handler_train.add_epoch_metrics(
                epoch_predictions=epoch_predictions_tensor,
                epoch_labels=epoch_labels_tensor,
                loss=train_loss
            )

            # Validate the model on the validation set
            self.validate(loss_function)

            if self.metrics_handler_val.history[-1][LOSS] < best_val_loss - 0.001:
                best_val_loss = self.metrics_handler_val.history[-1][LOSS]
                val_loss_not_improved = 0
                best_epoch_index = epoch
            else:
                val_loss_not_improved += 1

            if val_loss_not_improved >= 5:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break

        # Save the metrics of the training and validation phases after the training is finished
        self.train_logger.save_metrics(
            train_history=self.metrics_handler_train.history,
            validation_history=self.metrics_handler_val.history,
        )



        self.train_logger.save_summary(
            best_epoch_index=best_epoch_index,
            train_history=self.metrics_handler_train.history,
            validation_history=self.metrics_handler_val.history,
        )

        # Save the graphs of the metrics for the training and validation phases
        self.train_logger.save_all_metric_graphs(
            train_history=self.metrics_handler_train.history,
            validation_history=self.metrics_handler_val.history,
        )




    def validate(self, loss_function: torch.nn.CrossEntropyLoss) -> None:
        """
        Arguments:
            loss_function (torch.nn.CrossEntropyLoss): The loss function 
            used for calculating the validation loss.

        Returns:
            None
        """
        
        # Set the head to evaluation mode
        self.head.eval()

        # Loss for the validation
        validation_loss: float = 0.0
        # Total number of identity documents processed
        total_identity_documents: int = 0
        # Get the validation data from the dataset
        validation_data: list[IdentityDocumentImage] = self.data.val_data
        # List of all predictions for the current epoch
        epoch_all_predictions: list[torch.Tensor] = []
        # List of all labels for the current epoch
        epoch_all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            # Iterate over the validation data in batches
            for batch_iterator in range(0, len(validation_data), self.batch_size):
                # Get the current batch of data
                batch_data: list[IdentityDocumentImage] = validation_data[
                    batch_iterator:batch_iterator + self.batch_size
                ]
                
                # List of tensors of the embeddings for the current batch
                embeddings_tensor: list[torch.Tensor] = []
                # List of the labels for the current batch
                labels: list[int] = []

                # Extract embeddings for the current batch and collect labels
                for identity_document in batch_data:
                    embedding_tensor: torch.Tensor = self.extractor.extract(identity_document)
                    embeddings_tensor.append(embedding_tensor)
                    labels.append(identity_document.label.value)

                # Collect list of the tensors of the embedings to matrix with batch size
                # and move to the same device as the extractor and head
                batch_embedding_tensor: torch.Tensor = torch.stack(embeddings_tensor).to(
                    self.extractor.device
                )
                # Convert labels to tensor and move to the extractors and heads device
                batch_labels_tensor: torch.Tensor = torch.tensor(
                    labels, dtype=torch.long, device=self.extractor.device
                )
                
                # Forward pass through the head to get predictions
                predictions: torch.Tensor = self.head(batch_embedding_tensor)
                # Calculate the loss
                loss: torch.Tensor = loss_function(predictions, batch_labels_tensor)

                # Update the validation loss
                validation_loss += loss.item() * batch_embedding_tensor.size(0)
                # Update the number of processed identity documents
                total_identity_documents += len(batch_data)

                # Append the predictions of the current batch
                epoch_all_predictions.append(predictions)
                # Append the labels of the current batch
                epoch_all_labels.append(batch_labels_tensor)

        # Concatenate all batch predictions and labels into a single tensor
        epoch_predictions_tensor: torch.Tensor = torch.cat(epoch_all_predictions, dim=0)
        epoch_labels_tensor: torch.Tensor = torch.cat(epoch_all_labels, dim=0)

        # Save the metrics for the current epoch which is validated
        self.metrics_handler_val.add_epoch_metrics(
            epoch_predictions=epoch_predictions_tensor,
            epoch_labels=epoch_labels_tensor,
            loss=validation_loss / total_identity_documents
        )

    def test(self) -> None:
        """

        Arguments:
            None

        Returns:
            None
        """
        
        # Load the dataset if not loaded
        if self.data is None:
            self.data_handler.load_dataset()
            self.data = self.data_handler.get_dataset_for_training()

        # Move the head to the same device as the extractor
        self.head.to(self.extractor.device)
        # Set the head to evaluation mode
        self.head.eval()

        # Set up the loss function
        loss_function = torch.nn.CrossEntropyLoss()
        # Get the test data from the dataset
        test_data: list[IdentityDocumentImage] = self.data.test_data

        # Loss for the test phase
        test_loss: float = 0.0
        # Total number of identity documents processed
        total_identity_documents: int = 0
        # List of all predictions for the test phase
        epoch_all_predictions: list[torch.Tensor] = []
        # List of all labels for the test phase
        epoch_all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            # Iterate over the test data in batches
            for batch_iterator in range(0, len(test_data), self.batch_size):
                # Get data for the current batch
                batch_data: list[IdentityDocumentImage] = test_data[
                    batch_iterator:batch_iterator + self.batch_size
                ]

                # List of tensors of the embeddings for the current batch
                embeddings_tensor: list[torch.Tensor] = []
                # List of the labels for the current batch
                labels: list[int] = []

                # Extract embeddings for the current batch and collect labels
                for identity_document in batch_data:
                    embedding_tensor: torch.Tensor = self.extractor.extract(identity_document)
                    embeddings_tensor.append(embedding_tensor)
                    labels.append(identity_document.label.value)

                # Collect list of the tensors of the embedings to matrix with batch size
                # and move to the same device as the extractor and head
                batch_embeddings_tensor: torch.Tensor = torch.stack(embeddings_tensor).to(
                    self.extractor.device
                )
                # Convert labels to tensor and move to the extractors and heads device
                batch_labels_tensor: torch.Tensor = torch.tensor(
                    labels, dtype=torch.long, device=self.extractor.device
                )

                # Forward pass through the head to get predictions
                predictions: torch.Tensor = self.head(batch_embeddings_tensor)
                # Calculate the loss
                loss: torch.Tensor = loss_function(predictions, batch_labels_tensor)

                # Update the test loss
                test_loss += loss.item() * batch_embeddings_tensor.size(0)
                # Update the total number of identity documents processed
                total_identity_documents += len(batch_data)
                # Accumulate predictions and labels for the test phase
                epoch_all_predictions.append(predictions)
                epoch_all_labels.append(batch_labels_tensor)

        # Concatenate all predictions and labels for the test phase
        epoch_predictions_tensor: torch.Tensor = torch.cat(epoch_all_predictions, dim=0)
        epoch_labels_tensor: torch.Tensor = torch.cat(epoch_all_labels, dim=0)

        # Calculate and store metrics for the test phase
        self.metrics_handler_test.add_epoch_metrics(
            epoch_predictions=epoch_predictions_tensor,
            epoch_labels=epoch_labels_tensor,
            loss=test_loss / total_identity_documents
        )

        # Calculate the confusion
        confusion_matrix: dict[str, int] = self.metrics_handler_test.calculate_confusion_matrix(
            tests_predictions=epoch_predictions_tensor.argmax(dim=1),
            tests_labels=epoch_labels_tensor
        )

        # Save the test results
        self.train_logger.save_test_results(
            test_history=self.metrics_handler_test.history,
            confusion_matrix_values=confusion_matrix
        )
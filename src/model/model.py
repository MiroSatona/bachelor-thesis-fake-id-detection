"""
Author: Martin Zůbek
Date: 2026-04-05
File: model.py
Description: This file contains the implementation of the Model class which concatenates all the 
components of the project together for creating a representation of the model for training,
validating and testing.
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
    EXTRACTORS_DIMENSION,
    IMLVIT_KEY,
    MMFUSION_KEY,
    SWINT_KEY,
    TRUFOR_KEY,
)
from src.heads.mlp_head import MLPHead
from src.training.constants import (
    ACCURACY,
    CHECKPOINT_FILE_NAME,
    EARLY_STOPPING_DELTA,
    EARLY_STOPPING_PATIENCE,
    F1_SCORE,
    LOSS,
    SCHEDULER_FACTOR,
    SCHEDULER_MIN_LR,
    SCHEDULER_PATIENCE,
    PRECISION,
    RECALL,
)
from src.training.metrics import MetricsHandler
from src.training.train_logger import TrainLogger


class Model:
    """
    Class representing the model for training, validating and testing. It contains all 
    the components of the project together and provides methods for training,
    validating and testing the model.

    Attributes:
        batch_size (int): The batch size for training the model.
        number_of_epochs (int): The number of epochs for training the model.
        learning_rate (float): The learning rate for training the model.
        data_handler (DataHandler): The data handler for loading the data for the model.
        data (Dataset): The dataset for training and validating.
        extractor (BaseExtractor | None): The feature extractor for the model.
        head (BaseHead | None): The head for the model.
        metrics_handler_train (MetricsHandler): The metrics handler for the training phase.
        metrics_handler_val (MetricsHandler): The metrics handler for the validation phase.
        metrics_handler_test (MetricsHandler): The metrics handler for the test phase.
        train_logger (TrainLogger): The train logger for saving information 
        about the training process.
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
            dropout_percentage (float): The dropout percentage for the head
            of the model. 
            Default is 0.2.

        Returns:
            None
        """     
        self.batch_size: int = batch_size
        self.number_of_epochs: int = number_of_epochs
        self.learning_rate: float = learning_rate
        self.data_handler: DataHandler = DataHandler(Path(data_path))
        self.metrics_handler_train: MetricsHandler = MetricsHandler("train")
        self.metrics_handler_val: MetricsHandler = MetricsHandler("val")
        self.metrics_handler_test: MetricsHandler = MetricsHandler("test")
        self.train_logger: TrainLogger = TrainLogger(extractor_name)

        # Initialize the extractor and head based on the provided extractor name
        print(
            f"[Model] Initializing extractor: {extractor_name} "
            f"on {'GPU' if device else 'CPU'}"
        )
        self.extractor: BaseExtractor = self.create_extractor(extractor_name, device)
        self.head: BaseHead = MLPHead(
            input_dimension=EXTRACTORS_DIMENSION[extractor_name],
            dropout_precentage=dropout_percentage
        )

        # Load the extractor model
        print("[Model] Loading extractor weights.")
        self.extractor.load_extractor()
        print("[Model] Extractor ready.")

        # Get the dataset for training and validation and test
        print(f"[Model] Loading dataset from: {data_path}")
        self.data_handler.load_dataset()
        self.data: Dataset = self.data_handler.get_dataset_for_training()
        print(
            f"[Model] Dataset loaded with the following number of samples:"
            f"train: {len(self.data.train_data)}, "
            f"val: {len(self.data.val_data)}, "
            f"test: {len(self.data.test_data)}"
        )
        

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

    def create_extractor(self, extractor_name: str, device: bool) -> BaseExtractor:
        """
        Method for creating the extractor based on the provided extractor name.

        Arguments:
            extractor_name (str): The name of the extractor.
            device (bool): Flag for using GPU or CPU.

        Raises:
            ValueError: If an unknown extractor name is provided.

        Returns:
            BaseExtractor: The created extractor instance.
        """
        if extractor_name == TRUFOR_KEY:
            return TruForExtractor(gpu=device)
        if extractor_name == MMFUSION_KEY:
            return MMFusionExtractor(gpu=device)
        if extractor_name == SWINT_KEY:
            return SwinExtractor(gpu=device)
        if extractor_name == IMLVIT_KEY:
            return IMLViTExtractor(gpu=device)
        raise ValueError(f"Unknown extractor name was provided: {extractor_name}")

    def extract_embeddings(
            self,
            data: list[IdentityDocumentImage],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Method for extracting embeddings and labels from identity document images.
        This method is used for the training, validation, and test phases, and computes 
        the embeddings only once for optimization.

        Arguments:
            data (list[IdentityDocumentImage]): The list of identity document images for which
            to extract the embeddings.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The extracted embeddings and the labels.
        """

        # List for storing the extracted embeddings
        embeddings: list[torch.Tensor] = []
        # List for storing the labels
        labels: list[int] = []
        for identity_document in data:
            # Extract the embedding for the current identity document image
            embedding: torch.Tensor = self.extractor.extract(identity_document)
            # Append the embedding and move to the CPU
            embeddings.append(embedding.cpu())
            # Append the label of the current identity document
            if identity_document.label is None:
                raise ValueError(
                    f"Identity document image with path: {identity_document.path} "
                    f"has no label."
                )
            labels.append(identity_document.label.value)
        return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)

    def train(self) -> None:
        """
        Method for handling the training of the model.

        Arguments:
            None

        Returns:
            None
        """
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
        training_data: list[IdentityDocumentImage] = self.data.train_data
        # Extract the embeddings and store labels for the training data 
        training_embeddings: torch.Tensor
        training_labels: torch.Tensor
        training_embeddings, training_labels = self.extract_embeddings(
            training_data
        )
        # Extract the embeddings and store labels for the validation data
        validation_data: list[IdentityDocumentImage] = self.data.val_data
        validation_embeddings: torch.Tensor
        validation_labels: torch.Tensor
        validation_embeddings, validation_labels = self.extract_embeddings(
            validation_data
        )


        if self.extractor.extractor_model is None:
            raise RuntimeError("Extractor model is not loaded.")
        # Move the extractor_model of the extractor to CPU
        self.extractor.extractor_model.cpu()
        # Delete the extractor_model from the extractor to free up GPU memory
        del self.extractor.extractor_model
        self.extractor.extractor_model = None
        torch.cuda.empty_cache()

        # Create a tensor of indexes for the training data
        training_indexes: torch.Tensor = torch.arange(len(training_embeddings))
        print(f"[Train] Starting training for {self.number_of_epochs} epoch(s).")
        # Variable for tracking the best validation loss for early stopping
        best_validation_loss: float = float("inf")
        # Variable for tracking the number of epochs without improvement in validation loss
        # for early stopping
        validation_loss_not_improved: int = 0
        # Index of the best epoch
        best_epoch_index: int = -1
        # Scheduler for learning rate control
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=SCHEDULER_FACTOR,
                patience=SCHEDULER_PATIENCE,
                min_lr=SCHEDULER_MIN_LR,
            )
        )

        for epoch in range(self.number_of_epochs):
            print(f"[Train] Epoch {epoch + 1}/{self.number_of_epochs} started.")
            # Set the head to training mode
            self.head.train()
            # Loss for the current epoch
            epoch_loss: float = 0.0
            # Total number of identity documents processed
            total_identity_documents: int = 0
            # List of all predictions for the current epoch
            epoch_all_predictions: list[torch.Tensor] = []
            # List of all labels for the current epoch
            epoch_all_labels: list[torch.Tensor] = []
            # Shuffle indexes for the training data
            shuffled_training_indexes: torch.Tensor = training_indexes[
                torch.randperm(len(training_indexes))
            ]

            # Iterate over the training data in batches
            for batch_iterator in range(0, len(shuffled_training_indexes), self.batch_size):
                # Get the indexes for the current batch
                batch_indexes: torch.Tensor = shuffled_training_indexes[
                    batch_iterator:batch_iterator + self.batch_size
                ]
                # Get embeddings tensor of the batch and move to the device
                batch_embeddings_tensor: torch.Tensor = training_embeddings[batch_indexes].to(
                    self.extractor.device
                )
                # Get labels tensor of the batch and move to the device
                batch_labels_tensor: torch.Tensor = training_labels[batch_indexes].to(
                    self.extractor.device
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
                # Update the total number of identity documents processed
                total_identity_documents += batch_embeddings_tensor.size(0)
                # Append the predictions of the current batch
                epoch_all_predictions.append(predictions.detach().cpu())
                # Append the labels of the current batch
                epoch_all_labels.append(batch_labels_tensor.detach().cpu())

            # Calculate the training loss for the current epoch
            training_loss: float = epoch_loss / total_identity_documents

            # Concatenate all batch predictions and labels into a single tensor
            epoch_predictions_tensor: torch.Tensor = torch.cat(epoch_all_predictions)
            epoch_labels_tensor: torch.Tensor = torch.cat(epoch_all_labels)

            # Save the metrics for the current epoch
            self.metrics_handler_train.add_epoch_metrics(
                epoch_predictions=epoch_predictions_tensor,
                epoch_labels=epoch_labels_tensor,
                loss=training_loss
            )

            # Validate the model on the validation set
            self.validate(validation_embeddings, validation_labels, loss_function)
            # Step of the scheduler for learning rate control based on the validation loss
            current_validation_loss: float = (
                self.metrics_handler_val.history.history_values[LOSS][-1]
            )
            scheduler.step(current_validation_loss)

            training_history_values: dict[str, list[float]] =(
                self.metrics_handler_train.history.history_values
            )
            validation_history_values: dict[str, list[float]] = (
                self.metrics_handler_val.history.history_values
            )
            print(
                f"[Train] Epoch {epoch + 1}/{self.number_of_epochs} "
                f"| training loss: {training_history_values[LOSS][-1]:.4f}"
                f", training  accuracy: {training_history_values[ACCURACY][-1]:.4f}"
                f" | validation loss: {validation_history_values[LOSS][-1]:.4f}"
                f", validation accuracy: {validation_history_values[ACCURACY][-1]:.4f}"
            )

            # Check for improvement in validation loss for early stopping
            # and save the best model weights
            if current_validation_loss < best_validation_loss - EARLY_STOPPING_DELTA:
                best_validation_loss = current_validation_loss
                validation_loss_not_improved = 0
                best_epoch_index = epoch
                # Save the model weights
                self.train_logger.save_checkpoint(
                    CHECKPOINT_FILE_NAME, self.head
                )
                print(
                    f"[Train] New best model saved "
                    f"(validation loss: {best_validation_loss:.4f})."
                )
            else:
                validation_loss_not_improved += 1

            if validation_loss_not_improved >= EARLY_STOPPING_PATIENCE:
                print(f"[Train] Early stopping after {epoch + 1} epoch(s).")
                break

        print(f"[Train] Training finished. Best epoch: {best_epoch_index + 1}.")
        # Save the metrics of the training and validation phases after the training is finished
        self.train_logger.save_metrics(
            train_history=self.metrics_handler_train.history,
            validation_history=self.metrics_handler_val.history,
        )

        # Save the summary of the training process
        self.train_logger.save_summary(
            best_epoch=best_epoch_index + 1,
            train_history=self.metrics_handler_train.history,
            val_history=self.metrics_handler_val.history,
        )

        # Save the graphs of the metrics for the training and validation phases
        self.train_logger.save_all_metric_graphs(
            train_history=self.metrics_handler_train.history,
            val_history=self.metrics_handler_val.history,
        )

    def evaluate(
        self,
        embeddings_tensor: torch.Tensor,
        labels_tensor: torch.Tensor,
        loss_function: torch.nn.CrossEntropyLoss,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Method for evaluating the classifier on pre-computed embeddings.

        Arguments:
            embeddings_tensor (torch.Tensor): Tensor of embedding extracted from the data on CPU.
            labels_tensor (torch.Tensor): Tensor of labels for the data on CPU.
            loss_function (torch.nn.CrossEntropyLoss): The loss function used
            for calculating the loss.

        Returns:
            tuple[float, torch.Tensor, torch.Tensor]: The average loss, all predictions
            and all labels for the dataset.
        """
        # Loss for the evaluation
        total_loss: float = 0.0
        # Total number of identity documents processed
        total_identity_documents: int = 0
        # List of all predictions
        all_predictions: list[torch.Tensor] = []
        # List of all labels
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            # Iterate over the data in batches
            for batch_iterator in range(0, len(embeddings_tensor), self.batch_size):
                # Get embeddings tensor of the batch and move to the device
                batch_embeddings_tensor: torch.Tensor = embeddings_tensor[
                    batch_iterator:batch_iterator + self.batch_size
                ].to(self.extractor.device)
                # Get labels tensor of the batch and move to the device
                batch_labels_tensor: torch.Tensor = labels_tensor[
                    batch_iterator:batch_iterator + self.batch_size
                ].to(self.extractor.device)

                # Forward pass through the head to get predictions
                predictions: torch.Tensor = self.head(batch_embeddings_tensor)
                # Calculate the loss
                loss: torch.Tensor = loss_function(predictions, batch_labels_tensor)

                # Update the total loss
                total_loss += loss.item() * batch_embeddings_tensor.size(0)
                # Update the total number of identity documents processed
                total_identity_documents += batch_embeddings_tensor.size(0)
                # Append the predictions of the current batch
                all_predictions.append(predictions.detach().cpu())
                # Append the labels of the current batch
                all_labels.append(batch_labels_tensor.detach().cpu())

        # Return the loss and concatenated predictions and labels
        return (
            total_loss / total_identity_documents,
            torch.cat(all_predictions, dim=0),
            torch.cat(all_labels, dim=0)
        )

    def validate(
            self,
            validation_embeddings_tensor: torch.Tensor,
            validation_labels_tensor: torch.Tensor,
            loss_function: torch.nn.CrossEntropyLoss,
    ) -> None:
        """
        Method for handling the validation of the model.

        Arguments:
            validation_embeddings_tensor (torch.Tensor): Tensor of embedding 
            extracted from the validation data on CPU.
            validation_labels_tensor (torch.Tensor): Tensor of labels for the validation 
            data on CPU.
            loss_function (torch.nn.CrossEntropyLoss): The loss function 
            used for calculating the validation loss.

        Returns:
            None
        """
        # Set the head to evaluation mode
        self.head.eval()

        # Evaluate the model on the validation data
        validation_loss: float
        epoch_predictions_tensor: torch.Tensor
        epoch_labels_tensor: torch.Tensor
        validation_loss, epoch_predictions_tensor, epoch_labels_tensor = self.evaluate(
            validation_embeddings_tensor, validation_labels_tensor, loss_function
        )

        # Save the metrics for the current validation epoch
        self.metrics_handler_val.add_epoch_metrics(
            epoch_predictions=epoch_predictions_tensor,
            epoch_labels=epoch_labels_tensor,
            loss=validation_loss
        )

    def test(self) -> None:
        """
        Method for handling the testing of the model.

        Arguments:
            None

        Returns:
            None
        """
        # Move the head to the same device as the extractor
        self.head = self.head.to(self.extractor.device)
        # Load the best saved weights into the head
        print("[Test] Loading checkpoint into classifier head.")
        self.train_logger.load_checkpoint(CHECKPOINT_FILE_NAME, self.head)
        # Set the head to evaluation mode
        self.head.eval()
        # Set up the loss function
        loss_function = torch.nn.CrossEntropyLoss()

        print("[Test] Evaluating on test set.")
        # Extract the embeddings and store labels for the test data
        test_embeddings, test_labels = self.extract_embeddings(
            self.data.test_data
        )
        # Evaluate the model on the test data
        test_loss, epoch_predictions_tensor, epoch_labels_tensor = self.evaluate(
            test_embeddings, test_labels, loss_function
        )

        # Calculate and store metrics for the test phase
        self.metrics_handler_test.add_epoch_metrics(
            epoch_predictions=epoch_predictions_tensor,
            epoch_labels=epoch_labels_tensor,
            loss=test_loss
        )

        # Calculate the confusion matrix
        confusion_matrix: dict[str, int] = self.metrics_handler_test.calculate_confusion_matrix(
            tests_predictions=epoch_predictions_tensor.argmax(dim=1),
            tests_labels=epoch_labels_tensor
        )

        # Save the test results
        self.train_logger.save_test_results(
            test_history=self.metrics_handler_test.history,
            confusion_matrix_values=confusion_matrix
        )
        test_history_values: dict[str, list[float]] = (
            self.metrics_handler_test.history.history_values
        )

        print(
            f"[Test] Results"
            f" | loss: {test_history_values[LOSS][0]:.4f}"
            f", accuracy: {test_history_values[ACCURACY][0]:.4f}"
            f", precision: {test_history_values[PRECISION][0]:.4f}"
            f", recall: {test_history_values[RECALL][0]:.4f}"
            f", F1-score: {test_history_values[F1_SCORE][0]:.4f}"
        )
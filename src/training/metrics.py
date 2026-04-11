"""
Author: Martin Zůbek
Date: 2026-04-03
File: metrics.py
Description: File containing the MetricsHandler class, which is responsible for calculating 
and storing metrics during training and validation phases.
"""

import torch

from src.data_processing.enums import Label
from src.training.constants import ACCURACY, F1_SCORE, FN, FP, LOSS, PRECISION, RECALL, TN, TP
from src.training.history import History


class MetricsHandler:
    """
    Class responsible for calculating and storing metrics during training and validation phases.

    Attributes:
        phase_name (str): The name of the phase.
        history (History): An instance of the History dataclass to store metric values 
        across epochs.
    """

    def __init__(self, phase_name: str) -> None:
        """
        Constructor for MetricsHandler.

        Arguments:
            phase_name (str): The name of the phase.

        Returns:
            None
        """
        self.phase_name: str = phase_name
        self.history: History = History()

    def calculate_accuracy(
        self,
        epoch_predictions: torch.Tensor,
        epoch_labels: torch.Tensor
    ) -> float:
        """
        Method to calculate accuracy for the current epoch.

        Arguments:
            epoch_predictions (torch.Tensor): The predictions made by the model 
            for the current epoch.
            epoch_labels (torch.Tensor): The true labels for the current epoch.

        Returns:
            float: The calculated accuracy for the current epoch.        
        
        """
        true_positives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.FAKE)
            & (epoch_labels == Label.FAKE)
        ).sum().item()

        true_negatives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.REAL)
            & (epoch_labels == Label.REAL)
        ).sum().item()

        return (true_positives + true_negatives) / epoch_labels.size(0)
    
    def calculate_recall(
        self,
        epoch_predictions: torch.Tensor,
        epoch_labels: torch.Tensor
    ) -> float:
        """
        Method to calculate recall for the current epoch.

        Arguments:
            epoch_predictions (torch.Tensor): The predictions made by the model 
            for the current epoch.
            epoch_labels (torch.Tensor): The true labels for the current epoch.
        
        Returns:
            float: The calculated recall for the current epoch.
        """
        true_positives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.FAKE)
            & (epoch_labels == Label.FAKE)
        ).sum().item()

        false_negatives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.REAL)
            & (epoch_labels == Label.FAKE)
        ).sum().item()

        return(  
            true_positives / (true_positives + false_negatives) 
            if (true_positives + false_negatives) > 0 
            else 0.0
        )

    def calculate_precision(
        self,
        epoch_predictions: torch.Tensor,
        epoch_labels: torch.Tensor
    ) -> float:
        """
        Method to calculate precision for the current epoch.

        Arguments:
            epoch_predictions (torch.Tensor): The predictions made by the model 
            for the current epoch.
            epoch_labels (torch.Tensor): The true labels for the current epoch.

        Returns:
            float: The calculated precision for the current epoch.
        """
        true_positives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.FAKE)
            & (epoch_labels == Label.FAKE)
        ).sum().item()

        false_positives: int | float = (
            (epoch_predictions.argmax(dim=1) == Label.FAKE)
            & (epoch_labels == Label.REAL)
        ).sum().item()

        return(  
            true_positives / (true_positives + false_positives) 
            if (true_positives + false_positives) > 0 
            else 0.0
        )

    def calculate_f1_score(
        self,
        precision: float,
        recall: float
    ) -> float:
        """
        Method to calculate F1 score for the current epoch.

        Arguments:
            precision (float): The calculated precision for the current epoch.
            recall (float): The calculated recall for the current epoch.

        Returns:
            float: The calculated F1 score for the current epoch.
        """
        return(
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 
            else 0.0
        )

    def calculate_confusion_matrix(
        self,
        tests_predictions: torch.Tensor,
        tests_labels: torch.Tensor
    ) -> dict[str, int]:
        """
        Method to calculate the confusion matrix for the test results.

        Arguments:
            tests_predictions (torch.Tensor): The predictions of the model for the test set.
            tests_labels (torch.Tensor): The true labels for the test set.

        Returns:
            dict[str, int]: Dictionary with TP, TN, FP, FN counts.
        """

        tp_value: int = int(
            ((tests_predictions == Label.FAKE) & (tests_labels == Label.FAKE)).sum().item()
        )
        tn_value: int = int(
            ((tests_predictions == Label.REAL) & (tests_labels == Label.REAL)).sum().item()
        )
        fp_value: int = int(
            ((tests_predictions == Label.FAKE) & (tests_labels == Label.REAL)).sum().item()
        )
        fn_value: int = int(
            ((tests_predictions == Label.REAL) & (tests_labels == Label.FAKE)).sum().item()
        )

        return {
            TP: tp_value,
            TN: tn_value,
            FP: fp_value,
            FN: fn_value
        }

    def add_epoch_metrics(
        self,
        epoch_predictions: torch.Tensor,
        epoch_labels: torch.Tensor,
        loss: float
    ) -> None:
        """
        Method to calculate and add metrics of the current epoch to the history.

        Arguments:
            epoch_predictions (torch.Tensor): The predictions of the model for the current epoch.
            epoch_labels (torch.Tensor): The true labels for the current epoch.
            loss (float): The loss value for the current epoch.

        Returns:
            None
        """

        accuracy: float = self.calculate_accuracy(epoch_predictions, epoch_labels)
        precision: float = self.calculate_precision(epoch_predictions, epoch_labels)
        recall: float = self.calculate_recall(epoch_predictions, epoch_labels)
        f1_score: float = self.calculate_f1_score(precision, recall)

        self.history.append_epoch_results(LOSS, loss)
        self.history.append_epoch_results(ACCURACY, accuracy)
        self.history.append_epoch_results(PRECISION, precision)
        self.history.append_epoch_results(RECALL, recall)
        self.history.append_epoch_results(F1_SCORE, f1_score)
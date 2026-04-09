"""
Author: Martin Zůbek
Date: 2026-04-03
File: metrics.py
Description:

"""

import matplotlib.pyplot as plt
import torch

from src.training.constants import (
    ACCURACY,
    F1_SCORE,
    LOSS,
    METRICS,
    PRECISION,
    RECALL,
)


class MetricsHandler:
    """
    
    """

    def __init__(self, phase_name: str) -> None:
        """
        
        """
        self.phase_name: str = phase_name
        self.history: dict[str, list[float]] = {metric: [] for metric in METRICS}

    def calculate_accuracy(
        self,
        predicted_classes: torch.Tensor,
        true_labels: torch.Tensor
    ) -> float:
        """
        
        """
        correct: int = (predicted_classes == true_labels).sum().item()
        accuracy: float = correct / true_labels.size(0)
        return accuracy

    def calculate_precision(
        self,
        predicted_classes: torch.Tensor,
        true_labels: torch.Tensor
    ) -> float:
        """
       
        """
        true_positives: int = (
            (predicted_classes == 1) & (true_labels == 1)
        ).sum().item()
        predicted_positives: int = (predicted_classes == 1).sum().item()
        precision: float = (
            true_positives / predicted_positives if predicted_positives > 0 else 0.0
        )
        return precision

    def calculate_recall(
        self,
        predicted_classes: torch.Tensor,
        true_labels: torch.Tensor
    ) -> float:
        """
        
        """
        true_positives: int = (
            (predicted_classes == 1) & (true_labels == 1)
        ).sum().item()
        actual_positives: int = (true_labels == 1).sum().item()
        recall: float = (
            true_positives / actual_positives if actual_positives > 0 else 0.0
        )
        return recall

    def calculate_f1_score(self, precision: float, recall: float) -> float:
        """
       
        """
        f1_score: float = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        return f1_score

    def calculate_epoch_metrics(
        self,
        predictions: torch.Tensor,
        true_labels: torch.Tensor,
        loss: float
    ) -> None:
        """
       
        """
        predicted_classes: torch.Tensor = predictions.argmax(dim=1)

        accuracy: float = self.calculate_accuracy(predicted_classes, true_labels)
        precision: float = self.calculate_precision(predicted_classes, true_labels)
        recall: float = self.calculate_recall(predicted_classes, true_labels)
        f1_score: float = self.calculate_f1_score(precision, recall)

        self.append_epoch_metrics(loss, accuracy, precision, recall, f1_score)

    def append_epoch_metrics(
        self,
        loss: float,
        accuracy: float,
        precision: float,
        recall: float,
        f1_score: float
    ) -> None:
        """
        
        """
        self.history[LOSS].append(loss)
        self.history[ACCURACY].append(accuracy)
        self.history[PRECISION].append(precision)
        self.history[RECALL].append(recall)
        self.history[F1_SCORE].append(f1_score)

    def plot_metric(self, metric: str) -> None:
        """
        
        """
        if metric not in self.history:
            raise ValueError(
                f"Metric '{metric}' was not found in {self.phase_name} history."
            )

        metric_values: list[float] = self.history[metric]
        epochs: list[int] = list(range(1, len(metric_values) + 1))

        plt.figure()
        plt.plot(epochs, metric_values, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{self.phase_name.capitalize()} {metric}")
        plt.tight_layout()
        plt.show()
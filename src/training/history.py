"""
Author: Martin Zůbek
Date: 2026-04-09
File: history.py
Description: File containing the History dataclass, which is responsible for storing the history
of metrics across epochs.
"""

from dataclasses import dataclass, field

from src.training.constants import METRICS


@dataclass
class History:
    """
    Dataclass for storing the history of metrics across epochs.

    Attributes:
        history_values (dict[str, list[float]]): A dictionary where keys are metric names 
        and values are lists of metric values for each epoch.
    """
    history_values: dict[str, list[float]] = field(
        default_factory=lambda: {
            metric: [] for metric in METRICS
        }
    )

    def append_epoch_results(self, metric_name: str, value: float) -> None:
        """ 
        Append a metric value for a specific epoch.

        Arguments:
            metric_name (str): The name of the metric to append.
            value (float): The value of the metric to append.

        Raises:
            ValueError: If the metric_name is not supported.

        Returns:
            None        
        """
        if metric_name in self.history_values:
            self.history_values[metric_name].append(value)
        else:
            raise ValueError(f"{metric_name} is not supported metric.")
        
    def get_epoch_metric_history(self, metric_name: str) -> list[float]:
        """ 
        Get the history of a specific metric of the one epoch.

        Arguments:
            metric_name (str): The name of the metric.

        Raises:
            ValueError: If the metric_name is not supported.

        Returns:
            list[float]: The history of the specified metric.
        """
        if metric_name in self.history_values:
            return self.history_values[metric_name]
        else:
            raise ValueError(f"{metric_name} is not supported metric.")

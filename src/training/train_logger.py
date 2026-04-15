"""
Author: Martin Zůbek
Date: 2026-04-03
File: train_logger.py
Description: This file contains the TrainLogger class, which is responsible for saving training, 
validation and test metrics.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as pyplot
import torch

from src.training.constants import (
    ACCURACY,
    CHECKPOINTS_DIR,
    CONFIG_FILE_NAME,
    EXPERIMENTS_PATH,
    F1_SCORE,
    FN,
    FP,
    GRAPHS_DIR,
    LOSS,
    METRICS,
    METRICS_FILE_NAME,
    PRECISION,
    RECALL,
    SUMMARY_FILE_NAME,
    TEST_CONFUSION_MATRIX_FILE_MATRIX_NAME,
    TEST_CONFUSION_MATRIX_FILE_VALUES_NAME,
    TEST_DIR,
    TEST_METRICS_FILE_NAME,
    TN,
    TP,
    TRAIN_TITLE,
    VALIDATION_TITLE,
)
from src.training.history import History


class TrainLogger:
    """
    Class responsible for saving training, validation and test metrics.

    Attributes:
        extractor_name (str): The name of the extractor, which is used to get features
        from identity document images.
        root_dir (Path): The root directory for storing all data and metrics of the training 
        for the current head of the extractor.
        config_path (Path): The path to the configuration file of the training 
        and validation of the head.
        metrics_path (Path): The path to the metrics file for the current head.
        summary_path (Path): The path to the summary file for the current head.
        graphs_dir (Path): The directory where graphs of the training and validation metrics 
        for the current head are stored.
        checkpoints_dir (Path): The directory where checkpoints 
        for the current head are stored.
        test_dir (Path): The directory where test results for the current head are stored.
    """

    def __init__(
        self,
        extractor_name: str
    ) -> None:
        """
        Constructor for TrainLogger.

        Arguments:
            extractor_name (str): The name of the extractor, which is used to get features 
            from identity document images.

        Returns:
            None
        """

        # Create the experiments directory if it does not exist
        if not EXPERIMENTS_PATH.exists():
            EXPERIMENTS_PATH.mkdir(parents=True,exist_ok=True)
        
        # Set the extractor name and create the root directory for the current extractor
        self.extractor_name: str = extractor_name
        self.root_dir: Path = EXPERIMENTS_PATH / self.extractor_name 
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Create the paths to the configuration, metrics and summary files
        self.config_path: Path = self.root_dir / CONFIG_FILE_NAME
        self.metrics_path: Path = self.root_dir / METRICS_FILE_NAME
        self.summary_path: Path = self.root_dir / SUMMARY_FILE_NAME

        # Create the directories for graphs, checkpoints and test results
        self.graphs_dir: Path = self.root_dir / GRAPHS_DIR
        self.checkpoints_dir: Path = self.root_dir / CHECKPOINTS_DIR
        self.test_dir: Path = self.root_dir / TEST_DIR

        # Create the graphs, checkpoints and test directories if they do not exist
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def save_configuration(self, configuration: dict[str, Any]) -> None:
        """
        Method for saving the configuration of the training.

        Arguments:
            configuration (dict[str, Any]): Configuration to save.

        Returns:
            None
        """
        with self.config_path.open("w", encoding="utf-8") as configuration_file:
            json.dump(configuration, configuration_file, indent=4)

    def save_metrics(
            self,
            train_history: History,
            validation_history: History
            ) -> None:
        """
        Method for saving the metrics of the training and validation.

        Arguments:
            train_history (History): The history of the training metrics.
            validation_history (History): The history of the validation metrics.

        Returns:
            None
        """
        metrics_data: dict[str, Any] = {
            TRAIN_TITLE: train_history.history_values,
            VALIDATION_TITLE: validation_history.history_values,
        }
        with self.metrics_path.open("w", encoding="utf-8") as metrics_file:
            json.dump(metrics_data, metrics_file, indent=4)
            
    def save_summary(
        self,
        best_epoch: int,
        train_history: History,
        val_history: History
    ) -> None:
        """
        Method for saving the summary of the training.

        Arguments:
            best_epoch (int): The epoch with the best results.
            train_history (History): The history of the training metrics.
            val_history (History): The history of the validation metrics.

        Returns:
            None
        """

        # Get the history of metrics for training and validation
        training_history: dict[str, Any] = train_history.history_values
        validation_history: dict[str, Any] = val_history.history_values

        # Create the summary dictionary with the best epoch, number of epochs and best metric values
        summary: dict[str, Any] = {
            "best_epoch": best_epoch,
            "number_of_epochs": len(training_history[LOSS]),

            "best_train_loss": min(training_history[LOSS]) if training_history[LOSS] else None,
            "best_validation_loss": (
                min(validation_history[LOSS]) if validation_history[LOSS] else None
            ),
            "best_validation_accuracy": (
                max(validation_history[ACCURACY]) if validation_history[ACCURACY] else None
            ),
            "best_validation_precision": (
                max(validation_history[PRECISION]) if validation_history[PRECISION] else None
            ),
            "best_validation_recall": (
                max(validation_history[RECALL]) if validation_history[RECALL] else None
            ),
            "best_validation_f1_score": (
                max(validation_history[F1_SCORE]) if validation_history[F1_SCORE] else None
            ),

            "best_epoch_train_loss": training_history[LOSS][best_epoch - 1],
            "best_epoch_validation_loss": validation_history[LOSS][best_epoch - 1],
            "best_epoch_validation_accuracy": validation_history[ACCURACY][best_epoch - 1],
            "best_epoch_validation_precision": validation_history[PRECISION][best_epoch - 1],
            "best_epoch_validation_recall": validation_history[RECALL][best_epoch - 1],
            "best_epoch_validation_f1_score": validation_history[F1_SCORE][best_epoch - 1],
        }
        
        # Save the summary to a JSON file
        with self.summary_path.open("w", encoding="utf-8") as summary_file:
            json.dump(summary, summary_file, indent=4)
    
    def save_checkpoint(
        self,
        file_name: str,
        head: torch.nn.Module
    ) -> None:
        """
        Method for saving the checkpoint of the classifier head.

        Arguments:
            file_name (str): The name of the checkpoint file.
            head (torch.nn.Module): The classifier head to save.

        Returns:
            None
        """
        torch.save(head.state_dict(), self.checkpoints_dir / file_name)

    def load_checkpoint(
            self,
            file_name: str,
            head: torch.nn.Module
        ) -> None:
            """
            Method for loading a checkpoint into the classifier head.

            Arguments:
                file_name (str): The name of the checkpoint file.
                head (torch.nn.Module): The classifier head to load the checkpoint into.

            Returns:
                None
            """
            head.load_state_dict(
                torch.load(self.checkpoints_dir / file_name, weights_only=True)
            )


    def save_metric_graph(
            self,
            train_history: History,
            val_history: History,
            metric_name: str
    ) -> None:
        """
        Method for saving the graph of the specified metric for both training and validation.

        Arguments:
            train_history (History): The history of the training metrics.
            val_history (History): The history of the validation metrics.
            metric_name (str): The name of the metric to plot.

        Returns:
            None
        """

        # Get the history of the specified metric for training and validation
        training_history: list[float] = train_history.history_values[metric_name]
        validation_history: list[float] = val_history.history_values[metric_name]
        epochs: list[int] = list(range(1, len(training_history) + 1))
        
        # Create the graph for the specified metric
        pyplot.figure()
        # Plot the metric values for training and validation
        pyplot.plot(epochs, training_history, marker="o", label=TRAIN_TITLE.capitalize())
        pyplot.plot(epochs, validation_history, marker="o", label=VALIDATION_TITLE.capitalize())
        # Set the labels and title for the graph
        pyplot.xlabel("Epoch")
        pyplot.ylabel(metric_name.replace("_", " ").capitalize())
        pyplot.title(metric_name.replace("_", " ").capitalize())
        # Add legend and layout adjustments
        pyplot.legend()
        pyplot.grid(True)
        pyplot.tight_layout()
        # Save the graph to a file and close the plot
        pyplot.savefig(self.graphs_dir / f"{metric_name}.png" , dpi=300)
        pyplot.close()

    def save_all_metric_graphs(
        self,
        train_history: History,
        val_history: History
    ) -> None:
        """
        Method for saving the graphs of all metrics for both training and validation.

        Arguments:
            train_history (History): The history of the training metrics.
            val_history (History): The history of the validation metrics.
        
        Returns:
            None
        """
        for metric_name in METRICS:
            self.save_metric_graph(train_history, val_history, metric_name)

    def save_test_results(
            self,
            test_history: History,
            confusion_matrix_values: dict[str, int]
    ) -> None:
        """
        Method for saving the test results, including the history of test metrics
        and confusion matrix values and heatmap of the confusion matrix.

        Arguments:
            test_history (History): The history of the test metrics.
            confusion_matrix_values (dict[str, int]): The values of the confusion matrix components.
            
        Returns:
            None
        """

        # Save the test metrics to a JSON file
        test_metrics_data: dict[str, Any] = {
            LOSS: test_history.history_values[LOSS][0] if test_history
                .history_values[LOSS] else None,
            ACCURACY: test_history.history_values[ACCURACY][0] if test_history
                .history_values[ACCURACY] else None,
            PRECISION: test_history.history_values[PRECISION][0] if test_history
                .history_values[PRECISION] else None,
            RECALL: test_history.history_values[RECALL][0] if test_history
                .history_values[RECALL] else None,
            F1_SCORE: test_history.history_values[F1_SCORE][0] if test_history
                .history_values[F1_SCORE] else None,
        }

        with (self.test_dir / TEST_METRICS_FILE_NAME).open(
            "w", encoding="utf-8"
        ) as test_metrics_file:
            json.dump(test_metrics_data, test_metrics_file, indent=4)

        # Save the confusion matrix values to a JSON file
        with (self.test_dir / TEST_CONFUSION_MATRIX_FILE_VALUES_NAME).open(
            "w", encoding="utf-8"
        ) as confusion_matrix_values_file:
            json.dump(confusion_matrix_values, confusion_matrix_values_file, indent=4)

        # Build the confusion matrix array
        true_positives: int = confusion_matrix_values.get(TP, 0)
        true_negatives: int = confusion_matrix_values.get(TN, 0)
        false_positives: int = confusion_matrix_values.get(FP, 0)
        false_negatives: int = confusion_matrix_values.get(FN, 0)

        confusion_matrix: list[list[int]] = [
            [true_negatives, false_positives],
            [false_negatives, true_positives]
        ]

        # Display the confusion matrix as a heatmap and save it to a file
        pyplot.figure(figsize=(5, 5))
        pyplot.imshow(confusion_matrix, cmap="Blues")
        pyplot.colorbar()
        pyplot.xticks([0, 1], ["REAL", "FAKE"])
        pyplot.yticks([0, 1], ["REAL", "FAKE"])
        pyplot.xlabel("Predicted")
        pyplot.ylabel("Actual")
        pyplot.title("Confusion Matrix")
        max_value = max(map(max, confusion_matrix))
        for row in range(2):
            for column in range(2):
                value = confusion_matrix[row][column]
                color = "white" if value > max_value / 2 else "black"

                pyplot.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=12
                )

        pyplot.tight_layout()
        pyplot.savefig(self.test_dir / TEST_CONFUSION_MATRIX_FILE_MATRIX_NAME, dpi=300)
        pyplot.close()
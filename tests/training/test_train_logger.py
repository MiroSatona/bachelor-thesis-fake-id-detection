"""
Author: Martin Zůbek
Date: 2026-04-03
File: test_train_logger.py
Description: Unit tests for the TrainLogger class.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import matplotlib.pyplot as pyplot
import pytest
import torch

from src.training.constants import (
    ACCURACY,
    CHECKPOINTS_DIR,
    F1_SCORE,
    FN,
    FP,
    GRAPHS_DIR,
    LOSS,
    PRECISION,
    RECALL,
    TEST_CONFUSION_MATRIX_FILE_VALUES_NAME,
    TEST_DIR,
    TEST_METRICS_FILE_NAME,
    TN,
    TP,
    TRAIN_TITLE,
    VALIDATION_TITLE,
)
from src.training.history import History
from src.training.train_logger import TrainLogger


@pytest.fixture
def logger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainLogger:
    """
    Pytest fixture that creates a TrainLogger with EXPERIMENTS_PATH redirected to tmp_path.

    Arguments:
        tmp_path (Path): A temporary directory provided by pytest.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        TrainLogger: An instance of TrainLogger pointing to tmp_path.
    """
    monkeypatch.setattr("src.training.train_logger.EXPERIMENTS_PATH", tmp_path)
    return TrainLogger("test_extractor")


def create_history(values: dict[str, list[float]]) -> History:
    """
    Function to create a History instance with metrics values.

    Arguments:
        values (dict[str, list[float]]): A dictionary where keys are metric names and values 
        are lists of metric values for each epoch.

    Returns:
        History: An instance of History populated with the provided metric values.
    """
    history: History = History()
    for metric, epoch_values in values.items():
        for value in epoch_values:
            history.append_epoch_results(metric, value)
    return history


def test_required_directories_exists(logger: TrainLogger) -> None:
    """
    Test that the required directories for checkpoints, graphs, and test results are created.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    assert logger.root_dir.exists()
    assert (logger.root_dir / GRAPHS_DIR).exists()
    assert (logger.root_dir / CHECKPOINTS_DIR).exists()
    assert (logger.root_dir / TEST_DIR).exists()

def test_save_configuration_file(logger: TrainLogger) -> None:
    """
    Test that the configuration file is created with the correct configuration.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    configuration: dict = {"learning_rate": 0.001, "epochs": 10}
    logger.save_configuration(configuration)
    assert logger.config_path.exists()
    with logger.config_path.open(encoding="utf-8") as f:
        saved_configuration: dict = json.load(f)
    assert saved_configuration == configuration

def test_save_metrics(logger: TrainLogger) -> None:
    """
    Test that the save_metrics method correctly saves training and validation metrics.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.
    
    Returns:
        None
    """

    train_history = create_history({
        LOSS: [0.5, 0.4],
        ACCURACY: [0.8, 0.85],
        PRECISION: [0.7, 0.75],
        RECALL: [0.6, 0.65],
        F1_SCORE: [0.65, 0.7]
    })

    val_history = create_history({
        LOSS: [0.6, 0.55],
        ACCURACY: [0.75, 0.78],
        PRECISION: [0.65, 0.7],
        RECALL: [0.55, 0.6],
        F1_SCORE: [0.60, 0.65]
    })

    logger.save_metrics(train_history, val_history)
    assert logger.metrics_path.exists()
    
    with logger.metrics_path.open(encoding="utf-8") as metrics_file:
        saved_metrics: dict = json.load(metrics_file)
    assert TRAIN_TITLE in saved_metrics
    assert VALIDATION_TITLE in saved_metrics
    assert saved_metrics[TRAIN_TITLE][LOSS] == [0.5, 0.4]
    assert saved_metrics[VALIDATION_TITLE][LOSS] == [0.6, 0.55]

def test_save_summary_with_best_epoch(logger: TrainLogger) -> None:
    """
    Test that the save_summary method correctly saves the best epoch and number of epochs.
    
    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    train_history = create_history({
        LOSS: [0.5, 0.3],
        ACCURACY: [0.8, 0.9],
        PRECISION: [0.7, 0.8],
        RECALL: [0.6, 0.7],
        F1_SCORE: [0.65, 0.75]
    })

    val_history = create_history({
        LOSS: [0.6, 0.4],
        ACCURACY: [0.75, 0.88],
        PRECISION: [0.65, 0.78],
        RECALL: [0.55, 0.68],
        F1_SCORE: [0.60, 0.73]
    })

    logger.save_summary(best_epoch=2, train_history=train_history, val_history=val_history)
    with logger.summary_path.open(encoding="utf-8") as summary_file:
        saved_summary: dict = json.load(summary_file)
    assert saved_summary["best_epoch"] == 2
    assert saved_summary["number_of_epochs"] == 2

def test_save_summary_with_best_values(logger: TrainLogger) -> None:
    """
    Test that the save_summary method correctly saves the best metric values.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    train_history = create_history({
        LOSS: [0.5, 0.3],
        ACCURACY: [0.8, 0.9],
        PRECISION: [0.7, 0.8],
        RECALL: [0.6, 0.7],
        F1_SCORE: [0.65, 0.75]
    })

    val_history = create_history({
        LOSS: [0.6, 0.4],
        ACCURACY: [0.75, 0.88],
        PRECISION: [0.65, 0.78],
        RECALL: [0.55, 0.68],
        F1_SCORE: [0.60, 0.73]
    })

    logger.save_summary(best_epoch=2, train_history=train_history, val_history=val_history)
    with logger.summary_path.open(encoding="utf-8") as summary_file:
        saved_summary: dict = json.load(summary_file)
    
    assert saved_summary["best_train_loss"] == 0.3
    assert saved_summary["best_validation_loss"] == 0.4

def test_save_checkpoint(logger: TrainLogger) -> None:
    """
    Test that the save_checkpoint method saves the model checkpoint correctly.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    model = torch.nn.Linear(4, 2)
    logger.save_checkpoint("checkpoint.pth", model)
    checkpoint_path = logger.checkpoints_dir / "checkpoint.pth"
    assert checkpoint_path.exists()
    state_dict = torch.load(checkpoint_path, weights_only=True)
    assert "weight" in state_dict

def test_load_checkpoint(logger: TrainLogger) -> None:
    """
    Test that the load_checkpoint method correctly loads a saved checkpoint into a model.

    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.

    Returns:
        None
    """
    model = torch.nn.Linear(4, 2)
    logger.save_checkpoint("checkpoint.pth", model)

    new_model = torch.nn.Linear(4, 2)
    logger.load_checkpoint("checkpoint.pth", new_model)

    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], new_model.state_dict()[key])


def test_save_test_results(
        logger: TrainLogger,
        monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test that the save_test_results method saves test metrics and confusion matrix correctly.
    
    Arguments:
        logger (TrainLogger): The TrainLogger instance to test.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture to mock file saving.

    Returns:
        None
    """
    monkeypatch.setattr(pyplot, "savefig", MagicMock())
    monkeypatch.setattr(pyplot, "show", MagicMock())

    test_history = create_history({
        LOSS: [0.35],
        ACCURACY: [0.9],
        PRECISION: [0.88],
        RECALL: [0.85],
        F1_SCORE: [0.86]
    })

    confusion_matrix_values = {
        TP: 45,
        TN: 45,
        FP: 5,
        FN: 5
    }

    logger.save_test_results(test_history, confusion_matrix_values)
    assert (logger.test_dir / TEST_METRICS_FILE_NAME).exists()
    assert (logger.test_dir / TEST_CONFUSION_MATRIX_FILE_VALUES_NAME).exists()

    with (logger.test_dir / TEST_METRICS_FILE_NAME).open(encoding="utf-8") as metrics_file:
        metrics: dict = json.load(metrics_file)
    assert metrics[LOSS] == pytest.approx(0.35)
    assert metrics[ACCURACY] == pytest.approx(0.9)

    confusion_matrix_path = logger.test_dir / TEST_CONFUSION_MATRIX_FILE_VALUES_NAME
    with confusion_matrix_path.open(encoding="utf-8") as confusion_matrix_file:
        confusion_matrix: dict = json.load(confusion_matrix_file)
    assert confusion_matrix[TP] == 45
    assert confusion_matrix[TN] == 45
    assert confusion_matrix[FP] == 5
    assert confusion_matrix[FN] == 5
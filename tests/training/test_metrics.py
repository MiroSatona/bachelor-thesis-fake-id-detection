"""
Author: Martin Zůbek
Date: 2026-04-03
File: test_metrics.py
Description: Unit tests for the MetricsHandler class.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

import torch

from src.data_processing.enums import Label
from src.training.constants import ACCURACY, F1_SCORE, FN, FP, LOSS, PRECISION, RECALL, TN, TP
from src.training.metrics import MetricsHandler


def make_model_output(predicted_labels: list[int]) -> torch.Tensor:
    """
    Function to create a fake model output tensor where each row represents class probabilities.

    Arguments:
        predicted_labels (list[int]): List of predicted class indices (0=REAL, 1=FAKE).

    Returns:
        torch.Tensor: Tensor of shape (N, 2) with probabilities for each class.
    """
    rows: list[list[float]] = []
    for label in predicted_labels:
        if label == Label.REAL:
            rows.append([1.0, 0.0])
        else:
            rows.append([0.0, 1.0])
    return torch.tensor(rows)


def make_labels(labels: list[int]) -> torch.Tensor:
    """
    Function to create a tensor of ground-truth labels.
    
    Arguments:
        labels (list[int]): List of true class indices.

    Returns:
        torch.Tensor: 1D tensor with true labels.
    """
    return torch.tensor(labels)

def test_calculate_accuracy_with_hundred_percent_correct() -> None:
    """
    Test that calculate_accuracy returns 1.0 when all predictions are correct.

    Arguments:
        None
    
    Returns:
        None
    """
    predicted_labels: list[int] = [Label.REAL, Label.FAKE, Label.REAL, Label.FAKE]
    true_labels: list[int] = [Label.REAL, Label.FAKE, Label.REAL, Label.FAKE]
    model_output: torch.Tensor = make_model_output(predicted_labels)
    labels: torch.Tensor = make_labels(true_labels)
    metrics_handler = MetricsHandler("train")
    accuracy: float = metrics_handler.calculate_accuracy(model_output, labels)

    assert accuracy == 1.0

def test_calculate_accuracy_with_fifty_percent_correct() -> None:
    """
    Test that calculate_accuracy returns 0.5 when half of the predictions are correct.

    Arguments:
        None
    
    Returns:
        None
    """
    predicted_labels: list[int] = [Label.REAL, Label.FAKE, Label.REAL, Label.FAKE]
    true_labels: list[int] = [Label.REAL, Label.REAL, Label.FAKE, Label.FAKE]
    model_output: torch.Tensor = make_model_output(predicted_labels)
    labels: torch.Tensor = make_labels(true_labels)
    metrics_handler = MetricsHandler("train")
    accuracy: float = metrics_handler.calculate_accuracy(model_output, labels)

    assert accuracy == 0.5

def test_calculate_accuracy_with_zero_percent_correct() -> None:
    """
    Test that calculate_accuracy returns 0.0 when all predictions are incorrect.

    Arguments:
        None
    
    Returns:
        None
    """
    predicted_labels: list[int] = [Label.REAL, Label.REAL, Label.REAL, Label.REAL]
    true_labels: list[int] = [Label.FAKE, Label.FAKE, Label.FAKE, Label.FAKE]
    model_output: torch.Tensor = make_model_output(predicted_labels)
    labels: torch.Tensor = make_labels(true_labels)
    metrics_handler = MetricsHandler("train")
    accuracy: float = metrics_handler.calculate_accuracy(model_output, labels)

    assert accuracy == 0.0

def test_calculate_precision_hundred_percent_correct() -> None:
    """
    Test that calculate_precision returns 1.0 when all predicted positives are true positives.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    predictions = make_model_output([Label.FAKE, Label.FAKE])
    labels = make_labels([Label.FAKE, Label.FAKE])
    precision: float = handler.calculate_precision(predictions, labels)

    assert precision == 1.0

def test_calculate_precision_zero() -> None:
    """
    Test that calculate_precision returns 0.0 when there are no predicted positives (TP + FP = 0).

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    predictions = make_model_output([Label.REAL, Label.REAL])
    labels = make_labels([Label.FAKE, Label.REAL])
    precision: float = handler.calculate_precision(predictions, labels)

    assert precision == 0.0

def test_calculate_recall_hundred_percent_correct() -> None:
    """
    Test that calculate_recall returns 1.0 when all actual positives are correctly predicted.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    predictions = make_model_output([Label.FAKE, Label.FAKE])
    labels = make_labels([Label.FAKE, Label.FAKE])
    recall: float = handler.calculate_recall(predictions, labels)

    assert recall == 1.0

def test_calculate_recall_zero() -> None:
    """
    Test that calculate_recall returns 0.0 when there are no actual positives (TP + FN = 0).

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    predictions = make_model_output([Label.REAL, Label.REAL])
    labels = make_labels([Label.REAL, Label.REAL])
    recall: float = handler.calculate_recall(predictions, labels)

    assert recall == 0.0

def test_calculate_f1_score_hundred_percent_correct() -> None:
    """
    Test that calculate_f1_score returns 1.0 when precision and recall are both 1.0.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    precision: float = 1.0
    recall: float = 1.0
    f1_score: float = handler.calculate_f1_score(precision, recall)

    assert f1_score == 1.0

def test_calculate_f1_score_zero() -> None:
    """
    Test that calculate_f1_score returns 0.0 when precision and recall are both 0.0.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = handler.calculate_f1_score(precision, recall)

    assert f1_score == 0.0

def test_calculate_f1_score_with_standard_values() -> None:
    """
    Test that calculate_f1_score returns the correct value for typical precision and recall.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("train")
    f1: float = handler.calculate_f1_score(precision=0.8, recall=0.6)
    assert abs(f1 - (2 * 0.8 * 0.6 / (0.8 + 0.6))) < 1e-6


def test_calculate_confusion_matrix_hundred_percent_correct() -> None:
    """
    Test that calculate_confusion_matrix returns correct counts when all predictions are correct.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("test")
    predictions = make_model_output([Label.FAKE, Label.REAL, Label.FAKE, Label.REAL])
    labels = make_labels([Label.FAKE, Label.REAL, Label.FAKE, Label.REAL])
    cm: dict[str, int] = handler.calculate_confusion_matrix(predictions.argmax(dim=1), labels)

    assert cm[TP] == 2
    assert cm[TN] == 2
    assert cm[FP] == 0
    assert cm[FN] == 0

def test_calculate_confusion_matrix_zero_percent_correct() -> None:
    """
    Test that calculate_confusion_matrix returns correct counts when all predictions are wrong.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("test")
    predictions = make_model_output([Label.FAKE, Label.FAKE, Label.REAL, Label.REAL])
    labels = make_labels([Label.REAL, Label.REAL, Label.FAKE, Label.FAKE])
    cm: dict[str, int] = handler.calculate_confusion_matrix(predictions.argmax(dim=1), labels)

    assert cm[TP] == 0
    assert cm[TN] == 0
    assert cm[FP] == 2
    assert cm[FN] == 2

def test_calculate_confusion_matrix_with_normal_predictions() -> None:
    """
    Test that calculate_confusion_matrix returns correct counts for a mix of correct 
    and incorrect predictions.

    Arguments:
        None

    Returns:
        None
    """
    handler: MetricsHandler = MetricsHandler("test")
    predictions = make_model_output([Label.FAKE, Label.FAKE, Label.REAL, Label.REAL])
    labels = make_labels([Label.FAKE, Label.REAL, Label.FAKE, Label.REAL])
    cm: dict[str, int] = handler.calculate_confusion_matrix(predictions.argmax(dim=1), labels)

    assert cm[TP] == 1
    assert cm[TN] == 1
    assert cm[FP] == 1
    assert cm[FN] == 1

def test_add_epoch_metrics_to_history() -> None:
    """
    Test that add_epoch_metrics correctly calculates and adds metrics to the history.

    Arguments:
        None

    Returns:
        None
    """

    handler: MetricsHandler = MetricsHandler("train")
    predictions = make_model_output([Label.FAKE, Label.FAKE, Label.REAL, Label.REAL])
    labels = make_labels([Label.FAKE, Label.REAL, Label.FAKE, Label.REAL])
    loss: float = 0.5
    handler.add_epoch_metrics(predictions, labels, loss)

    assert len(handler.history.history_values[LOSS]) == 1
    assert handler.history.history_values[LOSS][0] == loss
    assert handler.history.history_values[ACCURACY][0] == 0.5
    assert handler.history.history_values[PRECISION][0] == 0.5
    assert handler.history.history_values[RECALL][0] == 0.5
    assert handler.history.history_values[F1_SCORE][0] == 0.5
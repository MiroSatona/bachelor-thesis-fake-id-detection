"""
Author: Martin Zůbek
Date: 2026-04-03
File: test_history.py
Description: Unit tests for the History dataclass.
"""

import pytest

from src.training.constants import ACCURACY, F1_SCORE, LOSS, METRICS, PRECISION, RECALL
from src.training.history import History


def test_history_initialies_with_all_metrics() -> None:
    """
    Test that the History dataclass initializes with all supported metrics as keys 
    and empty lists as values.
    
    Arguments:
        None
        
    Returns:
        None
    """
    history = History()
    assert set(history.history_values.keys()) == set(METRICS)
    for metric in METRICS:
        assert history.history_values[metric] == []

def test_history_different_states() -> None:
    """
    Test that the Hustory dataclass do not share state between different instances.
    
    Arguments:
        None
        
    Returns:
        None
    """
    history_first = History()
    history_second = History()
    history_first.append_epoch_results(ACCURACY, 0.8)
    assert history_second.history_values[ACCURACY] == []

def test_history_append_value() -> None:
    """
    Test that the append_epoch_results method correctly append value to the history.
    
    Arguments:
        None
        
    Returns:
        None
    """
    history = History()
    history.append_epoch_results(LOSS, 0.5)
    assert history.history_values[LOSS] == [0.5]

def test_history_append_multiple_values() -> None:
    """
    Test that the append_epoch_results method can append multiple values for the same metric.
    
    Arguments:
        None

    Returns:
        None
    """
    history = History()
    history.append_epoch_results(ACCURACY, 0.8)
    history.append_epoch_results(ACCURACY, 0.85)
    history.append_epoch_results(ACCURACY, 0.9)
    assert history.history_values[ACCURACY] == [0.8, 0.85, 0.9]

def test_history_append_unsupported_metric() -> None:
    """
    Test that the append_epoch_results method raises a ValueError when trying to append 
    a value for an unsupported metric.

    Arguments:
        None

    Returns:
        None
    """
    history = History()
    with pytest.raises(ValueError):
        history.append_epoch_results("unsupported_metric", 0.5)

def test_history_affects_other_metrics() -> None:
    """
    Test that appending a value to one metric does not affect the values of other metrics.

    Arguments:
        None

    Returns:
        None
    """
    history = History()
    history.append_epoch_results(PRECISION, 0.7)
    for metric in METRICS:
        if metric != PRECISION:
            assert history.history_values[metric] == []

def test_append_epoch_all_metrics() -> None:
    """
    Test that the append_epoch_results method can append values for all supported metrics.

    Arguments:
        None

    Returns:
        None
    """
    history = History()
    values: dict[str, float] = {
        LOSS: 0.3,
        ACCURACY: 0.9,
        PRECISION: 0.85,
        RECALL: 0.80,
        F1_SCORE: 0.82,
    }
    for metric, value in values.items():
        history.append_epoch_results(metric, value)
    for metric, value in values.items():
        assert history.history_values[metric] == [value]

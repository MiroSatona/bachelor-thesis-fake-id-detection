"""
Author: Martin Zůbek
Date: 2026-04-01
File: test_mlp_head.py
Description: Unit tests for the MLPHead class.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

from typing import cast

import torch

from src.heads.constants import CLASSES_COUNT, DROPOUT_PRECENTAGE
from src.heads.mlp_head import MLPHead


def test_right_output_shape_with_single_identity_document() -> None:
    """
    Test that the forward() method of MLPHead returns the correct output shape 
    for a single input sample, batch size = 1.

    Arguments:
        None

    Returns:
        None
    """
    input_dimension: int = 512
    head: MLPHead = MLPHead(input_dimension)
    embedding: torch.Tensor = torch.randn(1, input_dimension)
    output: torch.Tensor = head(embedding)
    assert output.shape == (1, CLASSES_COUNT)

def test_right_output_shape_with_batch() -> None:
    """
    Test that the forward() method of MLPHead returns the correct output shape 
    for a batch of input samples.

    Arguments:
        None

    Returns:
        None
    """
    input_dimension: int = 768
    batch_size: int = 8
    head: MLPHead = MLPHead(input_dimension)
    embedding: torch.Tensor = torch.randn(batch_size, input_dimension)
    output: torch.Tensor = head(embedding)
    assert output.shape == (batch_size, CLASSES_COUNT)

def test_create_head_with_default_parameters() -> None:
    """
    Test that MLPHead is initialized with the correct default parameters.

    Arguments:
        None

    Returns:
        None
    """
    input_dimension: int = 512
    head: MLPHead = MLPHead(input_dimension)
    assert head.input_dimension == input_dimension
    assert head.classification_classes_count == CLASSES_COUNT
    assert head.dropout_precentage == DROPOUT_PRECENTAGE

def test_create_head_with_custom_parameters() -> None:
    """
    Test that MLPHead can be initialized with custom parameters.

    Arguments:
        None

    Returns:
        None
    """
    input_dimension: int = 256
    custom_classes_count: int = 5
    custom_dropout_percentage: float = 0.3
    head: MLPHead = MLPHead(
        input_dimension=input_dimension,
        classification_classes_count=custom_classes_count,
        dropout_precentage=custom_dropout_percentage
    )
    assert head.input_dimension == input_dimension
    assert head.classification_classes_count == custom_classes_count
    assert head.dropout_precentage == custom_dropout_percentage

def test_right_dimension_of_intermediate_layer() -> None:
    """
    Test that the intermediate layer in MLPHead has the correct dimension.

    Arguments:
        None

    Returns:
        None
    """
    input_dimension: int = 512
    head: MLPHead = MLPHead(input_dimension)
    intermediate_layer: torch.nn.Linear = cast(torch.nn.Linear, head.mlp_classification_head[0])
    assert intermediate_layer.in_features == input_dimension
    assert intermediate_layer.out_features == input_dimension // 2
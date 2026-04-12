"""
Author: Martin Zůbek
Date: 2026-04-01
File: base_head.py
Description: This file contains the implementation of the BaseHead class. 
This class is used like a template for all the heads which will classify
extracted features.
"""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.heads.constants import CLASSES_COUNT, DROPOUT_PRECENTAGE


class BaseHead(ABC, nn.Module):
    """
    Base class for all the heads for classification of the extracted features.
    
    Attributes:
    input_dimension (int): Dimension of the input feature vector.
    classification_classes_count (int): Number of classes for classification.
    """
    def __init__(
            self,
            input_dimension: int,
            classification_classes_count: int = CLASSES_COUNT,
            dropout_precentage: float = DROPOUT_PRECENTAGE
        ) -> None:
        """
        Constructor for the BaseHead class.
        
        Arguments:
            input_dimension (int): Dimension of the input feature vector.
            classification_classes_count (int): Number of classes for classification.
            Default is 2.
            dropout_precentage (float): Dropout percentage for the MLP head. 
            Default is 0.2.
            
        Returns:
            None
        """
        super().__init__()
        self.input_dimension: int = input_dimension
        self.classification_classes_count: int = classification_classes_count
        self.dropout_precentage: float = dropout_precentage

    @abstractmethod
    def forward(self, embedding_vector: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the classification head independent
        of the architecture of the head.

        Arguments:
            embedding_vector (torch.Tensor): Input embedding vector.

        Returns:
            torch.Tensor: Output scores for each class.
        """
        pass
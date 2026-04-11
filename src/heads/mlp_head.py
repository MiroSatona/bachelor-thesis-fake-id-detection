"""
Author: Martin Zůbek
Date: 2026-04-01
File: mlp_head.py
Description: This file contains the implementation of the MLPHead class. 
This class is used for classification of the extracted features using a multi-layer perceptron.
"""
import torch
import torch.nn as nn

from src.heads.base_head import BaseHead
from src.heads.constants import CLASSES_COUNT, DROPOUT_PRECENTAGE


class MLPHead(BaseHead):
    """
    Class for classification of the extracted features using a multi-layer perceptron.
    
    Attributes:
        input_dimension (int): Dimension of the input feature vector.
        classification_classes_count (int): Number of classes for classification.
        intermediate_dimension (int): Dimension of the intermediate layer in the MLP.
        mlp_classifier (nn.Sequential): Sequential model for the MLP classifier.

    """
    def __init__(
            self,
            input_dimension: int,
            classification_classes_count: int = CLASSES_COUNT,
            dropout_precentage: float = DROPOUT_PRECENTAGE
            classification_classes_count: int = CLASSES_COUNT
        ) -> None:
        """
        Constructor for the MLPHead class.

        Arguments:
            input_dimension (int): Dimension of the input feature vector.
            classification_classes_count (int): Number of classes for classification.
            Default is 2.
            dropout_precentage (float): Dropout percentage for the MLP head.
            Default is 0.2.

        Returns:
            None
        """
        # Create the instance of the BaseHead class
        super().__init__(
            input_dimension, classification_classes_count, dropout_precentage
            )
        super().__init__(input_dimension, classification_classes_count)
        # Calculate the dimension of the hidden layer
        self.intermediate_dimension: int = self.input_dimension // 2

        
        # MLP classification head
        #
        # Architecture:
        # 1) Linear layer for projects embedding from input_dimension
        # 2) GELU activation
        # 3) Dropout for regularization
        # 4) Linear layer for maps to the classes
        #    (positive identity document or fraudulent identity document)
        self.mlp_classification_head: nn.Sequential = nn.Sequential(
            nn.Linear(self.input_dimension, self.intermediate_dimension),
            nn.GELU(),
            nn.Dropout(self.dropout_precentage),
            nn.Dropout(DROPOUT_PRECENTAGE),
            nn.Linear(self.intermediate_dimension, self.classification_classes_count)
        )

    def forward(self, embedding_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MLP classification head.
        
        Arguments:
            embedding_vector (torch.Tensor): Input embedding vector.
            
        Returns:
            torch.Tensor: Output scores for each class.
        """
        result: torch.Tensor = self.mlp_classification_head(embedding_vector)
        return result
"""
Author: Martin Zůbek
Date: 2026-03-26
File: base_extractor.py
Description: This file contains the implementation of the BaseExtractor class. 
This class is used like a template for all the future extractors.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import torch

from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.constants import CPU, CUDA, DEFAULT_USE_GPU, EXTRACTORS_DIR_STR


class BaseExtractor(ABC):
    """
    Base class for all the extractors. Other extractors should inherit from this class.

    Attributes:
        use_gpu (bool): Flag for using GPU or CPU for feature extractors.
        device (torch.device): Device for feature extractors.
        root_dir (Path): Root directory of the clonned code of the extractors.
        src_dir (Path): Directory with the source code for the extractor.
    """

    def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
        """
        Constructor for the BaseExtractor class.
        
        Args:
            gpu (bool): Flag for using GPU or CPU for feature extractors. Default is True.
            
        Returns:
            None
        """
        self.use_gpu: bool = gpu
        self.device: torch.device = ( 
            torch.device(CUDA) 
            if self.use_gpu and torch.cuda.is_available() 
            else torch.device(CPU)
        )
        self.root_dir: Path = Path(__file__).resolve().parents[2] / EXTRACTORS_DIR_STR
        self.src_dir: Path = self.root_dir
        self.extractor_model: torch.nn.Module | None = None

    @abstractmethod
    def load_extractor(self) -> None:
        """
        Method for loading the specific extractor model.
        
        Arguments:
            None
            
        Returns:
            None
        """
        pass

    @abstractmethod
    def preprocess_identity_document(
        self, 
        identity_document: IdentityDocumentImage
    ) -> torch.Tensor:
        """
        Method for preprocessing the image of the identity document to a tensor which can be used
        as an input for the extractor model.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation
            of the identity document image.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        pass

    @abstractmethod
    def extract(self, identity_document: IdentityDocumentImage) -> torch.Tensor:
        """
        Method for extracting the embedding vector from the image of the identity document using
        the extractor model.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation
            of the identity document image.

        Returns:
            torch.Tensor: The extracted embedding vector.
        """
        pass
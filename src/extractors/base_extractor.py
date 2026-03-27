"""
Atuhor: Martin Zůbek
Date: 2026-03-26
File: base_extractor.py
Description: This file contains the implementation of the BaseExtractor class. This class is used like a template for all the future extractors.
"""

import torch
from abc import ABC, abstractmethod

# Default flag for using GPU for feature extraction
DEFAULT_USE_GPU: bool = True
# String for selcting GPU
CUDA_STR: str = "cuda"
# String for selcting CPU
CPU_STR: str = "cpu"


class BaseExtractor(ABC):
    """
    Base class for all the extractors. Other extractors should inherit from this class.

    Attributes:
        use_gpu (bool): Flag for using GPU or CPU for feature extractors.
        device (torch.device): Device for feature extractors.
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
        self.device: torch.device = ( torch.device("CUDA_STR") if self.use_gpu and torch.cuda.is_available() else torch.device("CPU_STR"))

    @abstractmethod
    def extract(self, image):
        pass
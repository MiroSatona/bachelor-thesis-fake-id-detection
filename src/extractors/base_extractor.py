"""
Author: Martin Zůbek
Date: 2026-03-26
File: base_extractor.py
Description: This file contains the implementation of the BaseExtractor class. 
This class is used like a template for all the future extractors.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy
import numpy.typing as npt
import torch

from external.TruFor.TruFor_train_test.lib.models.cmx.builder_np_conf import EncoderDecoder
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.constans import DEFAULT_USE_GPU, EXTRACTORS_DIR_STR
from src.extractors.enums import DeviceType


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
            torch.device(DeviceType.CUDA) 
            if self.use_gpu and torch.cuda.is_available() 
            else torch.device(DeviceType.CPU)
        )
        self.root_dir: Path = Path(__file__).resolve().parents[2] / EXTRACTORS_DIR_STR
        self.src_dir: Path = self.root_dir
        self.extractor_model: EncoderDecoder | None = None

    @abstractmethod
    def load_extractor(self) -> None:
        pass

    @abstractmethod
    def preprocess_identity_document(
        self, 
        identity_document: IdentityDocumentImage
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def extract(self, identity_document: IdentityDocumentImage) -> npt.NDArray[numpy.float32]:
        pass
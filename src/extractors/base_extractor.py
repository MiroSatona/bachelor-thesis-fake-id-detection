"""
Atuhor: Martin Zůbek
Date: 2026-03-26
File: base_extractor.py
Description: This file contains the implementation of the BaseExtractor class. This class is used like a template for all the future extractors.
"""

import torch
from pathlib import Path
from abc import ABC, abstractmethod
from external.TruFor.TruFor_train_test.lib.models.cmx.builder_np_conf import EncoderDecoder 

# Default flag for using GPU for feature extraction
DEFAULT_USE_GPU: bool = True
# String for selcting GPU
CUDA_STR: str = "cuda"
# String for selcting CPU
CPU_STR: str = "cpu"
# String for the directory with all the code for extractors
EXTRACTORS_DIR_STR: str = "external"

####################### TruFor specific constants #######################
# String for the directory with the code for TruFor extractor
TRUFOR_STR: str = "TruFor"
# Name of the directory with the source code of the TruFor extractor
TRUFOR_SRC: str = "TruFor_train_test"
# Path to TruFor configuration file trufor_ph3.yaml
TRUFOR_CONFIG: Path = Path(__file__).resolve().parents[2] / EXTRACTORS_DIR_STR / TRUFOR_STR / TRUFOR_SRC / "lib" / "config" / "trufor_ph3.yaml"


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
        self.device: torch.device = ( torch.device("CUDA_STR") if self.use_gpu and torch.cuda.is_available() else torch.device("CPU_STR"))
        self.root_dir: Path = Path(__file__).resolve().parents[2] / EXTRACTORS_DIR_STR
        self.src_dir: Path = self.root_dir
        self.extractor_model: EncoderDecoder | None = None


    @abstractmethod
    def extract(self, image):
        pass

    @abstractmethod
    def load_extractor(self) -> None:
        pass

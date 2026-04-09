
from pathlib import Path

import torch

from src.extractors.base_extractor import BaseExtractor
from src.extractors.swint import SwinExtractor
from src.extractors.iml_vit import IMLViTExtractor
from src.extractors.trufor import TruForExtractor
from src.extractors.mmfusion import MMFusionExtractor

from src.heads.mlp_head import MLPHead
from src.heads.constants import(
    DROPOUT_PRECENTAGE,
    TRUFOR_MMFUISON_INPUT_DIMENSION,
    IML_VIT_SWINT_INPUT_DIMENSION
)

from src.data_processing.data_handler import DataHandler
from src.data_processing.dataset import Dataset


class Model():
    def __init__(
            self, 
            extractor_name:str, 
            device:bool,
            data_path: str,
            batch_size: int | None,
            number_of_epochs: int | None,
            learning_rate: float | None,
            dropout_percentage: float | None

             
    ) -> None:
        
        self.batch_size: int = batch_size
        self.number_of_epochs: int = number_of_epochs
        self.learning_rate: float = learning_rate
        self.data_handler: DataHandler = DataHandler(Path(data_path))
        
        self.dataset: Dataset | None = None
        self.extractor: BaseExtractor | None = None
        self.head: MLPHead | None = None



        input_dimension: int = 0

        if extractor_name == "trufor":
            self.extractor = TruForExtractor(device=device)
            #input_dimmension = 
        elif extractor_name == "mmfusion":
            self.extractor = MMFusionExtractor(device=device)
            #self.input_dimmension =
        elif extractor_name == "swin":
            self.extractor = SwinExtractor(device=device)
            #self.input_dimmension =
        elif extractor_name == "imlvvit":
            self.extractor = IMLViTExtractor(device=device)
            #self.input_dimmension =
        else:
            raise ValueError(
                f"Invalid extractor name was provided: {extractor_name}"
            )

        self.data_handler.load_dataset()
        self.dataset = self.data_handler.get_dataset_for_training()




    def train(self) -> None:
        
        
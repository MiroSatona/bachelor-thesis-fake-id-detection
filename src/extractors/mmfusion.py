"""
Author: Martin Zůbek
Date: 2026-04-01
File: mmfusion.py
Description: This file contains the implementation of the MMFusionExtractor class which is a wrapper
around the MMFusion (Multi-Modal Fusion, ITI-CERTH) for extracting features
from identity document images. The model combines four input modalities: 
RGB, Noiseprint++, Bayar convolution and SRM filter to detect image manipulation anomalies.
"""

import sys
from pathlib import Path

# Add the directory with the MMFusion code to the system path for importing extractor
MMFUSION_DIR: str = str(Path(__file__).resolve().parents[2] / "external" / "MMFusion")
if MMFUSION_DIR not in sys.path:
    sys.path.insert(0, MMFUSION_DIR)

import numpy
import numpy.typing as npt
import torch
import torchvision.transforms.functional
from PIL import Image

from external.MMFusion.configs.cmnext_init_cfg import _C as mmfusion_config
from external.MMFusion.configs.cmnext_init_cfg import update_config
from external.MMFusion.models.cmnext_conf import CMNeXtWithConf
from external.MMFusion.models.modal_extract import ModalitiesExtractor
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.constants import (
    DEFAULT_USE_GPU,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MMFUSION_CHECKPOINT,
    MMFUSION_CONFIG,
    MMFUSION_NAME,
    MMFUSION_NP_WEIGHTS,
)


class MMFusionExtractor(BaseExtractor):
    """
    Class which is wrapper around the MMFusion (Multi-Modal Fusion, ITI-CERTH) for extracting
    features from identity document images.

    Attributes:
        root_dir (Path): The root directory of the MMFusion.
        src_dir (Path): The source code directory of the MMFusion.
        modal_extractor (ModalitiesExtractor | None): The modality extractor for the MMFusion model.
    """
    def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
        """
        Constructor for the MMFusionExtractor class.
        
        Arguments:
            gpu (bool): Flag for using GPU or CPU for feature extractors. Default is True
            
        Returns:
            None
        """
        # Create the instance of the BaseExtractor class
        super().__init__(gpu)
        # Set root directory of the MMFusion
        self.root_dir: Path = self.root_dir / MMFUSION_NAME
        # Set source code directory of the MMFusion
        self.src_dir: Path = self.src_dir / MMFUSION_NAME
        # Modality extractor for MMFusion model (Noseprint++, Bayar convolution and SRM filter)
        self.modal_extractor: ModalitiesExtractor | None = None

    def load_extractor(self) -> None:
        """
        Method for loading the MMFuison extractor model and modality extractor with configuration. 
        If the model is already loaded, it will not reload it.

        Arguments:
            None

        Raise:
            RuntimeError: If the root directory, source directory, configuration file, or checkpoint
            of the MMFusion does not exist.

        Returns:
            None
        """
        # Check if the model is already loaded
        if self.extractor_model is not None:
            return
        
        # Check if the source directory of the MMFusion exists
        if not self.src_dir.exists():
            raise RuntimeError(
                f"MMFusion source code directory does not exist: {self.src_dir}"
        )

        # Check if the root directory of the MMFusion exists
        if not self.root_dir.exists():
            raise RuntimeError(
                f"MMFusion root directory does not exist: {self.root_dir}"
        )

        # Check if the configuration file of the MMFusion exists
        if not MMFUSION_CONFIG.exists():
            raise RuntimeError(
                f"MMFusion configuration file does not exist: {MMFUSION_CONFIG}"
        )

        # Check if the checkpoint file of the MMFusion exists
        if not MMFUSION_CHECKPOINT.exists():
            raise RuntimeError(
                f"MMFusion checkpoint file does not exist: {MMFUSION_CHECKPOINT}"
        )
        
        
        # Load the configuration to the global variable mmfusion_config
        update_config(mmfusion_config, str(MMFUSION_CONFIG))
        
        mmfusion_config.MODEL.defrost()
        mmfusion_config.MODEL.PRETRAINED = None
        mmfusion_config.MODEL.freeze()
        
        # Initialize the modality extractor and the MMFusion model
        self.modal_extractor = ModalitiesExtractor(
            modals=list(mmfusion_config.MODEL.MODALS[1:]),
            noiseprint_path=str(MMFUSION_NP_WEIGHTS) if MMFUSION_NP_WEIGHTS.exists() else None,
        )

        # Initialize the MMFusion model with the configuration
        # REF: external/MMFusion/models/cmnext_conf.py
        #     (App must be builded for get code which is referenced)
        self.extractor_model = CMNeXtWithConf(cfg=mmfusion_config.MODEL)


        # Load the trained weights from the checkpoint
        # REF: external/MMFusion/inference.py
        #    (App must be builded for get code which is referenced)
        checkpoint: dict = torch.load(
            str(MMFUSION_CHECKPOINT), 
            map_location=self.device,
            weights_only=False
        )
        self.extractor_model.load_state_dict(checkpoint["state_dict"])
        self.modal_extractor.load_state_dict(checkpoint["extractor_state_dict"])

        # Move the extractor and modality extractor to the device and set them to evaluation mode
        self.extractor_model.to(self.device)
        self.modal_extractor.to(self.device)
        self.extractor_model.eval()
        self.modal_extractor.eval()

    def preprocess_identity_document(
        self, 
        identity_document: IdentityDocumentImage
    ) -> torch.Tensor:
        """
        Method for preprocessing the image of the identity document to a tensor which can be used
        as input for the MMFusion model.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation 
            of the identity document image.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        # Load the image of the identity document and convert it to RGB to three channels
        identity_document_image: Image.Image = (
            Image.open(identity_document.path).convert("RGB")
        )

        # Normalize the pixel values by deviding by 256.0 to get values in the range [0, 1]
        # REF: external/MMFusion/data/datasets.py
        image_array: npt.NDArray[numpy.float32] = (
            numpy.array(identity_document_image, dtype=numpy.float32) / 256.0
        )

        # Reorganize the dimensions from [Height, Width, Channels] to [Channels, Height, Width]
        image_tensor: torch.Tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Add batch dimension and move the tensor to the device
        return image_tensor.unsqueeze(0).to(self.device)

    def extract(
            self, 
            identity_document: IdentityDocumentImage
    ) -> torch.Tensor:
        """
        Method for extracting a 512-dimensional embedding vector from an identity document image
        using the MMFusion model. The image is processed together with three additional modalities
        (Noiseprint++, Bayar convolution and SRM filter) through the CMNeXt multi-modal backbone,
        which produces four hierarchical feature maps with decreasing spatial resolution.
        The final embedding vector is obtained from the last-stage feature map (1, 512, H/32, W/32)
        by applying global average pooling over the spatial dimensions.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation
            of the identity document image.

        Raise:
            RuntimeError: If the extractor model or modality extractor cannot be loaded.

        Returns:
            torch.Tensor: The embedding vector with shape (512,) and dtype float32.
        """

        # Load the extractor model and modality extractor if they are not loaded yet
        self.load_extractor()

        if self.extractor_model is None or self.modal_extractor is None:
            raise RuntimeError(
                "MMFusion extractor model or modal extractor cannot be loaded."
            )
        
        # Get the MMFusion model and modality extractor
        mmfusion: CMNeXtWithConf = self.extractor_model
        modal_extractor: ModalitiesExtractor = self.modal_extractor

        # Preprocess the image of the identity document to a tensor
        image_tensor: torch.Tensor = self.preprocess_identity_document(identity_document)

        with torch.no_grad():
            
            # Extract the modalities features (Noiseprint++, Bayar convolution and SRM filter)
            # REF: external/MMFusion/models/modal_extract.py 
            #     (App must be builded for get code which is referenced)
            modals: list[torch.Tensor] = modal_extractor(image_tensor)

            # Normalize the RGB image tensor using the ImageNet mean and standard deviation
            image_tensor_noralized: torch.Tensor = torchvision.transforms.functional.normalize(
                image_tensor, 
                mean=IMAGENET_MEAN, 
                std=IMAGENET_STD
            )

            # Concatenate the RGB image tensor for backbone input
            concatenated_input : list[torch.Tensor] = [image_tensor_noralized] + modals

            # Will be returned four outputs from four stages of the backbone
            # stages_output[0]: (1,  64, H/4,  W/4)
            # stages_output[1]: (1, 128, H/8,  W/8)
            # stages_output[2]: (1, 320, H/16, W/16)
            # stages_output[3]: (1, 512, H/32, W/32)
            stages_output: list[torch.Tensor] = mmfusion.backbone(concatenated_input)
            # Get the output from the last stage of the backbone 
            last_stage_output: torch.Tensor = stages_output[-1]

            # Global average pooling to get the embedding vector -> 512
            embedding_vector: torch.Tensor = (
            torch.mean(last_stage_output, dim=[2, 3])
            .squeeze(0)
            )

            return embedding_vector



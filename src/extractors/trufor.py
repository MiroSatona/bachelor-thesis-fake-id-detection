"""
Author: Martin Zůbek
Date: 2026-03-28
File: trufor.py
Description: This file contains the implementation of the TruForExtractor class which as a wrapper 
around the TruFor (CVPR 2023, grip-unina/TruFor) for extracting features from identity 
document images.
"""

from pathlib import Path

import numpy
import numpy.typing as npt
import torch
from PIL import Image

from external.TruFor.TruFor_train_test.lib.config import config as trufor_config
from external.TruFor.TruFor_train_test.lib.config import update_config
from external.TruFor.TruFor_train_test.lib.models.cmx.builder_np_conf import EncoderDecoder
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.constants import DEFAULT_USE_GPU, TRUFOR_CONFIG, TRUFOR_NAME, TRUFOR_SRC


class TruForExtractor(BaseExtractor):
    """
    Class which is wrapper around the TruFor (CVPR 2023, grip-unina/TruFor) for extracting features 
    from identity document images.
    """
    def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
        """
        Constructor for the TruForExtractor class.
        
        Arguments:
            gpu (bool): Flag for using GPU or CPU for feature extractors. Default is True
            
        Returns:
            None
        """
        # Create the instance of the BaseExtractor class
        super().__init__(gpu)
        # Set root directory of the TruFor
        self.root_dir: Path = self.root_dir / TRUFOR_NAME
        # Set source directory of the TruFor
        self.src_dir: Path = self.root_dir / TRUFOR_SRC
    
    def load_extractor(self) -> None:
        """
        Method for loading the TruFor extractor model. If the model is already loaded, 
        it does nothing.
        
        Arguments:
            None

        Raise:
            RuntimeError: If the source directory of the TruFor does not exist 
            or if the extractor model cannot be loaded. 

        Returns:
            None
        """
        # Load the extractor model if it is not loaded
        if self.extractor_model is not None:
            return

        # Check if the source directory of the TruFor exists
        if not self.src_dir.exists():
            raise RuntimeError(f"Cannot find TruFor code in {self.src_dir}.")
        
        # Configure the path to the TruFor configuration file
        update_config(trufor_config, str(TRUFOR_CONFIG))

        # Get the model from the TruFor repository
        self.extractor_model = EncoderDecoder(cfg=trufor_config)
        # Move the model to the device
        self.extractor_model.to(self.device)
        # Set the model to evaluation mode
        self.extractor_model.eval()
    
    
    def preprocess_identity_document(
            self, 
            identity_document: IdentityDocumentImage
        ) -> torch.Tensor:
        """
        Method for preprocessing the image of the identity document to a tensor which can be used 
        as an input for the TruFor model.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation 
            of the identity document image.

        Returns:
            torch.Tensor: The preprocessed image tensor with shape [1, 3, H, W] and dtype float32 
            in the range [0, 1].
        """

        # Load the image of the identity document and convert it to RGB to three channels
        identity_document_image: Image.Image = Image.open(identity_document.path).convert("RGB")

        # Normalize the image to [0, 1]
        # TruFor needs this normalization, because the the NoisePrint++ was trained like this.
        # REF: external/TruFor/TruFor_train_test/lib/models/cmx/encoders/builder_np_conf.py, 
        #      On the line 265
        #      (App must be builded for get code which is referenced)
        noiseprint_convert: npt.NDArray[numpy.float32] = (
            numpy.array(identity_document_image, dtype=numpy.float32) / 255.0
        )

        # Reorganize the dimensions from [Height, Width, Channels] to [Channels, Height, Width]
        image_tensor: torch.Tensor = (
            torch.from_numpy(noiseprint_convert).permute(2, 0, 1)
        )

        # Add batch dimension and move the tensor to the device
        return image_tensor.unsqueeze(0).to(self.device)


    def extract(
        self,
        identity_document: IdentityDocumentImage,
    ) -> npt.NDArray[numpy.float32]:
        """
        Method for extracting a 512-dimensinal embedding vector from an identity document image 
        using the TruFor model. In the first step, the image is processed trough the NoisePrint++.
        The NoisePrint++ features are extracted using DnCNN and then extended to three channels.
        The original RGB features and the NoisePrint++ features are used as an input
        to the TruFor Vision Transformer backbone. The final embedding vector is getted
        from the last stage of the backbone by applying global average pooling.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation 
            of the identity document image.

        Raise:
            RuntimeError: If the extractor model cannot be loaded.

        Returns:
            npt.NDArray[numpy.float32]: The embedding vector with shape (512,) and dtype float32.
        """

        # Load the extractor model
        self.load_extractor()
        if self.extractor_model is None:
            raise RuntimeError("TruFor model cannot be loaded.")

        trufor: EncoderDecoder = self.extractor_model

        # Preprocess the image of the identity document to a tensor
        image_tensor: torch.Tensor = self.preprocess_identity_document(identity_document)

        with torch.no_grad():

            # Extract the NoisePrint++ features
            noiseprint_features: torch.Tensor = trufor.dncnn(image_tensor)

            # Repeat the NoisePrint++ features to have three channels
            noiseprint_features_extended_channels: torch.Tensor = noiseprint_features.repeat(
                1, 3, 1, 1
            )

            # RGB input for the backbone
            image_tensor_input: torch.Tensor = image_tensor

            if trufor.prepro is not None:
                image_tensor_input = trufor.prepro(image_tensor_input)

            # Will be returned four outputs from four stages of the backbone
            # stages_outputs[0]: (1,  64, H/4,  W/4)
            # stages_outputs[1]: (1, 128, H/8,  W/8)
            # stages_outputs[2]: (1, 320, H/16, W/16)
            # stages_outputs[3]: (1, 512, H/32, W/32)
            stages_outputs: list[torch.Tensor] = trufor.backbone.forward_features(
                image_tensor_input,
                noiseprint_features_extended_channels,
            )

        # Get last stage output (1, 512, H/32, W/32)
        last_stage_output: torch.Tensor = stages_outputs[3]

        # Global average pooling to get the embedding vector -> 512
        embedding_vector: npt.NDArray[numpy.float32] = (
            torch.mean(last_stage_output, dim=[2, 3])
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(numpy.float32)
        )

        return embedding_vector
"""
Author: Martin Zůbek
Date: 2026-05-01
File: mmfusion_app.py
Description: File contains the implementation of the MMFusionExtractorApp class which is responsible
for extracting features from identity documents using the MMFusion model for the application.
"""

import gc

import numpy
import numpy.typing
import torch
import torch.nn.functional
import torchvision.transforms.functional

from external.MMFusion.models.cmnext_conf import CMNeXtWithConf
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.constants import IMAGENET_MEAN, IMAGENET_STD
from src.extractors.mmfusion import MMFusionExtractor


class MMFusionExtractorApp(MMFusionExtractor):
    """
    Class responsible for extracting features from identity documents using the MMFusion model
    for the application.

    Attributes:
        None
    """

    def app_extract(
            self, 
            identity_document: IdentityDocumentImage
        ) -> tuple[torch.Tensor, numpy.typing.NDArray[numpy.float32]]:

        """
        Method for extracting features from identity documents using the MMFusion model 
        for the application. It preprocesses the input image, extracts features using 
        the MMFusion model, and returns the embedding vector and a fraud map.

        Arguments:
            identity_document (IdentityDocumentImage): An instance of IdentityDocumentImage class
            representing the input identity document.
        
        Raises:
            RuntimeError: If the MMFusion extractor model or modal extractor cannot be loaded.

        Returns:
            tuple[torch.Tensor, numpy.typing.NDArray[numpy.float32]]: A tuple containing
            the embedding vector as a torch.Tensor and the fraud map as a numpy 
            array of type float32.

        """
        # Load the MMFusion extractor model and the modality extractor
        self.load_extractor()

        if self.extractor_model is None or self.modal_extractor is None:
            raise RuntimeError(
                "MMFusion extractor model or modal extractor cannot be loaded."
            )
        
        # Preprocess the input identity document
        image_tensor: torch.Tensor = self.preprocess_identity_document(identity_document)
        inputed_height: int = image_tensor.shape[2]
        inputed_width: int = image_tensor.shape[3]

        mmfusion: CMNeXtWithConf = self.extractor_model

        with torch.no_grad():
            # Extract the modality tensors using the modality extractor
            modals_tensor:list[torch.Tensor] = self.modal_extractor(image_tensor)

            # Normalize the RGB image tensor using the ImageNet mean and standard deviation
            image_tensor_normalized: torch.Tensor = torchvision.transforms.functional.normalize(
                image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD
            )
            del image_tensor

            # Concatenate the RGB image tensor for backbone input
            concatenated_input: list[torch.Tensor] = [image_tensor_normalized] + modals_tensor
            del image_tensor_normalized, modals_tensor

            # Will be returned four outputs from four stages of the backbone
            # stages_output[0]: (1,  64, H/4,  W/4)
            # stages_output[1]: (1, 128, H/8,  W/8)
            # stages_output[2]: (1, 320, H/16, W/16)
            # stages_output[3]: (1, 512, H/32, W/32)
            stages_output: list[torch.Tensor] = mmfusion.backbone(concatenated_input)
            del concatenated_input

            # Global average pooling to get the embeding vector -> 512
            embedding: torch.Tensor = (
                torch.mean(stages_output[-1], dim=[2, 3])
                .squeeze(0)
                .detach()
                .clone()
            )

            # Segmentation head processes all 4 stages of the backbone
            # and returns segmentation of shape [1, C, H/4, W/4]
            segmentation: torch.Tensor = mmfusion.decode_head(stages_output)
            del stages_output

            # Resemple the segmentation output to the original input image size
            segmentation_resized: torch.Tensor = torch.nn.functional.interpolate(
                segmentation,
                size=(inputed_height, inputed_width),
                mode="bilinear",
                align_corners=False,    
            )
            del segmentation

            # Create a fraud map from the segmentation output
            fraud_map: numpy.typing.NDArray[numpy.float32] = (
                torch.softmax(segmentation_resized, dim=1)[0, 1].cpu().numpy()
            )
            del segmentation_resized

        gc.collect()
        return embedding, fraud_map
"""
Author: Martin Zůbek
Date: 2026-03-28
File: iml_vit.py
Description: This file contains the implementation of the IMLViTExtractor class which as a wrapper 
around the IML-ViT (Image Manipulation Localization via Vision Transformer) for extracting 
features from identity document images.
"""

import sys
from pathlib import Path

# Add the directory with the IML-ViT code to the system path for importing extractor
IML_VIT_DIR: str = str(Path(__file__).resolve().parents[2] / "external" / "IML_VIT")
if IML_VIT_DIR not in sys.path:
    sys.path.insert(0, IML_VIT_DIR)

import numpy
import numpy.typing as npt
import torch
from PIL import Image

from external.IML_VIT.iml_vit_model import iml_vit_model
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.constants import (
    DEFAULT_USE_GPU,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IML_VIT_CHECKPOINT,
    IML_VIT_INPUT_SIZE,
    IML_VIT_NAME,
)


class IMLViTExtractor(BaseExtractor):
    """
    Class which is wrapper around the IML-ViT (Image Manipulation Localization via 
    Vision Transformer) for extracting features from identity document images.

    Attributes:
        root_dir (Path): The root directory of the IML-ViT.
        src_dir (Path): The source code directory of the IML-ViT.
    """
    def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
        """
        Constructor for the IMLViTExtractor class.

        Arguments:
            gpu (bool): Flag for using GPU or CPU for feature extractors. Default is True.

        Returns:
            None
        """
        # Create the instance of the BaseExtractor class
        super().__init__(gpu)        
        # Set root directory of the IML-ViT
        self.root_dir: Path = self.root_dir / IML_VIT_NAME
        # Set source directory of the IML-ViT
        self.src_dir: Path = self.src_dir / IML_VIT_NAME

    def load_extractor(self) -> None:
        """
        Method for loading the IML-ViT extractor model. If the model is already loaded, 
        it does nothing.
        
        Arguments:
            None

        Raise:
            RuntimeError: If the root directory, source directory, or checkpoint 
            of the IML-ViT does not exist. 

        Returns:
            None
        """

        # Load the extractor model if it is not loaded
        if self.extractor_model is not None:
            return
        
        if not self.root_dir.exists():
            raise RuntimeError(f"Cannot find IML-ViT root directory in {self.root_dir}.")

        # Check if the source directory of the IML-ViT exists
        if not self.src_dir.exists():
            raise RuntimeError(f"Cannot find IML-ViT code in {self.src_dir}.")
        
        # Check if the directory with the checkpoint exists
        if not IML_VIT_CHECKPOINT.exists():
            raise RuntimeError(
                f"IML-ViT checkpoint not found at {IML_VIT_CHECKPOINT}. "
                "Download it from the official repository."
            )

        # Create the model with trained checkpoint
        # REF: external/IML_VIT/Demo.ipynb — checkpoint loading section
        #      (App must be builded for get code which is referenced)
        self.extractor_model: iml_vit_model = iml_vit_model(vit_pretrain_path=None)

        # Load the trained checkpoint
        checkpoint = torch.load(str(IML_VIT_CHECKPOINT), map_location=self.device)
        # Load the checkpoint into the model
        self.extractor_model.load_state_dict(checkpoint, strict=True)

        # Move the model to the device and set it to evaluation mode
        self.extractor_model.to(self.device)
        self.extractor_model.eval()

    def preprocess_identity_document(
        self, 
        identity_document: IdentityDocumentImage
    ) -> torch.Tensor:
        """
        Method for processing the image of the identity document to a tensor which can be used
        as an input for the IML-ViT model.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation
              of the identity document image.

        Returns:
            torch.Tensor: The preprocessed image tensor with shape 
            (1, 3, IML_VIT_INPUT_SIZE, IML_VIT_INPUT_SIZE) and dtype float32 normalized 
            by the ImageNet mean and standard deviation.
        """

        # Load the image of the identity document and convert it to RGB to three channels
        with Image.open(identity_document.path) as image:
            identity_document_image: Image.Image = image.convert("RGB")

        # Get size of the image
        image_width: int = 0    
        image_height: int = 0
        image_width, image_height = identity_document_image.size

        # Get target size for the image which is the maximum of the original 
        # size and the input size of the IML-ViT model
        target_width: int = max(image_width, IML_VIT_INPUT_SIZE)
        target_height: int = max(image_height, IML_VIT_INPUT_SIZE)

        # If the original size of the image is smaller than the input size of the IML-ViT model, 
        # will be padded with black pixels.
        #
        # This and other proccessing steps in this method are based
        # on the code from the IML-ViT repository.
        #
        # REF: external/IML_VIT/utils/datasets.py
        #      external/IML_VIT/utils/iml_transforms.py
        #      (App must be builded for get code which is referenced)
        if target_width != image_width or target_height != image_height:
            padded_identity_document: Image.Image = Image.new(
                "RGB", 
                (target_width, target_height),
                (0, 0, 0)
            )
            # Paste the original image to the left-top corner of the padded image
            padded_identity_document.paste(identity_document_image, (0, 0))
            identity_document_image = padded_identity_document

        # Crop the image to the left-top region to the required input size
        crop_box: tuple[float, float, float, float] = (
            0.0,
            0.0,
            float(IML_VIT_INPUT_SIZE),
            float(IML_VIT_INPUT_SIZE),
        )
        cropped_identity_document: Image.Image = identity_document_image.crop(crop_box)

        # Normalize the image to [0, 1]
        image_array: npt.NDArray[numpy.float32] = (
            numpy.array(cropped_identity_document, dtype=numpy.float32) / 255.0
        )

        # Normalize the image using the ImageNet mean
        imagenet_mean: npt.NDArray[numpy.float32] = numpy.array(
            IMAGENET_MEAN,
            dtype=numpy.float32
        )

        # Normalize the image using the ImageNet standard deviation
        imagenet_std: npt.NDArray[numpy.float32] = numpy.array(
            IMAGENET_STD,
            dtype=numpy.float32
        )

        # Uset the ImageNet mean and standard deviation to normalize the image
        preprocess_image: npt.NDArray[numpy.float32] = (image_array - imagenet_mean) / imagenet_std

        # Reorganize the dimensions from [Height, Width, Channels] to [Channels, Height, Width]
        image_tensor: torch.Tensor = torch.from_numpy(preprocess_image).permute(2, 0, 1)

        # Add batch dimension and move the tensor to the device
        return image_tensor.unsqueeze(0).to(self.device)
    
    def extract(
        self,
        identity_document: IdentityDocumentImage,
    ) -> torch.Tensor:
        """
        Method for extracting a 768-dimensional embedding vector from an identity document image 
        using the IML-ViT model. The image is processed through the Vision Transformer encoder 
        with window-based attention, which produces a spatial feature map 
        (last_feat in the IML-ViT implementation). The final embedding vector is obtained 
        from this feature map by applying global average pooling.

        Arguments:
            identity_document (IdentityDocumentImage): The object for internal representation
            of the identity document image.

        Raise:
            RuntimeError: If the extractor model cannot be loaded.

        Returns:
            torch.Tensor: The embedding vector with shape (768,) and dtype float32.
        """

        # Load the extractor model
        self.load_extractor()
        if self.extractor_model is None:
            raise RuntimeError("Extractor model cannot be loaded.")
        
        # Get the model for extracting the features
        iml_vit: iml_vit_model = self.extractor_model
        # Preprocess the image of the identity document to a tensor
        image_tensor: torch.Tensor = self.preprocess_identity_document(identity_document)
        
        with torch.no_grad():

            # Extract the features and get the last feature map from the encoder
            # Returns a dictionary: {'last_feat': tensor shape (1, 768, 64, 64)}
            encoder_outputs: dict[str, torch.Tensor] = iml_vit.encoder_net(image_tensor)

        # Get the last feature map from the encoder output
        last_feat: torch.Tensor = encoder_outputs["last_feat"]

        # Global average pooling over spatial dimensions (H, W) -> 768
        embedding_vector: torch.Tensor = (
            torch.mean(last_feat, dim=[2,3])
            .squeeze(0)
        )

        return embedding_vector
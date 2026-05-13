"""
Author: Martin Zůbek
Date: 2026-03-28
File: swint.py
Description: This file contains the implementation of SwinExtractor class which is a wrapper around
the Swin Transformer for extracting features from identity document images.
"""

import sys
from pathlib import Path
from typing import Any

# Add the directory with the Swin Transformer code to the system path for importing extractor
SWINT_DIR: str = str(Path(__file__).resolve().parents[2] / "external" / "SwinT")
if SWINT_DIR not in sys.path:
    sys.path.insert(0, SWINT_DIR)

import torch
from PIL import Image

from external.SwinT import config, models
from external.SwinT.data.build import build_transform
from external.SwinT.models.swin_transformer import SwinTransformer
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.base_extractor import BaseExtractor
from src.extractors.constants import (
    DEFAULT_USE_GPU,
    SWINT_CONFIG,
    SWINT_DIR_MODELS,
    SWINT_NAME,
    SWINT_PRETRAINED_PATH,
)


class SwinExtractor(BaseExtractor):
	"""
	Class which is a wrapper around the Swin Transformer for extracting
	features from identity document images.

	Attributes:
		root_dir (Path): The root directory of the Swin Transformer implementation.
		src_dir (Path): The source code directory of the Swin Transformer models.
		transform (Any | None): The transformation pipeline for preprocessing 
		the input identity document images.
	"""
	def __init__(self, gpu: bool = DEFAULT_USE_GPU) -> None:
		"""
		Constructor for the SwinExtractor class.

		Arguments:
			gpu (bool): Flag for using GPU or CPU for feature extraction. Default is True.

		Returns:
			None
		"""
		# Create the instance of the BaseExtractor class
		super().__init__(gpu)
		# Set root directory of the Swin Transformer implementation
		self.root_dir = self.root_dir / SWINT_NAME
		# Set source code directory of the Swin Transformer models
		self.src_dir = self.root_dir / SWINT_DIR_MODELS
		# Transformation pipeline for preprocessing the input identity document images
		self.transform: Any | None = None


	def load_extractor(self) -> None:
		"""
		Method for loading the Swin Transformer extractor model and preprocessing transform. 
		If the model is already loaded, it does nothing.

		Arguments:
			None

		Raise:
			RuntimeError: If the root directory, source code directory, configuration file, 
			or pretrained weights of the Swin Transformer cannot be found.

		Returns:
			None
		"""

		# Load the extractor model if it is not loaded
		if self.extractor_model is not None:
			return

		# Check if the root directory exists
		if not self.root_dir.exists():
			raise RuntimeError(f"Cannot find Swin implementation in: {self.root_dir}")

		# Check if the source code directory exists
		if not self.src_dir.exists():
			raise RuntimeError(f"Cannot find Swin source code in: {self.src_dir}")

		# Check if the configuration file exists
		if not SWINT_CONFIG.exists():
			raise RuntimeError(f"Cannot find Swin config file in: {SWINT_CONFIG}")

		# Check if the pretrained weights file exists
		if not SWINT_PRETRAINED_PATH.exists():
			raise RuntimeError(
				f"Cannot find Swin pretrained weights in: {SWINT_PRETRAINED_PATH}"
			)

		# Add the root directory of the Swin Transformer implementation to the system path
		if str(self.root_dir) not in sys.path:
			sys.path.append(str(self.root_dir))

		# Load the initial configuration for the Swin Transformer from the config file
		swint_config = config._C.clone()
		config._update_config_from_file(swint_config, str(SWINT_CONFIG))

		# Unfreeze the configuration for editing the values
		swint_config.defrost()
		# Configurate the Swin Transformer for feature extraction
		swint_config.DATA.BATCH_SIZE = 1
		swint_config.DATA.NUM_WORKERS = 0
		swint_config.MODEL.PRETRAINED = str(SWINT_PRETRAINED_PATH)
		swint_config.MODEL.RESUME = ""
		swint_config.LOCAL_RANK = 0
		swint_config.EVAL_MODE = True
		swint_config.THROUGHPUT_MODE = False
		swint_config.FUSED_WINDOW_PROCESS = False
		swint_config.FUSED_LAYERNORM = False
		# Freeze the configuration after editing
		swint_config.freeze()

		# Build the Swin Transformer model using the configuration
		self.extractor_model = models.build_model(swint_config)

		# Load the pretrained weights into the model
		checkpoint_path: str = swint_config.MODEL.PRETRAINED
		checkpoint = torch.load(checkpoint_path, map_location="cpu")
		self.extractor_model.load_state_dict(checkpoint["model"], strict=False)

		# Move the model to the device and set it to evaluation mode
		self.extractor_model.to(self.device)
		self.extractor_model.eval()

		# Build the transformation pipeline for preprocessing the input identity document images
		self.transform = build_transform(
			is_train=False,
			config=swint_config,
		)

	def preprocess_identity_document(
		self,
		identity_document: IdentityDocumentImage,
	) -> torch.Tensor:
		"""
		Method for preprocessing the image of the identity document, which used the 
		Swin Tranformer tranformation for the right preprocessing of the input 
		identity document image for the Swin Transformer model.

		Arguments:
			identity_document (IdentityDocumentImage): The object for internal representation
			of the identity document image.

		Raise:
			RuntimeError: If the transformation pipeline cannot be built.

		Returns:
			torch.Tensor: The preprocessed image tensor with shape [1, 3, H, W] and dtype float32,
			values are ImageNet-normalized and can be negative or greater than 1.0.
		"""
		
		# If the transformation pipeline is not built, load the extractor model to build it
		if self.transform is None:
			self.load_extractor()

		# If the transformation pipeline is still not built, raise an error
		if self.transform is None:
			raise RuntimeError("Swin preprocessing transform cannot be built.")

		# Load the image of the identity document and convert it to RGB to three channels
		identity_document_image: Image.Image = Image.open(identity_document.path).convert("RGB")
		# Apply the transformation pipeline on the image
		image_tensor: torch.Tensor = self.transform(identity_document_image)

		# Add batch dimension and move the tensor to the device
		return image_tensor.unsqueeze(0).to(self.device)

	def extract(
		self,
		identity_document: IdentityDocumentImage,
	) -> torch.Tensor:
		"""
		Method for extracting the feature vector from the image of the identity document using 
		the Swin Transformer model.

		Arguments:
			identity_document (IdentityDocumentImage): The object for internal representation
			of the identity document image.

		Raise:
			RuntimeError: If the Swin Transformer model cannot be loaded.

		Returns:
			torch.Tensor: The extracted feature vector with shape [768] and dtype float32.
		"""
		
		# Load the extractor model if it is not loaded
		self.load_extractor()

		# If the extractor model is still not loaded, raise an error
		if self.extractor_model is None:
			raise RuntimeError("Swin model cannot be loaded.")

		# Get the Swin Transformer model
		swint: SwinTransformer = self.extractor_model
		# Preprocess the image of the identity document to a tensor
		image_tensor: torch.Tensor = self.preprocess_identity_document(identity_document)

		# Extract the feature vector from the image tensor using the Swin Transformer model.
		# forward_features applies avgpool internally and returns shape (B, 768).
		with torch.no_grad():
			embedding: torch.Tensor = swint.forward_features(image_tensor)

		# Remove batch dimension to get the 768 dimensional feature vector
		embedding_vector: torch.Tensor = embedding.squeeze(0)

		return embedding_vector
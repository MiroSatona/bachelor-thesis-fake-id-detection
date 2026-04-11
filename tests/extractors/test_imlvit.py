"""
Author: Martin Zůbek
Date: 2026-04-02
File: test_imlvit.py
Description: Tests for the IMLViTExtractor class, which is a wrapper around the IML-ViT 
(Image Manipulation Localization via Vision Transformer) model for extracting features 
from identity document images. The tests verify the correct usage of the IML-ViT model 
and the integration between the IML_VIT repository and this project.
"""

from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image

from external.IML_VIT.iml_vit_model import iml_vit_model
from src.data_processing.enums import DatasetSubset, Label
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.constants import IML_VIT_INPUT_SIZE
from src.extractors.iml_vit import IMLViTExtractor


@pytest.fixture
def create_identity_document(tmp_path: Path) -> IdentityDocumentImage:
    """
    Pytest fixture for creating a testing IdentityDocumentImage 
    with a temporary 64x64 RGB PNG image.
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest 
        for storing the test image.
        
    Returns:
        IdentityDocumentImage: An instance of IdentityDocumentImage pointing 
        to the created test image.
    """
    identity_document_image_path: Path = tmp_path / "identity_document.png"
    created_identity_document_image: Image.Image = Image.fromarray(
        numpy.random.randint(0, 256, (64, 64, 3),
        dtype=numpy.uint8
        ), mode="RGB"
    )
    created_identity_document_image.save(identity_document_image_path)
    return IdentityDocumentImage(
        path=identity_document_image_path,
        label=Label.REAL,
        attack_type=None,
        dataset_part=None,
        dataset_subset=DatasetSubset.TEST,
    )

@pytest.fixture
def create_large_identity_document(tmp_path: Path) -> IdentityDocumentImage:
    """
    Pytest fixture for creating a testing IdentityDocumentImage with a temporary 1500x1200 RGB PNG 
    image, which is larger than the expected input size of the IML-ViT model (1024x1024).
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for storing the test image.
        
    Returns:
        IdentityDocumentImage: An instance of IdentityDocumentImage pointing to the created 
        test image.
    """
    large_identity_document_image_path: Path = tmp_path / "identity_document.png"
    created_large_identity_document_image: Image.Image = Image.fromarray(
        numpy.random.randint(0, 256, (1500, 1200, 3),
        dtype=numpy.uint8
        ), mode="RGB"
    )
    created_large_identity_document_image.save(large_identity_document_image_path)
    return IdentityDocumentImage(
        path=large_identity_document_image_path,
        label=Label.REAL,
        attack_type=None,
        dataset_part=None,
        dataset_subset=DatasetSubset.TEST,
    )

@pytest.fixture
def instantiate_iml_vit_extractor() -> IMLViTExtractor:
    """
    Pytest fixture for instantiating the IMLViTExtractor on CPU.
    
    Arguments:
        None
        
    Returns:
        IMLViTExtractor: An instance of the IMLViTExtractor class.
    """
    return IMLViTExtractor(gpu=False)


def test_load_iml_vit_extractor(instantiate_iml_vit_extractor: IMLViTExtractor) -> None:
    """
    Test for the load_extractor() method of the IMLViTExtractor class. This test verifies that the
    load_extractor() method successfully loads the IML-ViT model from the external IML-ViT
    repository.
    
     Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    assert instantiate_iml_vit_extractor.extractor_model is None
    instantiate_iml_vit_extractor.load_extractor()
    assert instantiate_iml_vit_extractor.extractor_model is not None
    assert isinstance(instantiate_iml_vit_extractor.extractor_model, iml_vit_model)

def test_shape_of_preprocess_image(
        instantiate_iml_vit_extractor: IMLViTExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the IMLViTExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly preprocesses 
    the input image and returns a tensor with the expected 
    shape (1, 3, 1024, 1024) and dtype float32.
    
     Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_iml_vit_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert image_tensor.ndim == 4
    assert image_tensor.shape == (1, 3, IML_VIT_INPUT_SIZE, IML_VIT_INPUT_SIZE)
    assert image_tensor.dtype == torch.float32

def test_preproccess_pads_small_images(
        instantiate_iml_vit_extractor: IMLViTExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the IMLViTExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly pads an image
    to at least IML_VIT_INPUT_SIZE(1024) and then crops it from the top-left corner.

    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided 
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_iml_vit_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert image_tensor.shape[2] == IML_VIT_INPUT_SIZE
    assert image_tensor.shape[3] == IML_VIT_INPUT_SIZE



def test_preproccess_crops_large_images(
        instantiate_iml_vit_extractor: IMLViTExtractor, 
        create_large_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the IMLViTExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly processes images larger than 
    1024x1024 by cropping them from the top-left corner to [1, 3, 1024, 1024].

    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided
        by the pytest fixture.
        create_large_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_iml_vit_extractor.preprocess_identity_document(
        create_large_identity_document
    )
    assert image_tensor.shape == (1, 3, IML_VIT_INPUT_SIZE, IML_VIT_INPUT_SIZE)



def test_preprocess_normalizes_images_by_imagenet(
        instantiate_iml_vit_extractor: IMLViTExtractor,
        create_identity_document: IdentityDocumentImage,
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the IMLViTExtractor class. This test
    verifies that the preprocess_identity_document() method applies ImageNet normalization 
    to the input image.

    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_iml_vit_extractor.preprocess_identity_document(
        create_identity_document
    )
    min_normalized_value = float(image_tensor.min())
    max_normalized_value = float(image_tensor.max())
    assert min_normalized_value < 0.0 or max_normalized_value > 1.0


def test_extract_returns_768dim_float32_embedding_vector(
        instantiate_iml_vit_extractor: IMLViTExtractor,
        create_identity_document: IdentityDocumentImage,
    ) -> None:
    """
    Test for the extract() method of the IMLViTExtractor class. This test verifies the full
    extraction pipeline: image -> ViT encoder (window attention) -> last_feat (1, 768, 64, 64)
    
    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    embedding: torch.Tensor = instantiate_iml_vit_extractor.extract(
        create_identity_document
    )
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (768,)
    assert embedding.dtype == torch.float32


def test_lazy_loading(
        instantiate_iml_vit_extractor: IMLViTExtractor
    ) -> None:
    """
    Tests that calling load_extractor() multiple times is and testing if the model is not reload,
    the same model instance is reused across repeated calls.
    
    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided 
        by the pytest fixture.
        
    Returns:
        None
    """
    instantiate_iml_vit_extractor.load_extractor()
    model_after_first_load = instantiate_iml_vit_extractor.extractor_model

    instantiate_iml_vit_extractor.load_extractor()
    model_after_second_load = instantiate_iml_vit_extractor.extractor_model

    assert model_after_first_load is model_after_second_load


def test_validate_embedding_vector_values(
        instantiate_iml_vit_extractor: IMLViTExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Tests that the embedding vector values extracted by the IMLViTExtractor 
    are valid float32 numbers.
    
    Arguments:
        instantiate_iml_vit_extractor (IMLViTExtractor): An IMLViTExtractor instance provided 
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    embedding_vector: torch.Tensor = instantiate_iml_vit_extractor.extract(
        create_identity_document
    )
    assert torch.all(torch.isfinite(embedding_vector))
    assert not torch.all(embedding_vector == 0.0)

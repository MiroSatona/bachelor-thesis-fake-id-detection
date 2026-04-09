"""
Author: Martin Zůbek
Date: 2026-04-02
File: test_swint.py
Description: Tests for the SwinExtractor class, which is a wrapper around the Swin Transformer 
(Microsoft/Swin-Transformer) model for extracting features from identity document images. 
The tests verify the correct usage of the Swin Transformer model and the integration 
between the SwinT repository and this project.
"""

from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image

from external.SwinT.models.swin_transformer import SwinTransformer
from src.data_processing.enums import DatasetSubset, Label
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.swint import SwinExtractor


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
    Pytest fixture for creating a testing IdentityDocumentImage with a temporary 512x512 RGB PNG 
    image, which is larger than the expected input size of the Swin Transformer model (224x224).
    
    Arguments:
        tmp_path (Path): A temporary directory provided by pytest for storing the test image.
        
    Returns:
        IdentityDocumentImage: An instance of IdentityDocumentImage pointing to the created 
        test image.
    """
    large_identity_document_image_path: Path = tmp_path / "identity_document.png"
    created_large_identity_document_image: Image.Image = Image.fromarray(
        numpy.random.randint(0, 256, (512, 512, 3),
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
def instantiate_swin_extractor() -> SwinExtractor:
    """
    Pytest fixture for instantiating the SwinExtractor on CPU.
    
    Arguments:
        None
        
    Returns:
        SwinExtractor: An instance of the SwinExtractor class.
    """
    return SwinExtractor(gpu=False)

def test_load_swin_extractor(instantiate_swin_extractor: SwinExtractor) -> None:
    """
    Test for the load_extractor() method of the SwinExtractor class. This test verifies that the
    load_extractor() method successfully loads the Swin Transformer model from the external SwinT
    repository.
    
     Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    assert instantiate_swin_extractor.extractor_model is None
    instantiate_swin_extractor.load_extractor()
    assert instantiate_swin_extractor.extractor_model is not None
    assert isinstance(instantiate_swin_extractor.extractor_model, SwinTransformer)

def test_shape_of_preprocess_image(
        instantiate_swin_extractor: SwinExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the SwinExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly preprocesses 
    the input image and returns a tensor with the expected 
    shape (1, 3, 224, 224) and dtype float32.
    
     Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_swin_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert image_tensor.ndim == 4
    assert image_tensor.shape == (1, 3, 224, 224)
    assert image_tensor.dtype == torch.float32

def test_preprocess_resizes_small_images(
        instantiate_swin_extractor: SwinExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the SwinExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly resizes an image
    and then center-crops it to 224x224.

    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided 
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_swin_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert image_tensor.shape[2] == 224
    assert image_tensor.shape[3] == 224

def test_preprocess_crops_large_images(
        instantiate_swin_extractor: SwinExtractor, 
        create_large_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the SwinExtractor class. This test 
    verifies that the preprocess_identity_document() method correctly resizes an image
    and then center-crops it to 224x224.

    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        create_large_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_swin_extractor.preprocess_identity_document(
        create_large_identity_document
    )
    assert image_tensor.shape == (1, 3, 224, 224)


def test_preprocess_normalizes_images_by_imagenet(
        instantiate_swin_extractor: SwinExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the SwinExtractor class. This test
    verifies that the preprocess_identity_document() method applies ImageNet normalization 
    to the input image.

    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.
    
    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_swin_extractor.preprocess_identity_document(
        create_identity_document
    )
    min_normalized_value = float(image_tensor.min())
    max_normalized_value = float(image_tensor.max())
    assert min_normalized_value < 0.0 or max_normalized_value > 1.0

def test_extract_and_get_768dim_embedding_vector(
        instantiate_swin_extractor: SwinExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the extract() method of the SwinExtractor class. This test verifies the full
    extraction pipeline: image -> Swin Transformer encoder 
    -> global average pooling -> 768-dimensional float32 embedding vector.

    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    embedding_vector: torch.Tensor = instantiate_swin_extractor.extract(
        create_identity_document
    )
    assert isinstance(embedding_vector, torch.Tensor)
    assert embedding_vector.shape == (768,)
    assert embedding_vector.dtype == torch.float32

def test_lazy_loading(
        instantiate_swin_extractor: SwinExtractor
    ) -> None:
    """
    Tests that calling load_extractor() multiple times does not reload the model -
    the same model instance is reused across repeated calls.
    
    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided by the pytest
        fixture.
        
    Returns:
        None
    """
    instantiate_swin_extractor.load_extractor()
    instance_after_first_load = instantiate_swin_extractor.extractor_model
    instantiate_swin_extractor.load_extractor()
    instance_after_second_load = instantiate_swin_extractor.extractor_model
    assert instance_after_first_load is instance_after_second_load

def test_extract_returns_768dim_float32_embedding_vector(
        instantiate_swin_extractor: SwinExtractor, 
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Tests that the embedding vector values extracted by the SwinExtractor are valid float32 numbers.
    
    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided 
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    embedding_vector: torch.Tensor = instantiate_swin_extractor.extract(
        create_identity_document
    )
    assert torch.all(torch.isfinite(embedding_vector))
    assert not torch.all(embedding_vector == 0.0)

def test_transform_is_built_on_load(
        instantiate_swin_extractor: SwinExtractor
    ) -> None:
    """
    Tests that the image transformation pipeline (self.transform) is built when the SwinExtractor
    is loaded.
    
    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided by the pytest
        fixture.
        
    Returns:
        None
    """
    assert instantiate_swin_extractor.transform is None
    instantiate_swin_extractor.load_extractor()
    assert instantiate_swin_extractor.transform is not None

def test_eval_mode_is_set_on_load(
        instantiate_swin_extractor: SwinExtractor
    ) -> None:
    """
    Tests that the SwinTransformer model is set to evaluation mode (model.eval())
    when loaded by the SwinExtractor.

    Arguments:
        instantiate_swin_extractor (SwinExtractor): A SwinExtractor instance provided
        by the pytest fixture.
        
    Returns:
        None
    """

    instantiate_swin_extractor.load_extractor()
    assert instantiate_swin_extractor.extractor_model is not None
    assert not instantiate_swin_extractor.extractor_model.training
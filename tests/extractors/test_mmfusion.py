"""
Author: Martin Zůbek
Date: 2026-04-05
File: test_mmfusion.py
Description: Tests for the MMFusionExtractor class, which is a wrapper around the MMFusion
 model for extracting features from identity document images. The tests verify the correct
usage of the MMFusion model and the integration between the MMFusion repository and
this project.

ACKNOWLEDGMENT:
Parts of this code were generated with the assistance of GitHub Copilot
and subsequently modified and adapted by the author.
"""

from pathlib import Path

import numpy
import pytest
import torch
from PIL import Image

from external.MMFusion.models.cmnext_conf import CMNeXtWithConf
from external.MMFusion.models.modal_extract import ModalitiesExtractor
from src.data_processing.enums import DatasetSubset, Label
from src.data_processing.identity_document import IdentityDocumentImage
from src.extractors.mmfusion import MMFusionExtractor


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
def instantiate_mmfusion_extractor() -> MMFusionExtractor:
    """
    Pytest fixture for instantiating the MMFusionExtractor on CPU.
    
    Arguments:
        None
        
    Returns:
        MMFusionExtractor: An instance of the MMFusionExtractor class.
    """
    return MMFusionExtractor(gpu=False)


def test_load_mmfusion_extractor(instantiate_mmfusion_extractor: MMFusionExtractor) -> None:
    """
    Test for the load_extractor() method of the MMFusionExtractor class. This test verifies that
    the load_extractor() method successfully loads the CMNeXtWithConf model and the
    ModalitiesExtractor from the external MMFusion repository.
    
     Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    assert instantiate_mmfusion_extractor.extractor_model is None
    assert instantiate_mmfusion_extractor.modal_extractor is None
    instantiate_mmfusion_extractor.load_extractor()
    assert instantiate_mmfusion_extractor.extractor_model is not None
    assert instantiate_mmfusion_extractor.modal_extractor is not None
    assert isinstance(instantiate_mmfusion_extractor.extractor_model, CMNeXtWithConf)
    assert isinstance(instantiate_mmfusion_extractor.modal_extractor, ModalitiesExtractor)


def test_shape_of_preprocess_image(
        instantiate_mmfusion_extractor: MMFusionExtractor,
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the MMFusionExtractor class. This test
    verifies that the preprocess_identity_document() method correctly preprocesses
    the input image and returns a float32 tensor with shape (1, 3, H, W).
    
     Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
     Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_mmfusion_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert image_tensor.ndim == 4
    assert image_tensor.shape[0] == 1
    assert image_tensor.shape[1] == 3
    assert image_tensor.dtype == torch.float32


def test_preprocess_normalizes_images_to_01_range(
        instantiate_mmfusion_extractor: MMFusionExtractor,
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the preprocess_identity_document() method of the MMFusionExtractor class. This test
    verifies that the preprocess_identity_document() method normalizes pixel values to [0, 1)
    by dividing by 256.0, consistent with MMFusion's ManipulationDataset.

    Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    image_tensor: torch.Tensor = instantiate_mmfusion_extractor.preprocess_identity_document(
        create_identity_document
    )
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) < 1.0


def test_extract_returns_512dim_float32_embedding_vector(
        instantiate_mmfusion_extractor: MMFusionExtractor,
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Test for the extract() method of the MMFusionExtractor class. This test verifies the full
    extraction pipeline: image -> ModalitiesExtractor -> CMNeXt backbone
    -> global average pooling -> 512-dimensional float32 embedding vector.
    The 512 channels correspond to the last stage of the MixCMNeXtMHSA-B2 backbone.

    Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance
        provided by the pytest fixture.

    Returns:
        None
    """
    embedding_vector: torch.Tensor = instantiate_mmfusion_extractor.extract(
        create_identity_document
    )
    assert isinstance(embedding_vector, torch.Tensor)
    assert embedding_vector.shape == (512,)
    assert embedding_vector.dtype == torch.float32


def test_lazy_loading(
        instantiate_mmfusion_extractor: MMFusionExtractor
    ) -> None:
    """
    Tests that calling load_extractor() multiple times does not reload the model. The same
    model and modality extractor instances are reused across repeated calls.
    
    Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    instantiate_mmfusion_extractor.load_extractor()
    model_after_first_load = instantiate_mmfusion_extractor.extractor_model
    modal_extractor_after_first_load = instantiate_mmfusion_extractor.modal_extractor

    instantiate_mmfusion_extractor.load_extractor()
    model_after_second_load = instantiate_mmfusion_extractor.extractor_model
    modal_extractor_after_second_load = instantiate_mmfusion_extractor.modal_extractor

    assert model_after_first_load is model_after_second_load
    assert modal_extractor_after_first_load is modal_extractor_after_second_load


def test_validate_embedding_vector_values(
        instantiate_mmfusion_extractor: MMFusionExtractor,
        create_identity_document: IdentityDocumentImage
    ) -> None:
    """
    Tests that the embedding vector values extracted by the MMFusionExtractor are valid
    float32 numbers.
    
    Arguments:
        instantiate_mmfusion_extractor (MMFusionExtractor): An MMFusionExtractor instance provided
        by the pytest fixture.
        create_identity_document (IdentityDocumentImage): An IdentityDocumentImage instance provided
        by the pytest fixture.
        
    Returns:
        None
    """
    embedding_vector: torch.Tensor = instantiate_mmfusion_extractor.extract(
        create_identity_document
    )
    assert torch.all(torch.isfinite(embedding_vector))
    assert not torch.all(embedding_vector == 0.0)

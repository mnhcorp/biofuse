import unittest
from unittest.mock import patch
import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image

class TestBioFuseModel(unittest.TestCase):
    #dict matching model names to their expected output dimensions
    model_dims = {
        "BioMedCLIP": 512,
        "BioMistral": 4096,
        "CheXagent": 1408,
        "CONCH": 512,
        "LLama-3-Aloe": 4096,
        "Prov-GigaPath": 1536,
        "PubMedCLIP": 512,
        "rad-dino": 768,
        "UNI": 1024
    }

    def setUp(self):
        self.model_names = ["BioMedCLIP", "rad-dino"]
        self.fusion_method = "concat"
        self.biofuse_model = BioFuseModel(self.model_names, self.fusion_method)
        self.text = "Patient has a fracture in the left arm."
        self.image = Image.open("data/xray.jpg")

    # def test_initialize_embedding_extractors(self):
    #     self.assertEqual(len(self.biofuse_model.embedding_extractors), len(self.model_names))
    #     self.assertIsInstance(self.biofuse_model.embedding_extractors[0], EmbeddingExtractor)

    def test_forward_pass_concat(self):
        # Create sample input data
        input_data = self.image

        output = self.biofuse_model(input_data)

        expected_output_dim = sum([TestBioFuseModel.model_dims[model] for model in self.model_names])
        # print("Expected output dim: ", expected_output_dim)
        # print("Output shape: ", output.shape)
        self.assertEqual(output.shape, (1, expected_output_dim))     

    # def test_forward_pass_avg(self):
    #     # Create sample input data
    #     input_data = self.image

    #     # Change the fusion method to "avg"
    #     self.biofuse_model.fusion_method = "avg"

    #     # Mock the forward pass of the embedding extractors
    #     with patch.object(EmbeddingExtractor, "forward", return_value=torch.randn(1, 512)):
    #         # Call the forward method
    #         output = self.biofuse_model(input_data)

    #         # Assert the output shape
    #         self.assertEqual(output.shape, (1, 512))

    def test_invalid_fusion_method(self):
        # Create sample input data
        input_data = self.image

        # Change the fusion method to an invalid value
        self.fusion_method = "invalid"
        self.biofuse_model = BioFuseModel(self.model_names, self.fusion_method)

        # Assert that a ValueError is raised
        with self.assertRaises(ValueError):
            self.biofuse_model(input_data)

if __name__ == "__main__":
    unittest.main()

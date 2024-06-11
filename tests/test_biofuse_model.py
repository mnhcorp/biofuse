import unittest
from unittest.mock import patch
import torch
from biofuse.models.biofuse_model import BioFuseModel
from biofuse.models.embedding_extractor import EmbeddingExtractor

class TestBioFuseModel(unittest.TestCase):
    def setUp(self):
        self.model_names = ["BioMedCLIP", "BioMistral"]
        self.fusion_method = "concat"
        self.biofuse_model = BioFuseModel(self.model_names, self.fusion_method)

    def test_initialize_embedding_extractors(self):
        self.assertEqual(len(self.biofuse_model.embedding_extractors), len(self.model_names))
        self.assertIsInstance(self.biofuse_model.embedding_extractors[0], EmbeddingExtractor)

    def test_forward_pass_concat(self):
        # Create sample input data
        input_data = torch.randn(3, 224, 224)

        # Mock the forward pass of the embedding extractors
        with patch.object(EmbeddingExtractor, "forward", return_value=torch.randn(1, 512)):
            # Call the forward method
            output = self.biofuse_model(input_data)

            # Assert the output shape
            expected_output_dim = 512 * len(self.model_names)
            self.assertEqual(output.shape, (1, expected_output_dim))

    def test_forward_pass_avg(self):
        # Create sample input data
        input_data = torch.randn(3, 224, 224)

        # Change the fusion method to "avg"
        self.biofuse_model.fusion_method = "avg"

        # Mock the forward pass of the embedding extractors
        with patch.object(EmbeddingExtractor, "forward", return_value=torch.randn(1, 512)):
            # Call the forward method
            output = self.biofuse_model(input_data)

            # Assert the output shape
            self.assertEqual(output.shape, (1, 512))

    def test_invalid_fusion_method(self):
        # Create sample input data
        input_data = torch.randn(3, 224, 224)

        # Change the fusion method to an invalid value
        self.biofuse_model.fusion_method = "invalid"

        # Assert that a ValueError is raised
        with self.assertRaises(ValueError):
            self.biofuse_model(input_data)

if __name__ == "__main__":
    unittest.main()

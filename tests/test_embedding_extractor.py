import unittest
from unittest.mock import patch
import torch
from biofuse.models.embedding_extractor import PreTrainedEmbedding

class TestPreTrainedEmbedding(unittest.TestCase):
    def setUp(self):
        self.model_name = "rad-dino"
        self.extractor = PreTrainedEmbedding(self.model_name)

    def test_load_model(self):
        self.assertIsNotNone(self.extractor.model)
        self.assertIsNotNone(self.extractor.processor)

    def test_rad_dino(self):
        # Create a sample image tensor
        image_tensor = torch.randn(3, 224, 224)

        output = self.extractor(image_tensor)

        self.assertEqual(output.shape, (1, 768))

    # def test_forward_pass_biomedclip(self):
    #     # Create a sample image tensor
    #     image_tensor = torch.randn(3, 224, 224)

    #     # Mock the tokenizer and model output
    #     with patch("open_clip.get_tokenizer"), patch("open_clip.create_model_from_pretrained") as mock_create_model:
    #         mock_model = mock_create_model.return_value[0]
    #         mock_model.encode_image.return_value = torch.randn(1, 512)

    #         # Call the forward method
    #         output = self.extractor(image_tensor)

    #         # Assert the output shape
    #         self.assertEqual(output.shape, (1, 512))

    # def test_forward_pass_biomistral(self):
    #     # Create a sample text input
    #     text_input = "Sample text"

    #     # Mock the tokenizer and model output
    #     with patch("transformers.AutoTokenizer"), patch("transformers.AutoModel") as mock_auto_model:
    #         mock_model = mock_auto_model.return_value
    #         mock_model.return_value.last_hidden_state = torch.randn(1, 768)

    #         # Call the forward method
    #         output = self.extractor(text_input)

    #         # Assert the output shape
    #         self.assertEqual(output.shape, (1, 768))

    # Add more test cases for other models and input types

if __name__ == "__main__":
    unittest.main()

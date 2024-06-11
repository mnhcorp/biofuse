import unittest
from unittest.mock import patch
import torch
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from PIL import Image
import os

class TestPreTrainedEmbedding(unittest.TestCase):
    def setUp(self):
        self.image = Image.open("data/xray.jpg")
        self.text = "Patient has a fracture in the left arm."

    # def test_load_model(self):
    #     self.assertIsNotNone(self.extractor.model)
    #     self.assertIsNotNone(self.extractor.processor)

    def test_rad_dino(self):     
        self.model_name = "rad-dino"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.image)

        self.assertEqual(output.shape, (1, 768))

    def test_biomedclip(self):
        self.model_name = "BioMedCLIP"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.image)

        self.assertEqual(output.shape, (1, 512))

    def test_biomistral(self):
        self.model_name = "BioMistral"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.text)

        self.assertEqual(output.shape, (1, 4096))

    def test_chexagent(self):
        self.model_name = "CheXagent"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.image)

        self.assertEqual(output.shape, (1, 1408))

    
    # Tear down method
    def tearDown(self):
        # close the image
        self.image.close()

if __name__ == "__main__":
    unittest.main()

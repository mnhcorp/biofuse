import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader
from biofuse.models.embedding_extractor import PreTrainedEmbedding
from biofuse.models.image_dataset import BioFuseImageDataset
from biofuse.models.processor import MultiModelPreprocessor
from PIL import Image
import os

class TestPreTrainedEmbedding(unittest.TestCase):
    def setUp(self):
        #self.image = Image.open("data/xray.jpg")
        #self.image = torch.randn(3, 224, 224)
        #self.hist_image = Image.open("data/hist.jpg")
        self.image = "data/hist.jpg"
        self.dummy_label = 0
        self.dataset = BioFuseImageDataset([self.image], [self.dummy_label])
        self.dataloder = DataLoader(self.dataset, batch_size=1, shuffle=False, collate_fn=self.custom_collate_fn)
        self.text = "Patient has a fracture in the left arm."

    def custom_collate_fn(self, batch):
        images, labels = zip(*batch)
        return list(images), torch.tensor(labels)

    def test_rad_dino(self):     
        model_name = "rad-dino"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, (1, 768))

    def test_biomedclip(self):
        model_name = "BioMedCLIP"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, (1, 512))

    def test_pubmedclip(self):
        model_name = "PubMedCLIP"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, (1, 512))

    def test_uni(self):
        model_name = "UNI"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, torch.Size([1024]))

    def test_conch(self):
        model_name = "CONCH"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, (1, 512))

    def test_prov_gigapath(self):
        model_name = "Prov-GigaPath"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image[0])
        
        self.assertEqual(output.shape, torch.Size([1536]))

    def test_chexagent(self):
        model_name = "CheXagent"
        processor = MultiModelPreprocessor([model_name])
        extractor = PreTrainedEmbedding(model_name)

        input, label = next(iter(self.dataloder))
        processed_image = processor.preprocess(input[0])
        output = extractor(processed_image)
        
        self.assertEqual(output.shape, (1, 1408))



    """
    def test_biomistral(self):
        self.model_name = "BioMistral"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.text)

        self.assertEqual(output.shape, (1, 4096))

    
    # CONCH
    
    
    # UNI
    
    # Prov-GigaPath
    
    # LLama-3-Aloe
    def test_llama_3_aloe(self):
        self.model_name = "LLama-3-Aloe"
        self.extractor = PreTrainedEmbedding(self.model_name)

        output = self.extractor(self.text)

        self.assertEqual(output.shape, (1, 4096))

    # Tear down method
    def tearDown(self):
        # close the image
        self.image.close()
    """

if __name__ == "__main__":
    unittest.main()

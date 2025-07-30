import unittest
import os
import json
import torch
from PIL import Image
import io
import numpy as np
from model.glide_model import GLIDELikeModel
from model_loader import ModelLoader
from app import app

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = GLIDELikeModel(device=cls.device)
        cls.test_prompt = "a delicious plate of food at a restaurant"
        
    def test_model_initialization(self):
        self.assertIsNotNone(self.model.text_encoder)
        self.assertIsNotNone(self.model.unet)
        self.assertIsNotNone(self.model.diffusion)
        
    def test_save_and_load_model(self):
        test_path = "test_model.pt"
        try:
            self.model.save_model(test_path)
            self.assertTrue(os.path.exists(test_path))
            
            new_model = GLIDELikeModel(device=self.device)
            new_model.load_model(test_path)
            self.assertIsNotNone(new_model.unet)
        finally:
            if os.path.exists(test_path):
                os.remove(test_path)

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        self.model_loader = ModelLoader()
        self.test_prompt = "a restaurant menu showing food items"
        
    def test_singleton_pattern(self):
        another_loader = ModelLoader()
        self.assertEqual(id(self.model_loader), id(another_loader))
        
    def test_generate_image(self):
        image_tensor = self.model_loader.generate_image(self.test_prompt)
        self.assertEqual(image_tensor.shape, (3, 64, 64))

if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('static/generated', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
from app.model_loader import ModelLoader
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.model_loader import ModelLoader
from PIL import Image
import numpy as np


def test_model_loading():
    try:
        model = ModelLoader(model_path="glide_like_model_final.pt")
        assert model is not None
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def test_image_generation():
    try:
        # Load model
        model = ModelLoader(model_path="glide_like_model_final.pt")

        # Generate an image from prompt
        prompt = "a delicious burger on a plate"
        image_tensor = model.generate(prompt)

        # Convert tensor to PIL image
        image_np = (image_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        image = Image.fromarray(image_np)

        # Save image
        save_path = os.path.join(os.path.dirname(__file__), 'test_output.png')
        image.save(save_path)

        print(f"✅ Image generated and saved to: {save_path}")
    except Exception as e:
        print(f"❌ Image generation failed: {e}")

if __name__ == "__main__":
    test_model_loading()
    test_image_generation()
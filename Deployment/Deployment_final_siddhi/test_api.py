# test_api.py
import requests
import json
import base64
from PIL import Image
import io

# API URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())

def test_generate_images():
    """Test image generation"""
    payload = {
        "num_images": 2,
        "seed": 42
    }
    
    response = requests.post(f"{BASE_URL}/generate", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Generated {len(data['images'])} images")
        
        # Save images
        for i, img_data in enumerate(data['images']):
            # Remove data:image/png;base64, prefix
            img_base64 = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_base64)
            
            # Save image
            with open(f"generated_image_{i}.png", "wb") as f:
                f.write(img_bytes)
            print(f"Saved: generated_image_{i}.png")
    else:
        print("Error:", response.json())

def test_generate_single_image():
    """Test single image generation"""
    response = requests.post(f"{BASE_URL}/generate-image?seed=123")
    
    if response.status_code == 200:
        with open("single_generated_image.png", "wb") as f:
            f.write(response.content)
        print("Saved: single_generated_image.png")
    else:
        print("Error:", response.text)

if __name__ == "__main__":
    print("Testing WGAN API...")
    test_health()
    test_generate_images()
    test_generate_single_image()
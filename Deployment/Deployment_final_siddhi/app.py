import torch
import torch.nn as nn
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import base64
import io
from PIL import Image
from model import Generator
import os

# App setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")  # We'll create this directory below

# Configuration
LATENT_DIM = 100
FEATURES = 64
MODEL_PATH = "wgan_final_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model():
    generator = Generator(latent_dim=LATENT_DIM, features=FEATURES, use_attention=False)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

    
    if isinstance(checkpoint, dict):
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        elif 'model_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['model_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    else:
        generator.load_state_dict(checkpoint)
    
    generator.to(DEVICE)
    generator.eval()
    return generator

try:
    model = load_model()
    print(f"Model loaded successfully on {DEVICE}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def tensor_to_image(tensor):
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.squeeze(0)
    image_array = tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)
    return Image.fromarray(image_array)

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

# Route to serve HTML form
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint for image generation
@app.post("/generate")
async def generate(seed: int = Form(default=None), num_images: int = Form(default=1)):
    try:
        if model is None:
            return JSONResponse(content={'success': False, 'error': 'Model not loaded'}, status_code=500)

        if num_images < 1 or num_images > 10:
            return JSONResponse(content={'success': False, 'error': 'Number of images must be between 1 and 10'}, status_code=400)

        images = []
        for i in range(num_images):
            if seed is not None:
                current_seed = int(seed) + i
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)
            else:
                current_seed = torch.randint(0, 10000, (1,)).item()
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(current_seed)

            with torch.no_grad():
                noise = torch.randn(1, LATENT_DIM, 1, 1, device=DEVICE)
                fake_image = model(noise)
                pil_image = tensor_to_image(fake_image)
                img_base64 = image_to_base64(pil_image)
                images.append({"image": img_base64, "seed": current_seed})

        return {"success": True, "images": images}

    except Exception as e:
        return JSONResponse(content={'success': False, 'error': str(e)}, status_code=500)

# Health check endpoint
@app.get("/health")
def health():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(DEVICE)
    }

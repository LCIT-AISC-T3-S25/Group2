# main_final.py - Final version matching your exact checkpoint
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
import os
from typing import Optional
from contextlib import asynccontextmanager
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Request/Response models
class GenerateRequest(BaseModel):
    num_images: int = 1
    seed: Optional[int] = None

class GenerateResponse(BaseModel):
    images: list
    message: str

# Self-attention layer - matching your checkpoint exactly
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_dim, in_dim // 8, 1, bias=True)  # 128 -> 16
        self.key = nn.Conv2d(in_dim, in_dim // 8, 1, bias=True)    # 128 -> 16  
        self.value = nn.Conv2d(in_dim, in_dim, 1, bias=True)       # 128 -> 128
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

# Generator class - exactly matching your checkpoint structure
class Generator(nn.Module):
    """WGAN Generator with attention - matching saved checkpoint exactly"""
    
    def __init__(self, latent_dim=100, features=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        
        # Main architecture (same as pre_attention but separate)
        self.main = nn.Sequential(
            # 0,1,2: 100 -> 1024 x 4 x 4
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),  # 0
            nn.BatchNorm2d(features * 8),                                        # 1
            nn.ReLU(True),                                                       # 2
            
            # 3,4,5: 1024 x 4 x 4 -> 512 x 8 x 8  
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False), # 3
            nn.BatchNorm2d(features * 4),                                         # 4
            nn.ReLU(True),                                                        # 5
            
            # 6,7,8: 512 x 8 x 8 -> 256 x 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False), # 6
            nn.BatchNorm2d(features * 2),                                         # 7
            nn.ReLU(True),                                                        # 8
            
            # 9,10,11: 256 x 16 x 16 -> 128 x 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),     # 9
            nn.BatchNorm2d(features),                                             # 10
            nn.ReLU(True),                                                        # 11
            
            # 12: 128 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),               # 12
            nn.Tanh()                                                            # 13
        )
        
        # Pre-attention architecture (identical to main but separate parameters)
        self.pre_attention = nn.Sequential(
            # 0,1,2: 100 -> 1024 x 4 x 4
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),  # 0
            nn.BatchNorm2d(features * 8),                                        # 1
            nn.ReLU(True),                                                       # 2
            
            # 3,4,5: 1024 x 4 x 4 -> 512 x 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False), # 3
            nn.BatchNorm2d(features * 4),                                         # 4
            nn.ReLU(True),                                                        # 5
            
            # 6,7,8: 512 x 8 x 8 -> 256 x 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False), # 6
            nn.BatchNorm2d(features * 2),                                         # 7
            nn.ReLU(True),                                                        # 8
            
            # 9,10,11: 256 x 16 x 16 -> 128 x 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),     # 9
            nn.BatchNorm2d(features),                                             # 10
            nn.ReLU(True),                                                        # 11
        )
        
        # Attention layer
        self.attention = SelfAttention(features)  # 128 channels
        
        # Post-attention layer
        self.post_attention = nn.Sequential(
            # 0: 128 x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),  # 0
            nn.Tanh()                                               # 1
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # Use attention path (pre_attention -> attention -> post_attention)
        x = self.pre_attention(z)
        x = self.attention(x)
        x = self.post_attention(x)
        return x

# Global variables for model
generator = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    """Load the WGAN model"""
    global generator
    try:
        # Create generator matching checkpoint exactly
        generator = Generator(latent_dim=100, features=128)
        
        # Load the model weights
        model_path = "wgan_final_model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("Loaded checkpoint with weights_only=False")
        
        # Load the generator state dict
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print("Successfully loaded generator_state_dict")
            
        generator.to(device)
        generator.eval()
        print(f"Model loaded successfully on {device}")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def tensor_to_base64(tensor):
    """Convert tensor to base64 encoded image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    transform = transforms.ToPILImage()
    image = transform(tensor.cpu())
    
    # Convert to base64
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return img_base64

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_model()
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(title="WGAN Image Generator", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "WGAN Image Generator API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": str(device)}

@app.post("/generate", response_model=GenerateResponse)
async def generate_images(request: GenerateRequest):
    """Generate images using WGAN"""
    try:
        if generator is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Set seed if provided
        if request.seed is not None:
            torch.manual_seed(request.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(request.seed)
        
        # Generate random noise
        with torch.no_grad():
            noise = torch.randn(request.num_images, 100, 1, 1, device=device)
            fake_images = generator(noise)
        
        # Convert to base64 images
        images_base64 = []
        for i in range(fake_images.size(0)):
            img_base64 = tensor_to_base64(fake_images[i])
            images_base64.append(f"data:image/png;base64,{img_base64}")
        
        return GenerateResponse(
            images=images_base64,
            message=f"Successfully generated {request.num_images} images"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")

@app.post("/generate-image")
async def generate_single_image(seed: Optional[int] = None):
    """Generate a single image and return as file"""
    try:
        if generator is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Set seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate image
        with torch.no_grad():
            noise = torch.randn(1, 100, 1, 1, device=device)
            fake_image = generator(noise)
        
        # Convert to PIL Image
        tensor = (fake_image[0] + 1) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        transform = transforms.ToPILImage()
        image = transform(tensor.cpu())
        
        # Convert to bytes
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(buffer.getvalue()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention layer for the generator"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # Attention
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention to value
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # Residual connection
        out = self.gamma * out + x
        return out

class Generator(nn.Module):
    """Advanced WGAN Generator optimized for 64x64 images"""
    
    def __init__(self, latent_dim=100, features=64, image_size=64, use_attention=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.features = features
        self.use_attention = use_attention
        
        # Simple fixed architecture for 64x64 images
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1 -> features*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            
            # features*8 x 4 x 4 -> features*4 x 8 x 8
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            
            # features*4 x 8 x 8 -> features*2 x 16 x 16
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            
            # features*2 x 16 x 16 -> features x 32 x 32
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # features x 32 x 32 -> 3 x 64 x 64
            nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # Self-attention layer (optional)
        if use_attention:
            # Insert attention before the final layer
            self.attention = SelfAttention(features)
            # Rebuild main without final layer
            self.pre_attention = nn.Sequential(*list(self.main.children())[:-2])  # Remove last 2 layers
            self.post_attention = nn.Sequential(
                nn.ConvTranspose2d(features, 3, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        if self.use_attention:
            x = self.pre_attention(z)
            x = self.attention(x)
            x = self.post_attention(x)
            return x
        else:
            return self.main(z)
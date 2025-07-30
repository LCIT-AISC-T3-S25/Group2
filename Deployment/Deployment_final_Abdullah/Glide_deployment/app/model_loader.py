import torch
from PIL import Image
import numpy as np
from app.utils import Config
import math
import torch.nn as nn
from tqdm import tqdm

class ModelLoader:
    _instance = None
    
    def __new__(cls, model_path="glide_like_model_final.pt"):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self, model_path="glide_like_model_final.pt"):
        if not self.initialized:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.load_model(model_path)
            self.initialized = True
        
    def load_model(self, model_path):
        """Load the trained model from file"""
        print(f"Loading model from {model_path}...")
        
        try:
            # Load the model architecture and weights
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model components with the EXACT parameters as when the model was saved
            # Based on the error analysis and original code structure
            self.text_encoder = ImprovedTextEncoder().to(self.device)
            self.unet = OptimizedUNet(
                model_channels=256,      # This matches the middle_block size of 768 (256*3=768)
                text_emb_dim=512,        # Keep this as 512 based on error logs
                channel_mult=[1, 2, 3]   # Original channel_mult from your code
            ).to(self.device)
            self.diffusion = SimpleDiffusion(device=self.device)
            
            # Load state dicts
            self.text_encoder.load_state_dict(checkpoint['text_encoder_state_dict'],
                    strict=False)
            self.unet.load_state_dict(checkpoint['unet_state_dict'],
                    strict=False)
            
            if 'ema_unet_state_dict' in checkpoint:
                self.ema_model = torch.optim.swa_utils.AveragedModel(self.unet)
                self.ema_model.load_state_dict(checkpoint['ema_unet_state_dict'],
                    strict=False)
            else:
                self.ema_model = None
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, prompt, use_ema=True):
        """Generate image from text prompt"""
        model_to_use = self.ema_model if (use_ema and self.ema_model is not None) else self.unet
        print(1, model_to_use)
        # Generate samples
        samples = self.diffusion.sample(
            model_to_use, 
            self.text_encoder, 
            [prompt],
            n_samples=1,
            img_size=Config.IMAGE_SIZE
        )
        print(2)
        # Denormalize and return first image
        sample = (samples[0] + 1) / 2  # Scale to [0, 1]
        return sample

# Include the model classes from your original code
class ImprovedTextEncoder(nn.Module):
    def __init__(self, vocab_size=Config.VOCAB_SIZE, embed_dim=Config.TEXT_EMB_DIM,
                 max_seq_len=Config.MAX_SEQ_LEN):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Expanded vocabulary for restaurant domain
        self.vocab = {
            '<PAD>': 0, '<UNK>': 1, 'a': 2, 'the': 3, 'of': 4, 'photo': 5,
            'delicious': 6, 'plate': 7, 'food': 8, 'at': 9, 'restaurant': 10,
            'menu': 11, 'showing': 12, 'items': 13, 'interior': 14, 'view': 15,
            'or': 16, 'cafe': 17, 'exterior': 18, 'building': 19, 'refreshing': 20,
            'drink': 21, 'beverage': 22, 'and': 23, 'in': 24, 'with': 25,
            'dining': 26, 'table': 27, 'chair': 28, 'window': 29, 'outdoor': 30,
            'pizza': 31, 'burger': 32, 'salad': 33, 'coffee': 34, 'bar': 35,
            'kitchen': 36, 'chef': 37, 'served': 38, 'fresh': 39, 'tasty': 40,
            'cozy': 41, 'modern': 42, 'traditional': 43, 'elegant': 44, 'casual': 45,
            'meal': 46, 'dish': 47, 'served': 48, 'appetizing': 49, 'beautiful': 50
        }

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, embed_dim))

        # Lighter transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=4, batch_first=True,
                                     dim_feedforward=embed_dim*2),
            num_layers=2
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def tokenize_caption(self, caption):
        """Tokenize a single caption"""
        words = caption.lower().replace(',', '').replace('.', '').split()
        tokens = []
        for word in words[:self.max_seq_len-1]:
            token = self.vocab.get(word, self.vocab['<UNK>'])
            tokens.append(token)

        # Pad to max_seq_len
        while len(tokens) < self.max_seq_len:
            tokens.append(self.vocab['<PAD>'])

        return tokens[:self.max_seq_len]

    def forward(self, captions):
        """Forward pass through text encoder"""
        if isinstance(captions, str):
            captions = [captions]

        tokenized = []
        for caption in captions:
            tokens = self.tokenize_caption(caption)
            tokenized.append(tokens)

        tokens_tensor = torch.tensor(tokenized, device=next(self.parameters()).device)
        embeddings = self.embedding(tokens_tensor)
        embeddings += self.positional_encoding[:embeddings.size(1)].unsqueeze(0)

        text_features = self.transformer(embeddings)
        text_features = self.layer_norm(text_features.mean(dim=1))

        return text_features
    
class EfficientResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, text_emb_dim, dropout=0.1):
        super().__init__()
        groups = min(8, out_channels // 4) if out_channels >= 16 else 1

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.text_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(text_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Dropout2d(dropout)
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.Dropout2d(dropout)
        )

        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, text_emb):
        h = self.block1(x)

        time_emb = self.time_mlp(time_emb)
        h += time_emb[:, :, None, None]

        text_emb = self.text_mlp(text_emb)
        h += text_emb[:, :, None, None]

        h = self.block2(h)
        return h + self.residual_conv(x)


class OptimizedUNet(nn.Module):
    def __init__(self, in_channels=3, model_channels=256, out_channels=3,
                 num_res_blocks=2, channel_mult=[1, 2, 3], num_heads=4,
                 text_emb_dim=512):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_heads = num_heads

        time_embed_dim = model_channels * 3  # Back to original multiplier
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Build encoder
        self.encoder_blocks = nn.ModuleList()
        ch = model_channels
        input_block_chans = [model_channels]

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [EfficientResnetBlock(ch, mult * model_channels, time_embed_dim, text_emb_dim)]
                ch = mult * model_channels
                self.encoder_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)

        # Middle block
        self.middle_block = EfficientResnetBlock(ch, ch, time_embed_dim, text_emb_dim)

        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [EfficientResnetBlock(ch + ich, model_channels * mult, time_embed_dim, text_emb_dim)]
                ch = model_channels * mult
                self.decoder_blocks.append(nn.Sequential(*layers))

            if level != 0:
                self.decoder_blocks.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )

        # Output layers
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, model_channels // 4), model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1)
        )

    def get_time_embedding(self, timesteps):
        half_dim = self.model_channels // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x, timesteps, text_emb):
        t_emb = self.get_time_embedding(timesteps)
        t_emb = self.time_embed(t_emb)

        h = self.input_conv(x)
        hs = [h]

        # Encoder
        for module in self.encoder_blocks:
            if isinstance(module, nn.Sequential):
                h = module[0](h, t_emb, text_emb)
            else:
                h = module(h)
            hs.append(h)

        # Middle
        h = self.middle_block(h, t_emb, text_emb)

        # Decoder
        for module in self.decoder_blocks:
            if isinstance(module, nn.Sequential):
                h = torch.cat([h, hs.pop()], dim=1)
                h = module[0](h, t_emb, text_emb)
            else:
                h = module(h)

        return self.output_conv(h)

class SimpleDiffusion:
    def __init__(self, noise_steps=Config.NOISE_STEPS, beta_start=Config.BETA_START,
                 beta_end=Config.BETA_END, device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Pre-compute diffusion parameters
        self.beta = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        noise = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, text_encoder, captions, n_samples=None, img_size=Config.IMAGE_SIZE):
        """Generate samples using the trained model"""
        if n_samples is None:
            n_samples = len(captions) if isinstance(captions, list) else 1

        if isinstance(captions, str):
            captions = [captions] * n_samples
        print(5)
        model.eval()
        print(3)
        with torch.no_grad():
            text_emb = text_encoder(captions)
            x = torch.randn((n_samples, 3, img_size, img_size)).to(self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Generating"):
                t = torch.full((n_samples,), i, dtype=torch.long, device=self.device)
                predicted_noise = model(x, t, text_emb)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]

                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        print(4)
        model.train()
        return x
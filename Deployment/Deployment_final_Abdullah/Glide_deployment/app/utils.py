import torch

"""class Config:
    # Model parameters
    BATCH_SIZE = 2
    IMAGE_SIZE = 64
    
    # Model architecture
    MODEL_CHANNELS = 96
    TEXT_EMB_DIM = 256
    MAX_SEQ_LEN = 50
    VOCAB_SIZE = 2000
    
    # Diffusion parameters
    NOISE_STEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02"""

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Config:
    # Update text embedding dimension to match saved model
    TEXT_EMB_DIM = 512  # Changed from 256 to 512
    
    # Model parameters
    BATCH_SIZE = 2
    IMAGE_SIZE = 64
    
    # Model architecture
    MODEL_CHANNELS = 96
    MAX_SEQ_LEN = 50
    VOCAB_SIZE = 2000
    
    # Diffusion parameters
    NOISE_STEPS = 50 #1000
    BETA_START = 1e-4
    BETA_END = 0.02
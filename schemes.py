from pydantic import validator
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class GeneratorInput():
    latent_dim: int
    dropout: float
    output_channels: int
    
    @validator('dropout')
    def is_probability(cls, value):
        if value > 1.0 or value < 0.0:
            raise ValueError('Dropout must be a float in [0, 1].')
        return value 

@dataclass
class RepresentationBaseInput():
    representation_dim: int
    dropout: float
    output_channels: int 
    contrastive_dim: Optional[int] = None
    
    @validator('dropout')
    def is_probability(cls, value):
        if value > 1.0 or value < 0.0:
            raise ValueError('Dropout must be a float in [0, 1].')
        return value 
    
@dataclass
class GANBaseInput():
    generator_dropout: float = 0.0
    generator_latent_dim: int = 100
    representation_dim: int = 1024
    discriminator_dropout: float = 0.0
    output_channels: int = 1
    contrastive_dim: int = 512
    
    @validator('generator_dropout', 'discriminator_dropout')
    def is_probability(cls, value):
        if value > 1.0 or value < 0.0:
            raise ValueError('Dropout must be a float in [0, 1].')
        return value 

    
@dataclass
class GANTrainInput():
    epochs: int = 100
    snapshot: int = 20
    generator_lr: float = 2e-4
    generator_beta1: float = 0.5
    generator_beta2: float = 0.999
    discriminator_lr: float = 2e-4
    discriminator_beta1: float = 0.5
    discriminator_beta2: float = 0.999
    representation_lr: float = 2e-4
    representation_beta1: float = 0.5
    representation_beta2: float = 0.999
    contrastive_temperature: float = 0.5
    random_seed: int = 1602
    
@dataclass
class DataInput():
    data_dir: str = '/home/petra/Documents/PhD/Repos/datasets'
    batch_size: int = 128
    num_workers: int = 0

    
    
    
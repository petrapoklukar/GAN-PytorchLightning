import torch.nn as nn
import torch
from schemes import GeneratorInput, RepresentationBaseInput
import torch.nn.functional as F

# --------------------------------------------------------------- #
# --- To keep track of the dimensions of convolutional layers --- #
# --------------------------------------------------------------- #
class PrintShape(nn.Module):
    def __init__(self, message:str):
        super(PrintShape, self).__init__()
        self.message = message
        
    def forward(self, feat:torch.Tensor):
        print(self.message, feat.shape)
        return feat 
    
    
# ----------------------------------------------- #
# --- Convolutional Generator & Distriminator --- #
# ----------------------------------------------- #
class ConvolutionalGenerator(nn.Module):
    def __init__(self, config:GeneratorInput):
        super(ConvolutionalGenerator, self).__init__()
        self.params = config

        self.model = nn.Sequential(
            nn.Linear(self.params.latent_dim, 4096), 
            nn.Unflatten(1, (256, 4, 4)), 
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.Dropout(p=self.params.dropout),
            
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1), # 128, 7, 7
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.Dropout(p=self.params.dropout),
            
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=0), # 64, 16, 16
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.Dropout(p=self.params.dropout),
            
            nn.ConvTranspose2d(64, self.params.output_channels, 4, stride=2, padding=1), # 3, 32, 32
            nn.Tanh())
            
    def forward(self, x:torch.Tensor):
        return self.model(x)


class ConvolutionalBase(nn.Module):
    def __init__(self, config:RepresentationBaseInput):
        super(ConvolutionalBase, self).__init__()
        self.params = config

        self.backbone = nn.Sequential(
            nn.Conv2d(self.params.output_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.params.dropout),
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.params.dropout),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten(start_dim=1))

    def forward(self, x):
        return self.backbone(x)


class Discriminator(ConvolutionalBase):
    def __init__(self, config:RepresentationBaseInput):
        super().__init__(config)
        
        self.model = nn.Sequential(
            nn.Linear(4*4*256, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 1), 
            nn.Sigmoid())
    
    def forward(self, x:torch.Tensor):
        mid_representation = self.backbone(x)
        prediction = self.model(mid_representation).squeeze()
        return prediction
    
class ContrastiveLearner(ConvolutionalBase):
    def __init__(self, config:RepresentationBaseInput):
        super().__init__(config)
        self.contrastive_dim = config.contrastive_dim
    
        self.representation_layer = nn.Linear(4*4*256, self.params.representation_dim)
        self.model = nn.Sequential(
            nn.Linear(self.params.representation_dim, int(self.params.representation_dim/2)), 
            nn.ReLU(),
            nn.Linear(int(self.params.representation_dim/2), self.params.contrastive_dim), 
        )
    
    def get_representations(self, x:torch.Tensor):
        base_representation = self.backbone(x)
        representation = self.representation_layer(base_representation)
        return representation
    
    def forward(self, x:torch.Tensor):
        base_representation = self.backbone(x)
        representation = self.representation_layer(base_representation)
        contrastive_output = self.model(representation)
        return F.normalize(representation, dim=-1), F.normalize(contrastive_output, dim=-1)

if __name__ == '__main__':
    def testGenerator():
        gen_input = GeneratorInput(latent_dim=50, dropout=0.)
        gen = ConvolutionalGenerator(gen_input)
        input = torch.randn(2, 50)
        output = gen(input)
        print(output.shape)
    
    def testDiscriminatorAndContrastiveLearner():
        d_input = RepresentationBaseInput(dropout=0., representation_dim=1024, contrastive_dim=512)
        discriminator = Discriminator(d_input)
        contrastive_learner = ContrastiveLearner(d_input)
        input = torch.randn(2, 3, 32, 32)
        d_output = discriminator(input)
        c_output = contrastive_learner(input)
        print(d_output.shape)
        print(c_output[0].shape, c_output[1].shape)
        print('Test')
    
    testDiscriminatorAndContrastiveLearner()
    
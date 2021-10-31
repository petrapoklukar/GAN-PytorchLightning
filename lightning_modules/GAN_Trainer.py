import torch
from schemes import GANBaseInput, GANTrainInput, GeneratorInput, RepresentationBaseInput
from architectures.neural_networks import ConvolutionalGenerator, Discriminator
from pytorch_lightning import LightningModule
from collections import OrderedDict
import torchvision
import torch.optim as optim

class GAN(LightningModule):
    def __init__(self, 
                 model_config:GANBaseInput, 
                 train_config:GANTrainInput,
                 experiment_id:str, 
                 ):
        super(GAN, self).__init__()
        self.save_hyperparameters()
        self.experiment_id = experiment_id
        self.model_params = model_config
        self.generator = self.init_generator() 
        self.discriminator = self.init_discriminator()
        self.train_params = train_config
        self.fixed_validation_z = self.sample_latent_noise(8)
            
    def init_generator(self) -> ConvolutionalGenerator:
        generator_config = GeneratorInput(latent_dim=self.model_params.generator_latent_dim, 
                                          dropout=self.model_params.generator_dropout, 
                                          output_channels=self.model_params.output_channels
                                          )
        return ConvolutionalGenerator(generator_config)
    
    def init_discriminator(self):
        discriminator_config = RepresentationBaseInput(representation_dim=self.model_params.representation_dim, 
                                                        dropout=self.model_params.discriminator_dropout,
                                                        output_channels=self.model_params.output_channels)
        return Discriminator(discriminator_config)  
      
    # ---
    # --- Utils functions --- #
    def dinput_noise(self, tensor):
        """Adds small Gaussian noise to the tensor."""
        dinput_noise = torch.zeros(tensor.size()).type_as(tensor)
        return tensor + dinput_noise
    
    def sample_latent_noise(self, batch_size):
        """Generates gaussian noise."""
        z_noise = torch.empty((batch_size, self.generator.params.latent_dim), 
                              requires_grad=False).normal_() # b, z_dim
        return z_noise
    
    # ---
    # --- Pytorch ligthning --- #
    def forward(self, n_samples:int):
        z = self.sample_latent_noise(n_samples)
        return self.generator(z)
        
    def configure_optimizers(self):
        optimiser_D = optim.Adam(
            self.discriminator.parameters(), 
            lr=self.train_params.discriminator_lr_schedule[0][-1], 
            betas=(self.train_params.discriminator_beta1, self.train_params.discriminator_beta2))
        optimiser_G = optim.Adam(
            self.generator.parameters(),
            lr=self.train_params.generator_lr_schedule[0][-1], 
            betas=(self.train_params.generator_beta1, self.train_params.generator_beta2))
        return [optimiser_D, optimiser_G], []    
    
    def loss(self, y_pred, y_true):
        return torch.nn.BCELoss()(y_pred, y_true)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # Ground truths
        real_imgs, _ = batch
        batch_size = real_imgs.shape[0]

        real_labels = torch.ones(batch_size).type_as(real_imgs)
        fake_labels = torch.zeros(batch_size).type_as(real_imgs)
        z_noise = self.sample_latent_noise(batch_size).type_as(real_imgs)
        
        # --- Train the Discriminator --- #
        if optimizer_idx == 0:

            # Loss for real images
            real_imgs = self.dinput_noise(real_imgs)
            real_pred = self.discriminator(real_imgs)                
            assert torch.sum(torch.isnan(real_pred)) == 0, real_pred
            assert(real_pred >= 0.).all(), real_pred
            assert(real_pred <= 1.).all(), real_pred
            d_real_loss = self.loss(real_pred, real_labels)
            D_img = real_pred.mean()
            
            # Loss for fake images
            fake_imgs = self.generator(z_noise)
            assert torch.sum(torch.isnan(fake_imgs)) == 0, fake_imgs
            fake_imgs_input = self.dinput_noise(fake_imgs.detach())
            fake_pred = self.discriminator(fake_imgs_input)
            assert(fake_pred >= 0.).all(), fake_pred
            assert(fake_pred <= 1.).all(), fake_pred
            d_fake_loss = self.loss(fake_pred, fake_labels)
            D_G_z1 = fake_pred.mean()
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss, 'D_img': D_img, 'D_G_z1': D_G_z1}
            output = OrderedDict({'loss': d_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output

        # --- Train the Generator --- #
        if optimizer_idx == 1:        
            fake_imgs = self.generator(z_noise)
            fake_imgs_input = self.dinput_noise(fake_imgs)
            fake_pred = self.discriminator(fake_imgs_input)
            g_loss = self.loss(fake_pred, real_labels)
            D_G_z2 = fake_pred.mean()
            
            # log sampled images
            sample_imgs = fake_imgs[:6]
            grid = torchvision.utils.make_grid(sample_imgs)
            self.logger.experiment.add_image('generated_images_after_update', grid, 0)

            tqdm_dict = {'g_loss': g_loss, 'D_G_z2': D_G_z2}
            output = OrderedDict({'loss': g_loss, 'progress_bar': tqdm_dict, 'log': tqdm_dict})
            return output
    
    def training_epoch_end(self, outputs:list):
        d_outputs, g_outputs = outputs    
        avg_losses = {}
        for log_key in d_outputs[0]['log']:
            avg_batch_log = torch.stack([batch_res['log'][log_key] for batch_res in d_outputs]).mean()
            self.logger.experiment.add_scalar(F"{log_key}/Train", avg_batch_log, self.current_epoch)
            if log_key == 'd_loss':
                avg_losses[log_key] = avg_batch_log
        for log_key in g_outputs[0]['log']:
            avg_batch_log = torch.stack([batch_res['log'][log_key] for batch_res in g_outputs]).mean()
            self.logger.experiment.add_scalar(F"{log_key}/Train", avg_batch_log, self.current_epoch)
            if log_key == 'g_loss':
                avg_losses[log_key] = avg_batch_log
            
        self.logger.experiment.add_scalars("Losses/Train", avg_losses, self.current_epoch)
        self.log_generated_images('generated_images')
        

    # def on_epoch_end(self) -> None:
    #     # log sampled images
    #     self.log_generated_images('generated_images')
    
    def log_generated_images(self, name:str):
        z = self.fixed_validation_z.type_as(self.generator.model[0].weight)
        sample_imgs = self.generator(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(name, grid, self.current_epoch)
        


import torch.nn as nn
import torch.optim as optim
import torch
import architectures.GAN_models as models
import numpy as np
import os
import matplotlib as mpl
if not "DISPLAY" in os.environ:
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

class GAN(nn.Module):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Training parameters
        train_config = config['train_config']
        self.batch_size = train_config['batch_size']
        self.epochs = train_config['epochs']
        self.current_epoch = None
        self.start_epoch = None
        self.snapshot = train_config['snapshot']
        self.console_print = train_config['console_print']
        self.z_dim = config['data_config']['usual_noise_dim']
        self.input_noise = train_config['input_noise']
        self.input_variance_increase = train_config['input_variance_increase']
        self.grad_clip = train_config['grad_clip']
        self.dis_grad_clip = train_config['dis_grad_clip']
        self.gen_grad_clip = train_config['gen_grad_clip']
        
        # Fixes noise to monitor the generator's progress
        self.fixed_z_noise = self.sample_latent_noise(100) 
        
        self.init_gen_lr_schedule = train_config['gen_lr_schedule']
        self.gen_lr_schedule = train_config['gen_lr_schedule']
        self.dis_lr_schedule = train_config['dis_lr_schedule']
        self.init_dis_lr_schedule = train_config['dis_lr_schedule']

        self.exp_dir = train_config['exp_dir']
        self.save_path = train_config['exp_dir'] + '/' + train_config['filename']
        self.model_path = self.save_path + '_model.pt'
        self.create_dirs()
        
        # Fix random seed
        torch.manual_seed(train_config['random_seed'])
        np.random.seed(train_config['random_seed'])
        self.monitor_generator = 5
        self.discriminator_update_step = 3

    # ---------------------- #
    # --- Init functions --- #
    # ---------------------- #
    def create_dirs(self):
        """Creates folders for saving training logs."""
        self.test_dir = '{0}/Testing/'.format(self.exp_dir)
        self.train_dir = '{0}/Training/'.format(self.exp_dir)
        if (not os.path.isdir(self.test_dir)):
            os.makedirs(self.test_dir)
        if (not os.path.isdir(self.train_dir)):
            os.makedirs(self.train_dir)
        
    def init_generator(self):
        """Initialises the generator."""
        try:
            print(self.config['generator_config'])
            class_ = getattr(GAN_models, self.config['generator_config']['class_name'])
            self.generator = class_(self.config['generator_config']).to(self.device)
            print(' *- Initialised generator: ', self.config['generator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Generator class {0} not recognized'.format(
                            self.config['generator_config']['class_name']))

    def init_discriminator(self):
        """Initialises the discriminator."""
        try:
            print(self.config['discriminator_config'])
            class_ = getattr(GAN_models, self.config['discriminator_config']['class_name'])
            self.discriminator = class_(self.config['discriminator_config']).to(self.device)
            print(' *- Initialised discriminator: ', self.config['discriminator_config']['class_name'])
        except: 
            raise NotImplementedError(
                    'Discriminator class {0} not recognized'.format(
                            self.config['discriminator_config']['class_name']))
            
    def init_weights(self, m):
        """Custom weight init"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)

    def init_optimisers(self):
        """Initialises the optimisers."""
        optim_config = self.config['train_config']
        optim_type = optim_config['optim_type']
        if optim_type == 'Adam':
            # Generator optimiser
            self.optimiser_G = optim.Adam(
                    self.generator.parameters(),
                    lr=self.gen_lr, 
                    betas=(optim_config['gen_b1'], optim_config['gen_b2']))
            print(' *- Initialised generator optimiser: Adam')
            
            # Discriminator optimiser
            self.optimiser_D = optim.Adam(
                    self.discriminator.parameters(), 
                    lr=self.dis_lr, 
                    betas=(optim_config['dis_b1'], optim_config['dis_b2']))
            print(' *- Initialised discriminator optimiser: Adam')
            
        else: 
            raise NotImplementedError(
                    'Optimiser {0} not recognized'.format(optim_type))
    
    def init_losses(self):
        """Initialises the loss."""
        self.gan_loss = torch.nn.BCELoss().to(self.device)    

    # ---------------------------- #
    # --- Monitoring functions --- #
    # ---------------------------- #    
    def get_gradients(self, model):
        total_norm = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_norm = param.grad.data.norm(2).item()
                total_norm.append(np.around(param_norm, decimals=3))
        return total_norm
#                print('===\ngradient:{}\n {}\n {}'.format(
#                        name, torch.mean(param.grad), param_norm))        
#        total_norm = total_norm ** (1. / 2)
#        print('===\ngradients:{}'.format(total_norm))        
        
        
    def count_parameters(self, model):
        """Counts the total number of trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def print_model_params(self):
        """Prints specifications of the trainable parameters."""
        def print_trainable_param(model, model_name, n_params):
            print(' *- {1} parameters: {0}'.format(n_params, model_name))
            for name, param in model.named_parameters():
                if param.requires_grad:
                    spacing = 1
                    print('{0:>2}{1}\n\t of dimension {2}'.format('', name, spacing),  
                          list(param.shape))

        num_gparameters = self.count_parameters(self.generator) 
        print_trainable_param(self.generator, 'Generator', num_gparameters)
        self.config['generator_config']['n_model_params'] = num_gparameters
        
        num_dparameters = self.count_parameters(self.discriminator) 
        print_trainable_param(self.discriminator, 'Discriminator', num_dparameters)
        self.config['discriminator_config']['n_model_params'] = num_dparameters

    def plot_snapshot_loss(self):
        """
        Plots the discriminator and generator losses at each snapshot interval.
        """
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['d_loss', 'g_loss']
        n_subplots = len(plt_labels)
        for i in range(n_subplots):
            plt.subplot(n_subplots,1,i+1)
            plt.plot(np.arange(self.snapshot)+(self.current_epoch//self.snapshot)*self.snapshot,
                     plt_data[self.current_epoch-self.snapshot+1:self.current_epoch+1, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.train_dir + 'SnapshotLosses_{0}'.format(self.current_epoch))
        plt.clf()
        plt.close()
    
    def plot_model_loss(self):
        """Plots epochs vs discriminator and generator losses."""
        plt_data = np.stack(self.epoch_losses)
        plt_labels = ['d_loss', 'g_loss']
        n_subplots = len(plt_labels)
        for i in range(n_subplots):
            plt.subplot(n_subplots,1,i+1)
            plt.plot(np.arange(self.current_epoch+1), plt_data[:, i], 
                     label=plt_labels[i])
            plt.ylabel(plt_labels[i])
            plt.xlabel('# epochs')
            plt.legend()
        plt.savefig(self.save_path + '_Losses')
        plt.clf()
        plt.close()
        
        fig, ax = plt.subplots()
        ax.plot(plt_data[:, 0], 'go-', linewidth=3, label='D loss')
        ax.plot(plt_data[:, 1], 'bo--', linewidth=2, label='G loss')
        ax.plot()
        ax.legend()
        ax.set_xlim(0, self.epochs)
        ax.set(xlabel='# epochs', ylabel='loss', title='Discriminator vs Generator loss')
        plt.savefig(self.save_path + '_DvsGLoss')
        plt.close()
        
        fig2, ax2 = plt.subplots()
        ax2.plot(plt_data[:, 0], 'go-', linewidth=3, label='D loss')
        ax2.plot()
        ax2.set_xlim(0, self.epochs)
        ax2.set(xlabel='# epochs', ylabel='loss', title='Discriminator loss')
        plt.savefig(self.save_path + '_DLoss')
        plt.close()
        
        fig3, ax3 = plt.subplots()
        ax3.plot(plt_data[:, 1], 'go-', linewidth=3, label='G loss')
        ax3.plot()
        ax3.set_xlim(0, self.epochs)
        ax3.set(xlabel='# epochs', ylabel='loss', title='Generator loss')
        plt.savefig(self.save_path + '_GLoss')
        plt.close()
        
    def plot_gradients(self):
        """Plots epochs vs average discriminator and generator gradients."""
        plt_ddata = np.stack(self.dgrad_norms)
        n_subplots = len(plt_ddata[0])
        for i in range(n_subplots):
            plt.plot(plt_ddata[:, i], label=str(i))
            plt.ylabel('gradient_norm')
            plt.xlabel('# epochs')
            plt.legend()
        plt.title('D Gradient norms - average per epoch')
        plt.savefig(self.save_path + '_Dgradients')
        plt.clf()
        plt.close()
            
        plt_gdata = np.stack(self.ggrad_norms)
        n_subplots = len(plt_gdata[0])
        for i in range(n_subplots):
            plt.plot(plt_gdata[:, i], label=str(i))
            plt.ylabel('gradient_norm')
            plt.xlabel('# epochs')
            plt.legend()
        plt.title('G Gradient norms - average per epoch')
        plt.savefig(self.save_path + '_Ggradients')
        plt.clf()
        plt.close()
        
        plt_gdata_total = np.array(self.ggrad_total_norm)
        plt_ddata_total = np.array(self.dgrad_total_norm)
        plt.plot(plt_gdata_total, label='Total G norms')
        plt.plot(plt_ddata_total, label='Total D norms')
        plt.ylabel('total_gradient_norm')
        plt.xlabel('# epochs')
        plt.legend()
        plt.title('Total G and D Gradient norms')
        plt.savefig(self.save_path + '_totalGaradients')
        plt.clf()
        plt.close()
    
    def sq_else_perm(self, img):
        """"""
        grayscale = True if img.shape[1] == 1 else False
        return img.squeeze() if grayscale else img.permute(1,2,0)
    
    def plot_image_grid(self, images, filename, directory, n=25):
        """Plots a grid of (generated) images."""
        n_subplots = np.sqrt(n).astype(int)
        plot_range = n_subplots ** 2
        images = self.sq_else_perm(images)
        for i in range(plot_range):
            plt.subplot(n_subplots, n_subplots, 1 + i)
            plt.axis('off')
            plt.imshow(images[i].detach().cpu().numpy())
        plt.savefig(directory + filename)
        plt.clf()
        plt.close()
        
    def format_loss(self, losses_list):
        """Rounds the loss and returns an np array"""
        reformatted = list(map(lambda x: round(x.item(), 2), losses_list))
        reformatted.append(self.current_epoch)
        return np.array(reformatted)
    
    # -------------------------- #
    # --- Training functions --- #
    # -------------------------- #        
    def sample_latent_noise(self, batch_size):
        """Generates gaussian noise."""
        z_noise = torch.empty((batch_size, self.z_dim, 1, 1), requires_grad=False, 
                              device=self.device).normal_() # b, z_dim, 1, 1
        return z_noise
    
    def dinput_noise(self, tensor):
        """Adds small Gaussian noise to the tensor."""
        if self.input_noise:                        
            dinput_std = max(0.75*(10. - self.current_epoch//self.input_variance_increase) / (10), 0.05)
            dinput_noise = torch.empty(tensor.size(), device=self.device).normal_(mean=0, std=dinput_std)
        else:
            dinput_noise = torch.zeros(tensor.size(), device=self.device)
            
        return tensor + dinput_noise
    
    def train(self, train_dataloader, chpnt_path=''):
        """Trains an InfoGAN."""
        
        print(('\nPrinting model specifications...\n' + 
               ' *- Path to the model: {0}\n' + 
               ' *- Number of epochs: {1}\n' + 
               ' *- Batch size: {2}\n' 
               ).format(self.model_path, self.epochs, self.batch_size))
        
        if chpnt_path: 
            # Pick up the last epochs specs
            self.load_checkpoint(chpnt_path)
    
        else:
            # Initialise the models, weights and optimisers
            self.init_generator()
            self.init_discriminator()
            self.generator.apply(self.init_weights)
            self.discriminator.apply(self.init_weights)
            self.start_gen_epoch, self.gen_lr = self.gen_lr_schedule.pop(0)
            self.start_dis_epoch, self.dis_lr = self.dis_lr_schedule.pop(0)
            assert(self.start_gen_epoch == self.start_dis_epoch)
            self.start_epoch = self.start_dis_epoch
            try:
                self.gen_lr_update_epoch, self.new_gen_lr = self.gen_lr_schedule.pop(0)
                self.dis_lr_update_epoch, self.new_dis_lr = self.dis_lr_schedule.pop(0)
            except:
                self.gen_lr_update_epoch, self.new_gen_lr = self.start_epoch - 1, self.gen_lr
                self.dis_lr_update_epoch, self.new_dis_lr = self.start_epoch - 1, self.dis_lr
            self.init_optimisers()
            self.epoch_losses = []
            self.dgrad_norms = []
            self.ggrad_norms = []

            self.dgrad_total_norm = []
            self.ggrad_total_norm = []
            print((' *- Generator' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.gen_lr, self.gen_lr_update_epoch, self.new_gen_lr, 
                   self.gen_lr_schedule))            
            print((' *- Discriminator' + 
                   '    *- Learning rate: {0}\n' + 
                   '    *- Next lr update at {1} to the value {2}\n' + 
                   '    *- Remaining lr schedule: {3}'
                   ).format(self.dis_lr, self.dis_lr_update_epoch, self.new_dis_lr, 
                   self.dis_lr_schedule))            

        self.print_model_params()
        self.init_losses()
        print('\nStarting to train the model...\n' )        
        for self.current_epoch in range(self.start_epoch, self.epochs):
            self.generator.train()
            self.discriminator.train()
            assert(self.generator.training)
            
            epoch_loss = np.zeros(3)
            epochs_d_norms = []
            epochs_g_norms = []
            for i, x in enumerate(train_dataloader):
                
                # Ground truths
                batch_size = x.shape[0]

                real_x = x.to(self.device)        
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)
                
                
                # ------------------------------- #
                # --- Train the Discriminator --- #
                # ------------------------------- #
                self.optimiser_D.zero_grad()
        
                # Loss for real images
                real_x = self.dinput_noise(real_x)
                real_pred = self.discriminator(real_x)                
                assert torch.sum(torch.isnan(real_pred)) == 0, real_pred
                assert(real_pred >= 0.).all(), real_pred
                assert(real_pred <= 1.).all(), real_pred
                d_real_loss = self.gan_loss(real_pred, real_labels)
                D_x = real_pred.mean().item()
                
                # Loss for fake images
                z_noise = self.sample_latent_noise(batch_size)
                fake_x = self.generator(z_noise)
                assert torch.sum(torch.isnan(fake_x)) == 0, fake_x
                fake_x_input = self.dinput_noise(fake_x.detach())
                fake_pred = self.discriminator(fake_x_input)
                assert(fake_pred >= 0.).all(), fake_pred
                assert(fake_pred <= 1.).all(), fake_pred
                d_fake_loss = self.gan_loss(fake_pred, fake_labels)
                D_G_z1 = fake_pred.mean().item()
                
                # Total discriminator loss
                d_loss = d_real_loss + d_fake_loss       
                d_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                            self.discriminator.parameters(), self.dis_grad_clip) 
#                torch.nn.utils.clip_grad_value_(
#                        self.discriminator.parameters(), 5)
                self.optimiser_D.step()
                
                # Track discriminator's gradients
                b_d_norms = self.get_gradients(self.discriminator)
                epochs_d_norms.append(b_d_norms)
                b_d_norm_total = np.around(np.linalg.norm(np.array(b_d_norms)), decimals=3)
            

                # --------------------------- #
                # --- Train the Generator --- #
                # --------------------------- #
                self.optimiser_G.zero_grad()
                
                fake_x_input = self.dinput_noise(fake_x)
                fake_pred = self.discriminator(fake_x_input)
                g_loss = self.gan_loss(fake_pred, real_labels)
                g_loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                            self.generator.parameters(), self.gen_grad_clip) 
                self.optimiser_G.step()
                
                # Track generator's gradients
                b_g_norms = self.get_gradients(self.generator)
                epochs_g_norms.append(b_g_norms)
                b_g_norm_total = np.around(np.linalg.norm(np.array(b_g_norms)), decimals=3)
                
                D_G_z2 = fake_pred.mean().item()

                epoch_loss += self.format_loss([d_loss, g_loss])
        
            # ------------------------ #
            # --- Log the training --- #
            # ------------------------ #      
            epoch_loss /= len(train_dataloader)
            print(
                "[Epoch %d/%d]\n\t[D loss: %f] [G loss: %f]"
                % (self.current_epoch, self.epochs, epoch_loss[0], epoch_loss[1]))
            print(
                "\t[D_x %f] [D_G_z1: %f] [D_G_z2: %f]"
                % (D_x, D_G_z1, D_G_z2))
            print("\tD Norm mean: ", np.mean(epochs_d_norms, axis=0))
            print("\tG Norm mean: ", np.mean(epochs_g_norms, axis=0))
            
                    
            # TODO: add logger here
            self.epoch_losses.append(epoch_loss)
            self.dgrad_norms.append(np.mean(epochs_d_norms, axis=0))
            self.ggrad_norms.append(np.mean(epochs_g_norms, axis=0))
            self.dgrad_total_norm.append(b_d_norm_total)
            self.ggrad_total_norm.append(b_g_norm_total)
            self.plot_model_loss() 
            self.plot_gradients()
            self.save_checkpoint(epoch_loss)
            
            if (self.current_epoch + 1) % self.snapshot == 0:
                # Save the checkpoint & logs, plot snapshot losses
                self.save_checkpoint(epoch_loss, keep=False)
                self.save_logs()

                # Plot snapshot losses
                self.plot_snapshot_loss()
                
            if (self.current_epoch + 1) % self.monitor_generator == 0:
                gen_x = self.generator(self.fixed_z_noise) 
                gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
                   
                filename = 'genImages' + str(self.current_epoch)
                self.plot_image_grid(gen_x_plotrescale, filename, self.train_dir, n=100)
                
        
        # ---------------------- #
        # --- Save the model --- #
        # ---------------------- # 
        print('Training completed.')
        self.plot_model_loss()
        self.generator.eval()
        self.discriminator.eval()
        torch.save({
                'generator': self.generator.state_dict(), 
                'discriminator': self.discriminator.state_dict()}, 
                self.model_path)       
        self.save_logs()
        
    # ---------------------------------- #
    # --- Saving & Loading functions --- #
    # ---------------------------------- #
    def save_logs(self, ):
        """Saves a txt file with logs"""
        log_filename = self.save_path + '_logs.txt'
        epoch_losses = np.stack(self.epoch_losses)
        
        with open(log_filename, 'w') as f:
            f.write('Model {0}\n\n'.format(self.config['train_config']['filename']))
            f.write( str(self.config) )
            f.writelines(['\n\n', 
                    '*- Model path: {0}\n'.format(self.model_path),
                    '*- Generator learning rate schedule: {0}\n'.format(self.init_gen_lr_schedule),
                    '*- Discriminator learning rate schedule: {0}\n'.format(self.init_dis_lr_schedule),
                    '*- Training epoch losses: (d_loss, g_loss)\n'
                    ])
            f.writelines(list(map(
                    lambda t: '{0:>3} Epoch {3}: ({1:.2f}, {2:.2f})\n'.format(
                            '', t[0], t[1], t[2]), 
                    epoch_losses)))
        print(' *- Model saved.\n')
    
    
    def save_checkpoint(self, epoch_loss, keep=False):
        """Saves a checkpoint during the training."""
        if keep:
            path = self.save_path + '_checkpoint{0}.pth'.format(self.current_epoch)
            checkpoint_type = 'epoch'
        else:
            path = self.save_path + '_lastCheckpoint.pth'
            checkpoint_type = 'last'
        training_dict = {
                'last_epoch': self.current_epoch,
                
                'generator_state_dict': self.generator.state_dict(),
                'discriminator_state_dict': self.discriminator.state_dict(),
            
                'optimiser_D_state_dict': self.optimiser_D.state_dict(),
                'optimiser_G_state_dict': self.optimiser_G.state_dict(),
                
                'last_epoch_loss': epoch_loss,
                'epoch_losses': self.epoch_losses,

                'snapshot': self.snapshot,
                'console_print': self.console_print,
                
                'current_gen_lr': self.gen_lr,
                'gen_lr_update_epoch': self.gen_lr_update_epoch, 
                'new_gen_lr': self.new_gen_lr, 
                'gen_lr_schedule': self.gen_lr_schedule,
                
                'current_dis_lr': self.dis_lr,
                'dis_lr_update_epoch': self.dis_lr_update_epoch, 
                'new_dis_lr': self.new_dis_lr, 
                'dis_lr_schedule': self.dis_lr_schedule
                }
        torch.save({**training_dict, **self.config}, path)
        print(' *- Saved {1} checkpoint {0}.'.format(self.current_epoch, checkpoint_type))
    
        
    def load_checkpoint(self, path):
        """
        Loads a checkpoint and initialises the models to continue training.
        """
        checkpoint = torch.load(path, map_location=self.device)
                
        self.init_generator()
        self.init_discriminator()
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        self.gen_lr = checkpoint['current_gen_lr']
        self.gen_lr_update_epoch = checkpoint['gen_lr_update_epoch']
        self.new_gen_lr = checkpoint['new_gen_lr']
        self.gen_lr_schedule = checkpoint['gen_lr_schedule']
        
        self.dis_lr = checkpoint['current_dis_lr']
        self.dis_lr_update_epoch = checkpoint['dis_lr_update_epoch']
        self.new_dis_lr = checkpoint['new_dis_lr']
        self.dis_lr_schedule = checkpoint['dis_lr_schedule']
        
        self.init_optimisers()
        self.optimiser_D.load_state_dict(checkpoint['optimiser_D_state_dict'])
        self.optimiser_G.load_state_dict(checkpoint['optimiser_G_state_dict'])
                
        self.snapshot = checkpoint['snapshot']
        self.console_print = checkpoint['console_print']
        self.current_epoch = checkpoint['last_epoch']
        self.start_epoch = checkpoint['last_epoch'] + 1
        self.epoch_losses = checkpoint['epoch_losses']

        print(('\nCheckpoint loaded.\n' + 
               ' *- Last epoch {0} with loss {1}.\n' 
               ).format(checkpoint['last_epoch'], 
               checkpoint['last_epoch_loss']))
        print(' *- Generator:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                      self.gen_lr, self.gen_lr_update_epoch, self.new_gen_lr)
              )
        print(' *- Discriminator:' +
              ' *- Current lr {0}, next update on epoch {1} to the value {2}'.format(
                self.dis_lr, self.dis_lr_update_epoch, self.new_dis_lr)
              )

        self.generator.train()
        self.discriminator.train()
        assert(self.generator.training)
        
    def load_model(self, eval_config):
        """Loads a trained GAN model into eval mode"""   

        filename = eval_config['filepath']
        model_dict = torch.load(filename, map_location=self.device)

        # Load the Generator
        g_config = self.config['generator_config']
        generator = getattr(GAN_models, g_config['class_name'])
        self.generator = generator(g_config).to(self.device)
        g_model = model_dict['generator']
        if eval_config['load_checkpoint']:
            self.generator.load_state_dict(g_model['model_state_dict'])
            print(' *- Loaded checkpoint.')
        else:
            self.generator.load_state_dict(g_model)

        # Load the discriminator        
        d_config = self.config['discriminator_config']
        discriminator = getattr(GAN_models, d_config['class_name'])
        self.discriminator = discriminator(d_config).to(self.device)
        d_model = model_dict['discriminator']
        if eval_config['load_checkpoint']:
            self.discriminator.load_state_dict(d_model['model_state_dict'])
            print(' *- Loaded discriminator checkpoint.')
        else:
            self.discriminator.load_state_dict(d_model)
        
        self.generator.eval()
        self.discriminator.eval()
        assert(not self.generator.training)

# --------------- #
# --- Testing --- #
# --------------- #
if __name__ == '__main__':
    config = {
        'discriminator_config': {
            'class_name': 'ConvolutionalDiscriminator',
            'channel_dims': [3, 64, 128, 256, 1]
            },

        'generator_config': {
            'class_name': 'ConvolutionalGenerator',
            'latent_dim': 100,
            'channel_dims': [256, 128, 64, 3]
            },
        
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 100,
                'path_to_data': '../datasets/MNIST'
                },
                
        'train_config': {
                'batch_size': 128,
                'epochs': 100,
                'snapshot': 20, 
                'console_print': 1,
                'optim_type': 'Adam',
                'gen_lr_schedule': [(0, 1e-3)],
                'gen_b1': 0.5,
                'gen_b2': 0.999,
                'dis_lr_schedule': [(0, 2e-4)],
                'dis_b1': 0.5,
                'dis_b2': 0.999,
                
                'filename': 'gan',
                'random_seed': 1201,
                'exp_dir': 'models/DUMMY'
                }
        }
    
    model = GAN(config)
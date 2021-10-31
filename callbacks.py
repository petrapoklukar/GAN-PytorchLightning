from pytorch_lightning.callbacks import Callback
import torch
import os

class OnEndTraining(Callback):

    def on_init_end(self, trainer):
        print(f'Initialised Trainer with {trainer.default_root_dir}')

    def on_train_end(self, trainer, pl_module):
        torch.save({'generator_state_dict': pl_module.generator.state_dict(),
                    'discriminator_state_dict': pl_module.discriminator.state_dict()}, 
                   os.path.join(trainer.default_root_dir, "model.pt")
                   )
        pl_module.log_generated_images('generated_images_after_training')
        print(f'Model {pl_module.experiment_id} trained for {trainer.max_epochs} saved to {trainer.default_root_dir}')

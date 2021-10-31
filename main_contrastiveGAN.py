import os
import typer
import torch
from schemes import DataInput, GANBaseInput, GANTrainInput
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, GPUStatsMonitor
from pytorch_lightning.profiler import AdvancedProfiler
from callbacks import OnEndTraining
from lightning_modules.contrastiveGAN_Trainer import ContrastiveGAN
from data_modules.CIFAR10_contrastiveGAN import CIFAR10_contrastiveGAN_DataModule
import json

AVAIL_GPUS = min(1, torch.cuda.device_count())
app = typer.Typer()

def read_json_hyperparameters(dir):
    gan_base_input = GANBaseInput(**json.load(open(os.path.join(dir, "GANBaseInput.json"), "r")))
    gan_train_input = GANTrainInput(**json.load(open(os.path.join(dir, "GANTrainInput.json"), "r")))
    data_input = DataInput(**json.load(open(os.path.join(dir, "DataInput.json"), "r")))
    return gan_base_input, gan_train_input, data_input

@app.command()
def train_in_debug(experiment_id:str='test_contrastiveGAN_cifar10', checkpoint_path:str=None):
    # Manual debug parameters
    TEST_limit_train_batches = 0.05
    # checkpoint_path = '1234_test-epoch=07.pth'
    
    # Init GAN input parameters
    model_config = GANBaseInput(output_channels=3)
    train_config = GANTrainInput(epochs=20, snapshot=2)
    
    # Set the seeds
    seed_everything(train_config.random_seed, workers=True)
    
    # Init model
    gan_model = ContrastiveGAN(model_config, train_config, experiment_id)
    
    # Init data
    data_config = DataInput(batch_size=4)
    data_model = CIFAR10_contrastiveGAN_DataModule(data_config)
    
    # Train
    save_dir = os.path.join('models', experiment_id)
    logger = TensorBoardLogger(os.path.join(save_dir, 'tb_logs'), name=experiment_id, 
                               default_hp_metric=False)
    
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    # Resume training
    if checkpoint_path:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{experiment_id}-'+'{epoch:02d}',
        monitor='epoch',
        period=train_config.snapshot,
        save_top_k=-1, 
        save_last=True
        )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    checkpoint_callback.FILE_EXTENSION = ".pth"
    
    gpu_stats_callback = GPUStatsMonitor() 
    on_end_training_callback = OnEndTraining()
    profiler = AdvancedProfiler(filename='profiler_out')
    
    trainer = Trainer(
        gpus=AVAIL_GPUS, 
        max_epochs=train_config.epochs, 
        progress_bar_refresh_rate=1,
        default_root_dir=save_dir, 
        logger=logger, 
        callbacks=[checkpoint_callback, gpu_stats_callback, on_end_training_callback], 
        flush_logs_every_n_steps=train_config.snapshot, 
        profiler=profiler,
        resume_from_checkpoint=checkpoint_path,
        limit_train_batches=TEST_limit_train_batches
        )
    trainer.fit(gan_model, data_model)


@app.command()
def eval():
    typer.echo(f"Not implemented.")


@app.command()
def train(experiment_id:str, checkpoint_path:str=None):
    save_dir = os.path.join('models', experiment_id)
    # Folder doesn't already exists but it is pending
    if os.path.isdir(save_dir):
        pass
    elif os.path.isdir(os.path.join('pending', experiment_id)):
        os.system(f"mv pending/{experiment_id} {save_dir}")
    else:
        raise ValueError(f'Hyperparameters for {experiment_id} not found.')
    
    # Init GAN input parameters
    model_config, train_config, data_config = read_json_hyperparameters(save_dir)

    # Set the seeds
    seed_everything(train_config.random_seed, workers=True)
    
    # Init model
    gan_model = ContrastiveGAN(model_config, train_config, experiment_id)
    data_model = CIFAR10_contrastiveGAN_DataModule(data_config)
    
    # Train
    logger = TensorBoardLogger(os.path.join(save_dir, 'tb_logs'), name=experiment_id, 
                               default_hp_metric=False)
    
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')
    
    # Resume training
    if checkpoint_path:
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f'{experiment_id}-'+'{epoch:02d}',
        monitor='epoch',
        period=train_config.snapshot,
        save_top_k=-1, 
        save_last=True
        )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    checkpoint_callback.FILE_EXTENSION = ".pth"
    
    gpu_stats_callback = GPUStatsMonitor() 
    on_end_training_callback = OnEndTraining()
    profiler = AdvancedProfiler(filename='profiler_out')
    
    trainer = Trainer(
        gpus=AVAIL_GPUS, 
        max_epochs=train_config.epochs, 
        progress_bar_refresh_rate=1,
        default_root_dir=save_dir, 
        logger=logger, 
        callbacks=[checkpoint_callback, gpu_stats_callback, on_end_training_callback], 
        flush_logs_every_n_steps=train_config.snapshot, 
        profiler=profiler,
        resume_from_checkpoint=checkpoint_path
        )
    trainer.fit(gan_model, data_model)


if __name__ == "__main__":
    typer.run(train_in_debug)
    # app()


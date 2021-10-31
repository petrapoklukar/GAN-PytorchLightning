from typing import Optional
from schemes import GANBaseInput, GANTrainInput, DataInput
import os
import uuid
import time
import json
import dataclasses
import inspect

def set_GANBaseInput(update_dict: dict, dir:str):
    gan_base_input = GANBaseInput(**update_dict)
    print("GANBaseInput parameters set.")
    save_json(dataclasses.asdict(gan_base_input), dir, 'GANBaseInput')

def set_GANTrainInput(update_dict: dict, dir:str):
    gan_train_input = GANTrainInput(**update_dict)
    print("GANTrainInput parameters set.")
    save_json(dataclasses.asdict(gan_train_input), dir, 'GANTrainInput')

def set_DataInput(update_dict: dict, dir:str):
    data_input = DataInput(**update_dict) if update_dict else DataInput()
    print("DataInput parameters set.")
    save_json(dataclasses.asdict(data_input), dir, 'DataInput')

def save_json(input_d:dict, dir:str, filename:str):
    json.dump(input_d, open(os.path.join(dir, f"{filename}.json"), "w"))

def parse_params(args, values):
    GANBaseInput_dict = {}
    GANTrainInput_dict = {}
    DataInput_dict = {}
    args.remove('experiment_name')
    for arg in args:
        if values[arg]:
            arg_type, arg_name = arg.split('_', 1)
            if 'GANBaseInput' in arg_type:
                GANBaseInput_dict[arg_name] = values[arg]
            if 'GANTrainInput' in arg_type:
                GANTrainInput_dict[arg_name] = values[arg]
            if 'DataInput' in arg_type:
                DataInput_dict[arg_name] = values[arg]
    return GANBaseInput_dict, GANTrainInput_dict, DataInput_dict
    

def create_experiment_config(experiment_name:str,
                             GANBaseInput_generator_dropout:Optional[float] = None, 
                             GANBaseInput_generator_latent_dim:Optional[int] = None, 
                             GANBaseInput_representation_dim:Optional[int] = None, 
                             GANBaseInput_discriminator_dropout:Optional[float] = None, 
                             GANBaseInput_output_channels:Optional[int] = None, 
                             
                             GANTrainInput_epochs:Optional[int] = None, 
                             GANTrainInput_snapshot:Optional[int] = None, 
                             GANTrainInput_random_seed:Optional[int] = None, 
                             
                             DataInput_data_dir:Optional[str] = None, 
                             DataInput_batch_size:Optional[int] = None, 
                             DataInput_num_workers:Optional[int] = None
                            ):
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    GANBaseInput_dict, GANTrainInput_dict, DataInput_dict = parse_params(args, values)
        
    # generate uniqueID
    timestamp = time.strftime('%b%d%y_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    exp_name = f'{experiment_name}_{timestamp}_{unique_id}'
    pending_exp_dir = os.path.join('pending', exp_name)
    
    if (not os.path.isdir(pending_exp_dir)):
        os.makedirs(pending_exp_dir)
    set_GANBaseInput(GANBaseInput_dict, pending_exp_dir)
    set_GANTrainInput(GANTrainInput_dict, pending_exp_dir)
    set_DataInput(DataInput_dict, pending_exp_dir)
    print(f"Experiment parameters pending in {pending_exp_dir}.")
        

if __name__ == "__main__":
    # create_experiment_config()
###########################
### StyleGAN 2 TRAINING ###
###########################

import warnings
from train import Train
from utils import build_folder_structure, get_iter

import torchaudio


# TODO:
# Add logger.
# Add profiler.


config_dict = {
    # Iterarion
    'iter_num': None,
    'epochs': 500,
    'datadir': 'datasets/dataset_2/',

    # Training
    'batch_size': 8,
    
    # Learning
    'learning_rate': 0.002,

    # Model
    'scale_factor': 4.0,
    'depth': 6,
    'num_filters': 512,
    'start_size': 32,

    # Setup
    'num_channels': 1,
    'sample_rate': 44100,
    'save_every': 1,
    'num_workers': 1,
    'device': 'cuda',
}


def suppress_warnings():
    warnings.filterwarnings("ignore", category = UserWarning)


def prepare_config():
    if config_dict['iter_num'] == None:
        config_dict['iter_num'] = get_iter()
        restart = False
    else:
        restart = True

    build_folder_structure(config_dict['iter_num'])
    
    # Write config to file
    f = open("runs/iter_" + str(config_dict['iter_num']) + "/log", "w")
    f.write(str(config_dict))
    f.close()

    return restart


if __name__ == '__main__':
    suppress_warnings()
    restart = prepare_config()
    
    Train(
        # Iteration
        restart = restart,
        iter_num = config_dict['iter_num'],
        epochs = config_dict['epochs'],
        datadir = config_dict['datadir'],
        
        # Training
        batch_size = config_dict['batch_size'], 
        
        # Learning
        learning_rate = config_dict['learning_rate'],

        # Model
        scale_factor = config_dict['scale_factor'],
        depth = config_dict['depth'],
        num_filters = config_dict['num_filters'],
        start_size = config_dict['start_size'],
        
        # Setup
        sample_rate = config_dict['sample_rate'],
        num_channels = config_dict['num_channels'],
        save_every = config_dict['save_every'],
        num_workers = config_dict['num_workers'],
        device = config_dict['device']
    )
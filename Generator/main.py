###########################
### StyleGAN 2 TRAINING ###
###########################

from train import Train
from utils import build_folder_structure, get_iter


### add logger
### add profiler
### add equalized learning to sa module
### add demodulation 
### define custom up/downsample function
### change loss


config_dict = {
    # Iterarion
    'restart_from_iter': False,
    'restart_iter_num': None,
    'epochs': 500,
    'datadir': 'datasets/dataset_4/',

    # Training
    'batch_size': 4,
    
    # Learning
    'learning_rate': 0.002,

    # Model
    'scale_factor': 4,
    'depth': 7,
    'num_filters': 512,
    'start_size': 32,

    # Setup
    'num_channels': 1,
    'sample_rate': 44100,
    'save_every': 1,
    'num_workers': 1,
    'device': 'cuda',
}


if __name__ == '__main__':
    if config_dict['restart_from_iter'] == True:
        iter_num = config_dict['restart_iter_num']
    else:
        iter_num = get_iter()
        config_dict['restart_iter_num'] = None

    build_folder_structure(iter_num)
    
    # Write config to file
    f = open("runs/iter_" + str(iter_num) + "/log", "w")
    f.write(str(config_dict))
    f.close()
    
    Train(
        # Iteration
        restart_from_iter = config_dict['restart_from_iter'],
        iter_num = iter_num,
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

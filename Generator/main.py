import warnings
from train import Train
from misc.utils import build_folder_structure, get_iter


# TODO:
# Add logger.
# Add profiler.

config_dict = {
    'program_version': 0.9,

    # Iterarion
    'iter_num': None,
    'epochs': 500,
    'datadir': 'datasets/dataset_3/',

    # Training
    'batch_size': 8,
    'learning_rate': 0.002,
    'g_loss': 'wgan',
    'd_loss': 'wgangp',

    # Model
    'scale_factor': 4,
    'depth': 4,
    'num_filters': 512,
    'start_size': 64,

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
        learning_rate = config_dict['learning_rate'],
        g_loss = config_dict['g_loss'],
        d_loss = config_dict['d_loss'],

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
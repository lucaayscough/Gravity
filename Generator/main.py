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
    'restart_from_iter': True,
    'restart_iter_num': 2,
    'epochs': 500,
    'datadir': 'datasets/dataset_4/',

    # Training
    'batch_size': 4,
    
    # Learning
    'learning_rate': 0.002,

    # Generator
    'use_linear_upsampling': False,
    'use_pixel_norm': False,
    'use_instance_norm': True,
    'use_style': True,
    'use_noise': True,
    'use_ema': True,
    'ema_beta': 0.999,

    # Discriminator
    'use_linear_downsampling': False,

    # Model
    'z_dim': 512,
    'scale_factor': 4,
    'depth': 7,
    'num_filters': 512,
    'kernel_size': 9,
    'stride': 1,
    'padding': 4,
    'dilation': 1,
    'start_size': 32,
    'use_self_attention': False,

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
        
        # Generator
        use_linear_upsampling = config_dict['use_linear_upsampling'],
        use_pixel_norm = config_dict['use_pixel_norm'],
        use_instance_norm = config_dict['use_instance_norm'],
        use_style = config_dict['use_style'],
        use_noise = config_dict['use_noise'],
        use_ema = config_dict['use_ema'],
        ema_beta = config_dict['ema_beta'],

        # Discriminator
        use_linear_downsampling = config_dict['use_linear_downsampling'],

        # Model
        z_dim = config_dict['z_dim'],
        scale_factor = config_dict['scale_factor'],
        depth = config_dict['depth'],
        num_filters = config_dict['num_filters'],
        kernel_size = config_dict['kernel_size'],
        stride = config_dict['stride'],
        padding = config_dict['padding'],
        dilation = config_dict['dilation'],
        start_size = config_dict['start_size'],
        use_self_attention = config_dict['use_self_attention'],
        
        # Setup
        sample_rate = config_dict['sample_rate'],
        num_channels = config_dict['num_channels'],
        save_every = config_dict['save_every'],
        num_workers = config_dict['num_workers'],
        device = config_dict['device']
    )

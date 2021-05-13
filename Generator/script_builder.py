import torch
from script_model import Generator


# Instantiate model
netG = Generator(
    z_dim = 512,
    nf = 512,
    kernel_size = 9,
    stride = 1,
    padding = 4,
    dilation = 1,
    depth = 7,
    num_channels = 1,
    scale_factor = 4,
    use_linear_upsampling = False,
    use_self_attention = False,
    use_pixel_norm = False,
    use_instance_norm = True,
    use_style = True,
    use_noise = True,
    start_size = 32
)


# Load model.
checkpointG = torch.load("runs/iter_2/model/checkpoint_027_G.pth.tar")
netG.load_state_dict(checkpointG['state_dict'])


# Serialization.
script_module = torch.jit.script(netG)
script_module.save("traced_generator.pt")

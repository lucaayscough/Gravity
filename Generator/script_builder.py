import torch
from script_model import Generator, Mapper


# Instantiate model
mapper = Mapper(
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


generator = Generator(
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


model_dir = "runs/iter_2/model/checkpoint_052_G.pth.tar"


# Load model.
checkpointG = torch.load(model_dir)

mapper.load_state_dict(checkpointG['state_dict'])
generator.load_state_dict(checkpointG['state_dict'])

# Serialization.
mapper_module = torch.jit.script(mapper)
mapper_module.save("scripted_modules/mapper_module.pt")

generator_module = torch.jit.script(generator)
generator_module.save("scripted_modules/generator_module.pt")

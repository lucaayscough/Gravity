import torch
from torch import Tensor
import torchaudio
import numpy as np
import random

# ------------------------------------------------------------
# Scripted functions.

@torch.jit.script
def normalize(x: Tensor, epsilon: float=1e-8) -> Tensor:
    return x * (x.square().mean(dim=1, keepdim=True) + epsilon).rsqrt()

@torch.jit.script
def modulate(x: Tensor, style: Tensor, weight: Tensor, demodulate: bool=True):
    batch_size = x.size(0)
    out_channels, in_channels, ks = weight.shape
    weight = weight.unsqueeze(0)
    weight = weight * style.reshape(batch_size, 1, -1, 1)
    if demodulate:
        dcoefs = (weight.square().sum(dim=[2,3]) + 1e-8).rsqrt()
        weight = weight * dcoefs.reshape(batch_size, -1, 1, 1)
    x = x.reshape(1, -1, x.shape[2])
    weight = weight.reshape(-1, in_channels, ks)
    return x, weight

@torch.jit.script
def mini_batch_std_dev(x: Tensor, group_size: int=4, num_channels: int=1, alpha: float=1e-8) -> Tensor:    
    N, C, S = x.shape
    G = torch.min(torch.as_tensor(group_size), torch.as_tensor(N))
    F = num_channels
    c = C // F

    y = x.reshape(G, -1, F, c, S)       # [GnFcS]   Split minibatch N into n groups of size G, and channels C into F groups of size c.
    y = y - y.mean(dim=0)               # [GnFcS]   Subtract mean over group.
    y = y.square().mean(dim=0)          # [nFcS]    Calc variance over group.
    y = (y + 1e-8).sqrt()               # [nFcS]    Calc stddev over group.
    y = y.mean(dim=[2, 3])              # [nF]      Take average over channels and pixels.
    y = y.reshape(-1, F, 1)             # [nF1]     Add missing dimensions.
    y = y.repeat(G, 1, S)               # [NFS]     Replicate over group and pixels.
    return torch.cat([x, y], dim=1)     # [NCS]     Append to input as new channels.

# ------------------------------------------------------------
# Fully-connected layer.

class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_channels: int,                   # Number of input features.
        out_channels: int,                  # Number of output features.
        bias: bool              = True,     # Apply additive bias before the activation function?
        activation: str         = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier: float    = 1,        # Learning rate multiplier.
        bias_init: float        = 0,        # Initial value for the additive bias.
    ) -> None:
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_channels], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_channels)
        self.lr_multiplier = lr_multiplier

    def forward(self, x: Tensor, alpha: float=0.2) -> Tensor:
        weight = self.weight.to(x.dtype) * self.weight_gain
        bias = self.bias

        if bias is not None:
            bias = bias.to(x.dtype)
            if self.lr_multiplier != 1:
                bias = (bias * self.lr_multiplier).unsqueeze(0)
        
        if self.activation == "linear" and bias is not None:
            return torch.addmm(bias, x, weight.t())
        else:
            x = x.matmul(weight.t())

        if bias is not None:
            x = x + bias
        
        if self.activation == "lrelu":
            x = torch.nn.functional.leaky_relu(x, alpha)
    
        return x


# ------------------------------------------------------------
# Convolution layer.

class Conv1dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int        = 1,
        stride: int             = 1,
        padding: int            = 0,
        resample_filter: Tensor = None,
        bias: bool              = True,
        apply_style: bool       = False,
        apply_noise: bool       = False,
        up: int                 = 1,
        down: int               = 1,
        w_dim: int              = 512,
        to_sound: bool          = False
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.resample_filter = resample_filter
        self.up = up
        self.down = down
        self.apply_style = apply_style
        self.apply_noise = apply_noise
        self.to_sound = to_sound

        if apply_style:
            self.style_affine = FullyConnectedLayer(w_dim, in_channels, activation="lrelu", bias_init=1)

        if apply_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([out_channels]))

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size]))

        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels])) if bias is not None else None

    def forward(
        self,
        x: Tensor,
        style: Tensor       = None,
        gain: float         = 1.0,
        alpha: float        = 0.2
    ) -> Tensor:
        # Setup bias.
        bias = self.bias 
        if bias is not None:
            bias = bias.to(x.dtype).reshape(1, -1, 1)

        # Setup weights.
        weight = self.weight
        batch_size = x.size(0)
        out_channels, in_channels, ks = weight.shape

        if self.apply_style and not self.to_sound:
            style = self.style_affine(style)
            x, weight = modulate(x, style, weight)
            groups = batch_size

        elif self.apply_style and self.to_sound:
            style = self.style_affine(style) * self.weight_gain
            x, weight = modulate(x, style, weight, False)
            groups = batch_size

        else:
            weight = weight * self.weight_gain
            groups = 1

        # Flip weights if upsampling.
        if self.up != 1:
            weight = weight.flip([2])

        # Blur input and upsample with transposed convolution.
        if self.up > 1:
            x = self.resample_filter(x)

        # Do convolution.
        x = torch.nn.functional.conv1d(x, weight, stride=self.stride, padding=self.padding, groups=groups)
        
        # Downsample with convolution and blur output.
        if self.down > 1:
            x = self.resample_filter(x)

        # Demodulate weights.
        if self.apply_style:
            x = x.reshape(batch_size, -1, x.shape[2])

        # Add noise.
        if self.apply_noise:
            noise = torch.randn(batch_size, 1, x.size(2), device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_strength.view(1, -1, 1)

        # Add bias and activation function.
        if bias is not None:
            x = x + bias

        x = torch.nn.functional.leaky_relu(x, alpha)
        
        if gain != 1:
            x = x * gain
        
        return x

# ------------------------------------------------------------
# Constant input.

class ConstantInput(torch.nn.Module):
    def __init__(self, nf: int, start_size: int):
        super().__init__()

        self.constant_input = torch.nn.Parameter(torch.randn(1, nf, start_size))
        self.bias = torch.nn.Parameter(torch.zeros(nf))
    
    def forward(self, batch_size: int) -> Tensor:
        x = self.constant_input.expand(batch_size, -1, -1)
        x = x + self.bias.view(1, -1, 1)
        return x

# ------------------------------------------------------------
# Mapping network.

class MappingNetwork(torch.nn.Module):
    def __init__(self,
        broadcast: int,                     # Latent broadcasting for style blocks.
        depth: int              = 8,        # Network layer depth.
        z_dim: int              = 512,      # Dimensions in each fully connected layer.
        lr_multiplier: float    = 0.01,     # Learning rate multiplier.
        w_avg_beta              = 0.995,    # Decay for tracking the moving average of W during training.
    ) -> None:
        super().__init__()

        self.broadcast = broadcast
        self.w_avg_beta = w_avg_beta

        self.layers = torch.nn.ModuleList([])
        for l in range(depth):
            self.layers.append(FullyConnectedLayer(in_channels=z_dim, out_channels=z_dim, lr_multiplier=lr_multiplier, activation="lrelu"))

        self.register_buffer('w_avg', torch.zeros([z_dim]))

    def forward(self,
        x: Tensor,                          # Input tensor.
        alpha: float            = 0.2,      # Alpha parameter for activation function.
        truncation_psi: float   = 1,        # Value for truncation psi.
        truncation_cutoff: int  = None,     # Value for truncation cutoff.
        skip_w_avg_update: bool = False     # Determines if the average of the weights should be updated.
    ) -> Tensor:
        x = normalize(x)

        for layer in self.layers:
            x = layer(x)

        if skip_w_avg_update is not True:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        x = x.unsqueeze(1).repeat(1, self.broadcast, 1)

        if truncation_psi != 1:
            if truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)

        return x

# ------------------------------------------------------------
# Synthesis block.
    
class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int,
        scale_factor: int
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        self.resample_filter = torchaudio.transforms.Resample(orig_freq=1, new_freq=scale_factor, rolloff=0.9, dtype=torch.float32)

        self.block_1 = Conv1dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=9, padding=4, apply_style=True, apply_noise=True, up=scale_factor, resample_filter=self.resample_filter)
        self.block_2 = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=9, padding=4, apply_style=True, apply_noise=True, resample_filter=self.resample_filter)

        self.converter = Conv1dLayer(in_channels=out_channels, out_channels=num_channels, bias=True, apply_style=True, to_sound=True)

    def forward(self, x: Tensor, latent_w: Tensor, sound: Tensor=None):
        x = self.block_1(x, latent_w[:, 0])
        x = self.block_2(x, latent_w[:, 1])
        y = self.converter(x, latent_w[:, 2])
        
        if sound is not None:
            sound = self.resample_filter(sound)
            sound = sound.add_(y)
        else:
            sound = y

        #torchaudio.save(filepath = 'runs/an/' + '_analyses' + str(i) + '.wav', src=copy[0].detach().cpu(), sample_rate=44100)
        return x, sound

# ------------------------------------------------------------
# Synthesis network.
    
class SynthesisNetwork(torch.nn.Module):
    def __init__(
        self,
        nf: int,
        depth: int,
        num_channels: int,
        scale_factor: int
    ):
        super().__init__()

        self.depth = depth

        self.blocks = torch.nn.ModuleList([])
        c = 1 / np.sqrt(0.5)
        for i in range(depth):
            self.blocks.append(SynthesisBlock(in_channels=int(nf), out_channels=int(nf/c), num_channels=num_channels, scale_factor=scale_factor))
            nf = nf / c

    def forward(self, x: Tensor, latent_w: Tensor):
        for i in range(self.depth): 
            if i == 0:
                x, sound = self.blocks[i](x=x, latent_w=latent_w[:, 3 * i : 3 * i + 3])
            else:
                x, sound = self.blocks[i](x=x, latent_w=latent_w[:, 3 * i : 3 * i + 3], sound=sound)
            copy = sound
            #torchaudio.save(filepath = 'runs/an/' + '_analyses' + str(i) + '.wav', src=copy[0].detach().cpu(), sample_rate=44100)
        return sound

# ------------------------------------------------------------
# Generator network.

class Generator(torch.nn.Module):
    def __init__(
        self,
        nf: int,
        depth: int,
        num_channels: int,
        scale_factor: int,
        start_size: int
    ):
        super().__init__()
        
        self.depth = depth
        
        self.constant_input = ConstantInput(nf, start_size)
        self.mapping_network = MappingNetwork(depth * 3)
        self.synthesis_network = SynthesisNetwork(nf, depth, num_channels, scale_factor)

    def forward(self,
        latent_z: Tensor,
        latent_w: Tensor    = None,
        return_w: bool      = False
    ):
        x = self.constant_input(batch_size=latent_z.size(0))
        latent_w = self.mapping_network(latent_z, truncation_psi=1, truncation_cutoff=None)

        # TODO:
        # Style mixing.
        #latent_w = self._mixing_regularization(latent_z, latent_w, self.depth)

        sound = self.synthesis_network(x, latent_w)
    
        if return_w:
            return sound, latent_w
        else:
            return sound

    def _mixing_regularization(self, latent_z, latent_w, depth):
        latent_z_2 = torch.randn(latent_z.shape).to(latent_z.device)
        latent_w_2 = self.mapping_network(latent_z_2)

        layer_idx = torch.from_numpy(np.arange(self.depth * 3)[np.newaxis, :, np.newaxis]).to(latent_z.device)
        cur_layers = 2 * (depth + 1)

        mixing_cutoff = random.randint(1, depth + 1) if random.random() < 0.9 else cur_layers
        latent_w = torch.where(layer_idx < mixing_cutoff, latent_w, latent_w_2)
        return latent_w


# ------------------------------------------------------------
# General discriminator block.

class DicriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        scale_factor: int,
        resample_filter: Tensor = None
    ):
        super().__init__()

        self.conv_block_1 = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=9, padding=4, down=scale_factor, resample_filter=resample_filter)
        self.conv_block_2 = Conv1dLayer(in_channels=out_channels, out_channels=out_channels, kernel_size=9, padding=4)
        self.residual = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, bias=False, down=scale_factor, resample_filter=resample_filter)

    def forward(self, x: Tensor) -> Tensor:
        y = self.residual(x, gain=np.sqrt(0.5))
        
        x = self.conv_block_1(x)
        x = self.conv_block_2(x, gain=np.sqrt(0.5))
        x = x.add_(y)
        return x

# ------------------------------------------------------------
# Final discriminator block.

class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_channels,
        scale_factor: int,
        start_size: int,
        resample_filter: Tensor = None,
    ):
        super().__init__()
        
        in_channels += 1

        self.conv_block = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=9, padding=4, down=scale_factor, resample_filter=resample_filter)
        
        self.fc = FullyConnectedLayer(in_channels=out_channels*start_size, out_channels=out_channels, activation="lrelu")
        self.out_fc = FullyConnectedLayer(in_channels=out_channels, out_channels=num_channels, activation="lrelu")
    
    def forward(self, x: Tensor, group_size: int=4) -> Tensor:
        x = mini_batch_std_dev(x, group_size)
        x = self.conv_block(x)
        x = self.fc(x.flatten(1))
        x = self.out_fc(x)
        return x


# ------------------------------------------------------------
# Discriminator network.

class Discriminator(torch.nn.Module):
    def __init__(self,
        nf: float,
        depth,
        num_channels,
        scale_factor: int,
        start_size
    ):
        super().__init__()
    
        self.depth = depth
        self.resample_filter = torchaudio.transforms.Resample(orig_freq=scale_factor, new_freq=1, rolloff=0.9, dtype=torch.float32)

        self.layers = torch.nn.ModuleList([])
     
        n = nf
        c = 1 / np.sqrt(0.5)
        for l in range(depth - 1):
            self.layers.append(DicriminatorBlock(in_channels=int(n), out_channels=int(n*c), scale_factor=scale_factor, resample_filter=self.resample_filter))
            n = n*c

        self.layers.append(DiscriminatorEpilogue(in_channels=int(n), out_channels=int(n*c), num_channels=num_channels, scale_factor=scale_factor, start_size=start_size, resample_filter=self.resample_filter))
        
        self.converter = Conv1dLayer(in_channels=num_channels, out_channels=int(nf), bias=False)

    def forward(self, x):
        x = self.converter(x)

        for i in range(self.depth):
            x = self.layers[i](x)
        return x

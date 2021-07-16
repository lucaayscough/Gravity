import torch
from torch import Tensor
import numpy as np
import random


resample_kernel = [1, 2, 4, 2, 1]
resample_kernel = torch.tensor(resample_kernel, dtype=torch.float)
resample_kernel = resample_kernel.expand(1, 1, -1)
resample_kernel = resample_kernel / resample_kernel.sum()


# ------------------------------------------------------------
# Scripted functions.

@torch.jit.script
def normalize(x: Tensor, epsilon: float=1e-8) -> Tensor:
    return x / (x.pow(2.0).mean(dim=1, keepdim=True).add(epsilon).sqrt())

@torch.jit.script
def modulate(x: Tensor, styles: Tensor, weight: Tensor):
    w = weight.unsqueeze(0)
    w = w * styles.reshape(x.size(0), 1, -1, 1)
    torch.flip(w, [0, 1, 2, 3])
    dcoefs = (w.square().sum(dim=[2,3]) + 1e-8).rsqrt()
    x = x * styles.reshape(x.size(0), -1, 1)
    return x, dcoefs

@torch.jit.script
def demodulate(x: Tensor, dcoefs: Tensor) -> Tensor:
    return x * dcoefs.reshape(x.size(0), -1, 1)

@torch.jit.script
def blur(x: Tensor, kernel: Tensor) -> Tensor:
    return torch.nn.functional.conv1d(x, kernel, stride=1, padding=2, groups=x.size(1))

@torch.jit.script
def blur_down(x: Tensor, kernel: Tensor, scale_factor: float) -> Tensor:
    x = torch.nn.functional.conv1d(x, kernel, stride=1, padding=2, groups=x.size(1))
    return torch.nn.functional.interpolate(input=x, scale_factor=scale_factor, mode="linear")

@torch.jit.script
def blur_up(x: Tensor, kernel: Tensor, scale_factor: float) -> Tensor:
    x = torch.nn.functional.interpolate(input=x, scale_factor=scale_factor, mode="linear")
    return torch.nn.functional.conv1d(x, kernel, stride=1, padding=2, groups=x.size(1))

#@torch.jit.script
def resample(x: Tensor, scale_factor: float) -> Tensor:
    kernel = resample_kernel.expand(x.size(1), -1, -1).to(x.device)
    if scale_factor < 1:
        return blur_down(x, kernel, scale_factor)
    if scale_factor > 1:
        return blur_up(x, kernel, scale_factor)

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
# Low level network components.

# ------------------------------------------------------------
# Convolution layer with equalized learning.

class Conv1dLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool          = True,
        apply_style: bool   = False,
        apply_noise: bool   = False,
        scale_factor: float = 1
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.scale_factor = scale_factor

        self.apply_style = apply_style
        self.apply_noise = apply_noise

        if apply_style:
            self.style_affine = FullyConnectedLayer(512, in_channels, activation="lrelu")

        if apply_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([out_channels]))

        weight = torch.randn([out_channels, in_channels, kernel_size])
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        bias = torch.zeros([out_channels]) if bias else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
    
    def forward(
        self,
        x: Tensor,
        style: Tensor       = None,
        gain: float         = 1.0,
        alpha: float        = 0.2
    ) -> Tensor:
        weight = self.weight * self.weight_gain
        bias = self.bias

        if bias is not None:
            bias = bias.to(x.dtype).reshape(1, -1, 1)

        # Modulate weights.
        if self.apply_style:
            style = self.style_affine(style)
            x, dcoefs = modulate(x, style, weight)

        # Resample input.
        if self.scale_factor != 1:
            x = resample(x, self.scale_factor)

        x = torch.nn.functional.conv1d(x, weight, stride=self.stride, padding=self.padding)

        # Demodulate weights.
        if self.apply_style:
            x = demodulate(x, dcoefs)

        # Add noise.
        if self.apply_noise:
            noise = torch.randn(x.size(0), 1, x.size(2), device=x.device, dtype=x.dtype)
            x = x + noise * self.noise_strength.view(1, -1, 1)

        # Add bias and activation function.
        if bias is not None:
            x = x + bias

        x = torch.nn.functional.leaky_relu(x, alpha)
        
        if gain != 1:
            x = x * gain
        
        return x

# ------------------------------------------------------------
# Fully-connected layer with equalized learning.

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
# Blocks composed of low level network components.

# ------------------------------------------------------------
# General generator block.

class StyleBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale_factor: float,
        bias = True,
    ):
        super().__init__()
        
        self.conv_block_1 = Conv1dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, apply_style=True, apply_noise=True, bias=bias, scale_factor=scale_factor)
        self.conv_block_2 = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, apply_style=True, apply_noise=True, bias=bias)

    def forward(self, x: Tensor, latent_w: Tensor) -> Tensor:
        x = self.conv_block_1(x, latent_w[:, 0])
        return self.conv_block_2(x, latent_w[:, 1], gain=np.sqrt(0.5))

# ------------------------------------------------------------
# General discriminator block.

class DicriminatorBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale_factor: float,
        bias = True
    ):
        super().__init__()

        self.conv_block_1 = Conv1dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, scale_factor=scale_factor)
        self.conv_block_2 = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.residual = Conv1dLayer(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, scale_factor=scale_factor)

    def forward(self, x: Tensor) -> Tensor:
        y = self.residual(x, gain=np.sqrt(0.5))

        x = self.conv_block_1(x)
        x = self.conv_block_2(x, gain=np.sqrt(0.5))
        return x + y

# ------------------------------------------------------------
# Final discriminator block.

class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_channels,
        kernel_size,
        stride,
        padding,
        scale_factor: float,
        bias = True
    ):
        super().__init__()

        in_channels += 1

        self.conv_block_1 = Conv1dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, scale_factor=scale_factor)
        self.conv_block_2 = Conv1dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        
        self.fc = FullyConnectedLayer(in_channels=in_channels, out_channels=out_channels, activation="lrelu")
        self.out_fc = FullyConnectedLayer(in_channels=out_channels, out_channels=num_channels, activation="lrelu")
    
    def forward(self, x: Tensor, group_size: int=4) -> Tensor:
        x = mini_batch_std_dev(x, group_size)
        
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        
        x = self.fc(x.flatten(1))
        x = self.out_fc(x)
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
# Generator network.

class Generator(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,
        nf: int,
        kernel_size: int,
        stride: int,
        padding: int,
        depth: int,
        num_channels: int,
        scale_factor: float,
        start_size: int
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.nf = nf
        self.depth = depth
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        
        self.constant_input = ConstantInput(nf, start_size)
        
        # Base network layers
        self.layers = torch.nn.ModuleList([])
        
        n = self.nf
        for l in range(self.depth):
            if l == 0:
                self.scale_factor = 1.0
            else:
                self.scale_factor = scale_factor

            self.layers.append(StyleBlock(in_channels=n, out_channels=n//2, kernel_size=kernel_size, stride=stride, padding=padding, scale_factor=self.scale_factor))
            n = n // 2

        # Network converter layers.
        self.converters = torch.nn.ModuleList([])
        
        n = self.nf // 2
        for i in range(depth):
            self.converters.append(Conv1dLayer(in_channels=n, out_channels=num_channels, kernel_size=1, stride=1, padding=0, bias=False))
            n = n // 2

        # Mapping network.
        self.mapping_network = MappingNetwork(depth*2)

    def forward(self,
        latent_z: Tensor,
        latent_w: Tensor    = None,
        return_w: bool      = False
    ):  
        x = self.constant_input(batch_size=latent_z.size(0))
        latent_w = self.mapping_network(latent_z, truncation_psi=1, truncation_cutoff=None)
        
        # Style mixing.
        #latent_w = self._mixing_regularization(latent_z, latent_w, self.depth)
        
        for i in range(self.depth): 
            x = self.layers[i](x, latent_w[:, 2 * i : 2 * i + 2])
            skip = self.converters[i](x, gain=np.sqrt(0.5))

            if i == 0:
                out = skip
            else:
                out = resample(out, scale_factor=self.scale_factor)
                out = out + skip

        if return_w:
            return out, latent_w
        else:
            return out

# ------------------------------------------------------------
# Regularize latent mixture.

    def _mixing_regularization(self, latent_z, latent_w, depth):
        latent_z_2 = torch.randn(latent_z.shape).to(latent_z.device)
        latent_w_2 = self.mapping_network(latent_z_2)

        layer_idx = torch.from_numpy(np.arange(self.depth * 2)[np.newaxis, :, np.newaxis]).to(latent_z.device)
        cur_layers = 2 * (depth + 1)

        mixing_cutoff = random.randint(1, depth + 1) if random.random() < 0.9 else cur_layers
        latent_w = torch.where(layer_idx < mixing_cutoff, latent_w, latent_w_2)
        return latent_w

# ------------------------------------------------------------
# Discriminator network.

class Discriminator(torch.nn.Module):
    def __init__(self,
        nf,
        kernel_size,
        stride,
        padding,
        depth,
        num_channels,
        scale_factor: float,
        start_size,
    ):
        super().__init__()
    
        self.depth = depth

        # Main discriminator blocks.
        self.layers = torch.nn.ModuleList([])
        
        n = nf
        for l in range(depth - 1):
            self.layers.append(DicriminatorBlock(in_channels=n, out_channels=n*2, kernel_size=kernel_size, stride=stride, padding=padding, scale_factor=scale_factor))
            n = n * 2
        
        # Final discriminator block.
        self.layers.append(DiscriminatorEpilogue(in_channels=n, out_channels=n*2, num_channels=num_channels, kernel_size=kernel_size, stride=stride, padding=padding, scale_factor=1/start_size))
        
        # Layer used to convert sound into tensor for the network.
        self.converter = Conv1dLayer(in_channels=num_channels, out_channels=nf, kernel_size=1, stride=1, padding=0, bias=False)

        self.linear = FullyConnectedLayer(n + 1, num_channels)

    def forward(self, x):
        x = self.converter(x)

        for i in range(self.depth):
            x = self.layers[i](x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import random
import math
import time


# ------------------------------------------------------------
# Scripted functions.

@torch.jit.script
def pixel_norm(x: Tensor, epsilon: float = 1e-8) -> Tensor:
    return x / (x.pow(2.0).mean(dim = 1, keepdim = True).add(epsilon).sqrt())

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
    return F.conv1d(x, kernel, stride = 1, padding = 2, groups = x.size(1))

@torch.jit.script
def blur_down(x: Tensor, kernel: Tensor, scale_factor: float) -> Tensor:
    x = F.conv1d(x, kernel, stride = 1, padding = 2, groups = x.size(1))
    return F.interpolate(input = x, scale_factor = scale_factor, mode = "linear")

@torch.jit.script
def blur_up(x: Tensor, kernel: Tensor, scale_factor: float) -> Tensor:
    x = F.interpolate(input = x, scale_factor = scale_factor, mode = "linear")
    return F.conv1d(x, kernel, stride = 1, padding = 2, groups = x.size(1))

@torch.jit.script
def mini_batch_std_dev(x: Tensor, group_size: int, channels: int, samples: int, alpha: float = 1e-8) -> Tensor:
    y = torch.reshape(x, [group_size, -1, channels, samples])
    y = y - y.mean(dim=0, keepdim=True)
    y = torch.sqrt(y.square().mean(dim=0, keepdim=False) + alpha)
    y = y.mean(dim=[1, 2], keepdim=True)
    y = y.repeat(group_size, 1, samples)
    return torch.cat([x, y], 1)

# ------------------------------------------------------------
# Low level network components.

# ------------------------------------------------------------
# Convolution layer with equalized learning.

class Conv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
        apply_style: bool = False,
        apply_noise: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.apply_style = apply_style
        self.apply_noise = apply_noise

        if apply_style:
            self.style_affine = EqualizedLinear(512, in_channels, gain = 1)

        if apply_noise:
            self.noise_strength = nn.Parameter(torch.zeros([out_channels]))

        weight = torch.randn([out_channels, in_channels, kernel_size])
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        bias = torch.zeros([out_channels]) if bias else None
        self.bias = torch.nn.Parameter(bias) if bias is not None else None
    
    def forward(
        self,
        x: Tensor,
        style: Tensor = torch.tensor(0),
        gain: float = 1.0
    ) -> Tensor:
        weight = self.weight * self.weight_gain
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        bias = bias.reshape(1, -1, 1)

        if self.apply_style:
            style = self.style_affine(style)
            x, dcoefs = modulate(x, style, weight)

        x = F.conv1d(x, weight, stride = self.stride, padding = self.padding)

        if self.apply_style:
            x = demodulate(x, dcoefs)

        if self.apply_noise:
            noise = torch.randn(x.size(0), 1, x.size(2), device = x.device, dtype = x.dtype)

            x = x + noise * self.noise_strength.view(1, -1, 1)

        x = x + bias * gain
        return F.leaky_relu(x, 0.2 * gain)

# ------------------------------------------------------------
# Fully-connected layer with equalized learning.

class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gain: float = 2 ** 0.5,
        use_wscale: bool = True,
        lrmul: float = 1, 
        bias: bool = True
    ):
        super().__init__()

        he_std = gain * in_channels ** (-0.5)  # He init
        # Equalized learning rate and custom learning rate multiplier.
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return F.linear(x, self.weight * self.w_mul, bias)

# ------------------------------------------------------------
# Gaussian noise concatenation layer.

class ApplyNoise(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor, noise: Tensor = torch.tensor(0)) -> Tensor:
        if noise == 0:
            noise = torch.randn(x.size(0), 1, x.size(2), device = x.device, dtype = x.dtype)
        return x + self.weight.view(1, -1, 1) * noise

# ------------------------------------------------------------
# Style modulation layer.

class ApplyStyle(nn.Module):
    def __init__(
        self,
        conv,
        channels,
        latent_size = 512
    ):
        super(ApplyStyle, self).__init__()
        self.conv = conv
        self.lin = EqualizedLinear(latent_size, channels, gain = 1)

        self.noise_strength = nn.Parameter(torch.zeros(channels))
    
    def forward(
        self,
        x: Tensor,
        styles: Tensor,
        weight: Tensor,
        noise: Tensor = torch.tensor(0),
        gain: float = 1.0
    ) -> Tensor:
        styles = self.lin(styles)
        x, dcoefs = modulate(x, styles, weight)

        if noise == 0:
            noise = torch.randn(x.size(0), 1, x.size(2), device = x.device, dtype = x.dtype)
        
        noise = noise * self.noise_strength.view(1, -1, 1)
        return demodulate(self.conv(x, noise), dcoefs)

# ------------------------------------------------------------
# Resample layers.

class Resample(nn.Module):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction
        kernel = [1, 2, 4, 2, 1]
        kernel = torch.tensor(kernel, dtype = torch.float)
        kernel = kernel.expand(1, 1, -1)
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor, scale_factor: float) -> Tensor:
        kernel = self.kernel.expand(x.size(1), -1, -1)

        if self.direction == "up":
            return blur_up(x, kernel, scale_factor)
        else:
            return blur_down(x, kernel, scale_factor)

# ------------------------------------------------------------
# Minibatch standard deviation layer for discriminator. Used to increase diversity in generator.

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size = 4) -> None:
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"

    def forward(self, x: Tensor) -> Tensor:
        batch_size, channels, samples = x.shape
        if batch_size > self.group_size:
            assert batch_size % self.group_size == 0, (
                f"batch_size {batch_size} should be "
                f"perfectly divisible by group_size {self.group_size}"
            )
            group_size = self.group_size
        else:
            group_size = batch_size

        return mini_batch_std_dev(x, group_size, channels, samples)

# ------------------------------------------------------------
# Blocks composed of low level network components.

# ------------------------------------------------------------
# General generator block.

class GenGeneralConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale_factor: float,
        resample: Resample,
        bias = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.resample = resample
        
        self.conv_block_1 = Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, apply_style = True, apply_noise = True, bias = bias)
        self.conv_block_2 = Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, apply_style = True, apply_noise = True, bias = bias)
        self.conv_block_3 = Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)

    def forward(self, x: Tensor, latent_w: Tensor, noise: Tensor = torch.tensor(0)) -> Tensor:
        x = self.resample(x, scale_factor = self.scale_factor)

        x = self.conv_block_1(x, latent_w[:, 0])
        x = self.conv_block_2(x, latent_w[:, 1], gain = np.sqrt(0.5))
        x = self.conv_block_3(x)

        return x

# ------------------------------------------------------------
# General discriminator block.

class DisGeneralConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        scale_factor: float,
        resample,
        bias = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.resample = resample

        self.conv_block_1 = Conv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
        self.conv_block_2 = Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = bias)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_block_1(x)
        x = self.conv_block_2(x, gain = np.sqrt(0.5))
        return self.resample(x, scale_factor = self.scale_factor)

# ------------------------------------------------------------
# Final discriminator block.

class DisFinalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_channels,
        kernel_size,
        stride,
        padding,
        scale_factor: float,
        resample,
        bias = True
    ):
        super().__init__()
        self.resample = resample
        self.scale_factor = scale_factor
        
        in_channels += 1
        out_channels += 1

        self.mini_batch = MiniBatchStdDev()

        self.conv_block_1 = Conv1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )

        self.conv_block_2 = Conv1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )

        self.conv_block_3 = Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.mini_batch(x)
    
        x = self.conv_block_1(x)
        x = self.conv_block_2(x, gain = np.sqrt(0.5))
        x = self.conv_block_3(x)

        x = self.resample(x, scale_factor = self.scale_factor)
        return x

# ------------------------------------------------------------
# Constant input.

class ConstantInput(nn.Module):
    def __init__(self, nf: int, start_size: int):
        super().__init__()

        self.constant_input = nn.Parameter(torch.randn(1, nf, start_size))
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def forward(self, batch_size: int) -> Tensor:
        x = self.constant_input.expand(batch_size, -1, -1)
        x = x + self.bias.view(1, -1, 1)
        return x

# ------------------------------------------------------------
# Truncation module.

class Truncation(nn.Module):
    def __init__(self, avg_latent, max_layer=8, threshold=0.7, beta=0.995):
        super().__init__()
        self.max_layer = max_layer
        self.threshold = threshold
        self.beta = beta
        self.register_buffer('avg_latent', avg_latent)

    def update(self, last_avg):
        self.avg_latent.copy_(self.beta * self.avg_latent + (1. - self.beta) * last_avg)

    def forward(self, x):
        assert x.dim() == 3
        interp = torch.lerp(self.avg_latent, x, self.threshold)
        do_trunc = (torch.arange(x.size(1)) < self.max_layer).view(1, -1, 1).to(x.device)
        return torch.where(do_trunc, interp, x)

# ------------------------------------------------------------
# Mapping network.
# Maps latent space Z to W.

class MappingNetwork(nn.Module):
    def __init__(
        self,
        broadcast,
        depth = 8,
        z_dim = 512,
        lrmul = 0.01
    ):
        super().__init__()

        self.broadcast = broadcast

        # Fully connected layers.
        self.layers = nn.ModuleList([])
        for l in range(depth - 1):
            self.layers.append(EqualizedLinear(in_channels = z_dim, out_channels = z_dim, lrmul = lrmul))
        
        self.layers.append(EqualizedLinear(in_channels = z_dim, out_channels = z_dim, lrmul = lrmul))

    def forward(self, x: Tensor) -> Tensor:
        x = pixel_norm(x)

        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)

        return x.unsqueeze(1).expand(-1, self.broadcast, -1)

# ------------------------------------------------------------
# Generator network.

class Generator(nn.Module):
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
        self.truncation = Truncation(avg_latent = torch.zeros(z_dim))
        self.resample = Resample(direction = "up")
        
        # Base network layers
        self.layers = nn.ModuleList([])
        
        n = self.nf
        for l in range(self.depth):
            if l == 0:
                self.scale_factor = 1.0
            else:
                self.scale_factor = scale_factor

            self.layers.append(
                GenGeneralConvBlock(
                    in_channels = n,
                    out_channels = n // 2,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    scale_factor = self.scale_factor,
                    resample = self.resample
                )
            )
            n = n // 2

        # Network converter layers.
        self.converters = nn.ModuleList([])
        
        n = self.nf // 2
        for i in range(depth):
            self.converters.append(Conv1d(n, num_channels, 1, 1, 0))
            n = n // 2

        # Mapping network.
        self.mapping_network = MappingNetwork(broadcast = depth * 2)

# ------------------------------------------------------------
# Forward pass of the generator network.

    def forward(
        self,
        latent_z: Tensor,
        step: int = None,
        is_training: bool = True,
        latent_w: Tensor = torch.tensor(0),
        noise: Tensor = torch.tensor(0),
        return_w: bool = False
    ):  
        if is_training:
            x = self._train(latent_z, step, return_w)
            return x
        else:
            x = self._generate(latent_z)
            return x

# ------------------------------------------------------------
# Sample generator.

    def _generate(self, latent_z):
        pass

# ------------------------------------------------------------
# Network trainer.

    def _train(self, latent_z, step, return_w):
        batch_size = latent_z.size(0)

        x = self.constant_input(batch_size)
        latent_w = self.mapping_network(latent_z)
        
        # Style mixing and truncation.
        self.truncation.update(latent_w[0, 0].detach())
        latent_w = self._mixing_regularization(latent_z, latent_w, self.depth)
        latent_w = self.truncation(latent_w)
        
        i = 0
        for layer_block in self.layers[: self.depth]: 
            x = layer_block(x, latent_w[:, 2 * i : 2 * i + 2])
            skip = self.converters[i](x, gain = np.sqrt(0.5))

            if i == 0:
                out = skip
            else:
                out = self.resample(out, scale_factor = self.scale_factor)
                out = out + skip
            i += 1
            if(step == i):
                break
        
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

class Discriminator(nn.Module):
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
        
        self.nf = nf
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.depth = depth
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.start_size = start_size
        self.resample = Resample(direction = "down")

        # Main discriminator convolution blocks.
        self.layers = nn.ModuleList([])
        
        n = self.nf
        for l in range(depth - 1):
            self.layers.append(
                DisGeneralConvBlock(
                    in_channels = n,
                    out_channels = n * 2,
                    kernel_size = self.kernel_size,
                    stride = self.stride,
                    padding = self.padding,
                    scale_factor = self.scale_factor,
                    resample = self.resample
                )
            )
            n = n * 2
        
        # Final discriminator convolution block.
        self.layers.append(
            DisFinalConvBlock(
                in_channels = n,
                out_channels = n * 2,
                num_channels = self.num_channels,
                kernel_size = self.kernel_size,
                stride = self.stride,
                padding = self.padding,
                scale_factor = 1 / self.start_size,
                resample = self.resample
            )
        )
        
        # List of converters that broadcast channels from "num_channels" to "n".
        self.converters = nn.ModuleList([])
        self.res_converters = nn.ModuleList([])
        n = self.nf
        for l in range(self.depth):
            if l == self.depth - 1:
                self.converters.append(Conv1d(num_channels, n, 1, 1, 0))
                self.res_converters.append(Conv1d(n, n * 2 + 1, 1, 1, 0))
            else:
                self.converters.append(Conv1d(num_channels, n, 1, 1, 0))
                self.res_converters.append(Conv1d(n, n * 2, 1, 1, 0))
            n = n * 2

        self.linear = EqualizedLinear(n + 1, num_channels)

# ------------------------------------------------------------
# Forward pass of the discriminator network.

    def forward(self, x, step = None):
        if step == None:
            step = self.depth
        
        x = self.converters[self.depth - step](x)

        for i in range(self.depth):
            if self.depth - step <= i:
                if i < self.depth - 1:
                    residual = self.res_converters[i](self.resample(x, scale_factor = self.scale_factor), gain = np.sqrt(0.5))
                else:
                    residual = self.res_converters[i](self.resample(x, scale_factor = 1 / self.start_size), gain = np.sqrt(0.5))
                x = self.layers[i](x)
                x = (x + residual) * (1 / np.sqrt(2))
            
        x = self.linear(x.squeeze(2))

        return x

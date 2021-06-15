import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

# ------------------------------------------------------------
# Low level network components.

# ------------------------------------------------------------
# Convolution layer with equalized learning.

class EqualizedConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation = 1,
        bias = True
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            bias = bias
        )

        nn.init.normal_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias) 

        fan_in = np.prod(kernel_size) * in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)
    
    def forward(self, x):
        return self.conv(x * self.scale)

# ------------------------------------------------------------
# Fully-connected layer with equalized learning.

class EqualizedLinear(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gain = 2 ** 0.5,
        use_wscale = True,
        lrmul = 1, 
        bias = True
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

    def forward(self, x):
        bias = self.bias
        if bias is not None:
            bias = bias * self.b_mul
        return nn.functional.linear(x, self.weight * self.w_mul, bias)

# ------------------------------------------------------------
# Gaussian noise concatenation layer.

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise = None):
        if noise == None:
            noise = torch.randn(x.size(0), 1, x.size(2), device = x.device, dtype = x.dtype)
        x = x + self.weight.view(1, -1, 1) * noise
        return x

# ------------------------------------------------------------
# Style modulation layer.

class StyleMod(nn.Module):
    def __init__(self, channels, latent_size = 512):
        super(StyleMod, self).__init__()
        self.lin = EqualizedLinear(latent_size,
                                   channels * 2,
                                   gain=1.0)

    def forward(self, x, latent):
        style = self.lin(latent)  # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1)] + (x.dim() - 2) * [1]
        style = style.view(shape)  # [batch_size, 2, n_channels, ...]
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

# ------------------------------------------------------------
# Normalization layer.

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, epsilon: float = 1e-8):
        x = x / (x.pow(2.0).mean(dim = 1, keepdim = True).add(epsilon).sqrt())
        return x
    
# ------------------------------------------------------------
# Resample layers.

class Resample(nn.Module):
    def __init__(self, direction, scale_factor):
        super().__init__()
        self.direction = direction
        self.scale_factor = scale_factor

        kernel = torch.tensor([1, 2, 4, 16, 256, 16, 4, 2, 1], dtype = torch.float32, device = "cuda")
        self.kernel = kernel.expand(1, 1, -1)

    def _blur(self, x):
        kernel = self.kernel.expand(x.size(1), -1, -1)
        kernel = kernel / kernel.sum()
        x = F.conv1d(
            x,
            kernel,
            stride = 1,
            padding = 4,
            groups = x.size(1)
        )
        return x

    def forward(self, x):
        if self.direction == "up":
            x = F.interpolate(input = x, scale_factor = self.scale_factor, mode = "linear")
            x = self._blur(x)
        else:
            x = self._blur(x)
            x = F.interpolate(input = x, scale_factor = self.scale_factor, mode = "linear")
        return x

# ------------------------------------------------------------
# Minibatch standard deviation layer for discriminator. Used to increase diversity in generator.

class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size = 4) -> None:
        super(MiniBatchStdDev, self).__init__()
        self.group_size = group_size

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"

    def forward(self, x, alpha: float = 1e-8):
        batch_size, channels, samples = x.shape
        if batch_size > self.group_size:
            assert batch_size % self.group_size == 0, (
                f"batch_size {batch_size} should be "
                f"perfectly divisible by group_size {self.group_size}"
            )
            group_size = self.group_size
        else:
            group_size = batch_size

        y = torch.reshape(x, [group_size, -1, channels, samples])
        y = y - y.mean(dim=0, keepdim=True)
        y = torch.sqrt(y.square().mean(dim=0, keepdim=False) + alpha)
        y = y.mean(dim=[1, 2], keepdim=True)
        y = y.repeat(group_size, 1, samples)
        y = torch.cat([x, y], 1)
        return y

# ------------------------------------------------------------
# Blocks composed of low level network components.

# ------------------------------------------------------------
# Epilogue layer.

class LayerEpilogue(nn.Module):
    def __init__(
        self,
        channels
    ):
        super().__init__()

        self.instance_norm = nn.InstanceNorm1d(channels)
        self.apply_noise = ApplyNoise(channels)
        self.style_mod = StyleMod(channels)
    
    def forward(self, x, latent_w, noise = None):
        self.apply_noise(x, noise)
        x = F.leaky_relu(x, 0.2)
        x = self.instance_norm(x)
        x = self.style_mod(x, latent_w)
        
        return x

# ------------------------------------------------------------
# General generator block.

class GenGeneralConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        scale_factor,
        resample,
        bias = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.resample = resample
        
        self.conv_block_1 = EqualizedConv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
        self.conv_block_2 = EqualizedConv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)

        self.layer_epilogue_1 = LayerEpilogue(channels = in_channels)
        self.layer_epilogue_2 = LayerEpilogue(channels = out_channels)
    
    def forward(self, x, latent_w, noise = None):
        x = self.resample(x)

        # First block
        x = self.conv_block_1(x)
        x = self.layer_epilogue_1(x, latent_w[:, 0], noise)

        # Second block
        x = self.conv_block_2(x)
        x = self.layer_epilogue_2(x, latent_w[:, 1], noise)
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
        dilation,
        scale_factor,
        resample,
        bias = True
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.resample = resample

        self.conv_block_1 = EqualizedConv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
        self.conv_block_2 = EqualizedConv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
    
    def forward(self, x):
        x = self.conv_block_1(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv_block_2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.resample(x, scale_factor = 1 / self.scale_factor, direction = "down")

        return x

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
        dilation,
        scale_factor,
        resample,
        bias = True
    ):
        super().__init__()
        self.resample = resample
        
        in_channels += 1
        out_channels += 1

        self.mini_batch = MiniBatchStdDev()

        self.conv_block_1 = EqualizedConv1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            bias = bias
        )

        self.conv_block_2 = EqualizedConv1d(
            in_channels = in_channels,
            out_channels = in_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            bias = bias
        )

        self.conv_block_3 = EqualizedConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = bias
        )
    
    def forward(self, x):
        x = self.mini_batch(x)

        x = F.leaky_relu(x, 0.2)
        x = self.conv_block_1(x)

        x = F.leaky_relu(x, 0.2)
        x = self.conv_block_2(x)
    
        x = self.conv_block_3(x)

        x = self.resample()

        return x

# ------------------------------------------------------------
# Constant input.

class ConstantInput(nn.Module):
    def __init__(self, nf, start_size):
        super().__init__()

        self.constant_input = nn.Parameter(torch.randn(1, nf, start_size))
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def forward(self, batch_size):
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

        # Normalization layer.
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        x = self.pixel_norm(x)

        for layer in self.layers:
            x = layer(x)
            x = F.leaky_relu(x, 0.2)

        return x.unsqueeze(1).expand(-1, self.broadcast, -1)
            
        return x

# ------------------------------------------------------------
# Generator network.

class Generator(nn.Module):
    def __init__(
        self,
        z_dim,
        nf,
        kernel_size,
        stride,
        padding,
        dilation,
        depth,
        num_channels,
        scale_factor,
        start_size
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.nf = nf
        self.depth = depth
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        
        self.constant_input = ConstantInput(nf, start_size)
        self.truncation = Truncation(avg_latent = torch.zeros(z_dim))
        self.resample = Resample(direction = "up", scale_factor = scale_factor)
        
        # Base network layers
        self.layers = nn.ModuleList([])
        
        n = self.nf
        for l in range(self.depth):
            if l == 0:
                self.scale_factor = 1
            else:
                self.scale_factor = scale_factor

            self.layers.append(
                GenGeneralConvBlock(
                    in_channels = n,
                    out_channels = n // 2,
                    kernel_size = kernel_size,
                    stride = stride,
                    padding = padding,
                    dilation = dilation,
                    scale_factor = self.scale_factor,
                    resample = self.resample
                )
            )
            n = n // 2

        # Network converter layers.
        self.converters = nn.ModuleList([])
        
        n = self.nf // 2
        for i in range(depth):
            self.converters.append(EqualizedConv1d(n, num_channels, 1, 1, 0))
            n = n // 2

        # Mapping network.
        self.mapping_network = MappingNetwork(broadcast = depth * 2)
     
    def forward(
        self,
        latent_z,
        is_training = True,
        latent_w = None,
        noise = None
    ):     
        if is_training:
            x = self._train(latent_z)
            return x
        else:
            x = self._generate(latent_z)
            return x
        
    def _generate(self, latent_z):
        pass

    def _train(self, latent_z):
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
            skip = self.converters[i](x)

            if i == 0:
                out = skip
            else:
                print(out.shape)
                out = self.resample(x)
                print(out.shape)
                print(skip.shape)
                quit()
                out = out + skip
            i += 1

        return out

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
        dilation,
        depth,
        num_channels,
        scale_factor,
        start_size,
    ):
        super().__init__()
        
        self.nf = nf
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.depth = depth
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.start_size = start_size
        self.resample = Resample(scale_factor = scale_factor, direction = "down")

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
                    dilation = self.dilation,
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
                dilation = self.dilation,
                scale_factor = self.start_size,
                resample = self.resample
            )
        )
        
        # List of converters that broadcast channels from "num_channels" to "n".
        self.converters = EqualizedConv1d(num_channels, self.nf, 1, 1, 0)

        self.res_converters = nn.ModuleList([])
        n = self.nf
        for l in range(self.depth):
            if l == self.depth - 1:
                self.res_converters.append(EqualizedConv1d(n, n * 2 + 1, 1, 1, 0))
            else:
                self.res_converters.append(EqualizedConv1d(n, n * 2, 1, 1, 0))
            n = n * 2

        self.linear = EqualizedLinear(n + 1, num_channels)


    def forward(self, x):
        x = self.converters(x)

        for i in range(self.depth):
            if i < self.depth - 1:
                residual = self.res_converters[i](self.resampel(x))
            else:
                residual = self.res_converters[i](self.resample(x))
            x = self.layers[i](x)
            x = (x + residual) * (1 / math.sqrt(2))
        
        x = self.linear(x.squeeze(2))

        return x

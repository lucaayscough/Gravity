import torch
import torch.nn as nn
from torch.nn.functional import interpolate
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

    def forward(self, x, noise):
        if noise == torch.tensor(0):
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
# Self-attention layer.

class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.query = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.key = nn.Conv1d(in_channel, in_channel // 8, 1)
        self.value = nn.Conv1d(in_channel, in_channel, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        shape = x.shape
        x_copy = x
        query = self.query(x_copy).permute(0, 2, 1)
        key = self.key(x_copy)
        value = self.value(x_copy)
        query_key = torch.bmm(query, key)
        attn = nn.functional.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        out = self.gamma * attn + x
        return out

# ------------------------------------------------------------
# Blocks composed of low level network components.

# ------------------------------------------------------------
# Epilogue layer.

class LayerEpilogue(nn.Module):
    def __init__(
        self,
        use_pixel_norm,
        use_noise,
        use_instance_norm,
        use_style,
        channels
    ):
        super().__init__()

        self.use_instance_norm = use_instance_norm
        self.use_pixel_norm = use_pixel_norm
        self.use_noise = use_noise
        self.use_style = use_style

        if use_instance_norm:
            self.instance_norm = nn.InstanceNorm1d(channels)

        if use_noise:
            self.apply_noise = ApplyNoise(channels)

        if use_style:
            self.style_mod = StyleMod(channels)
        
        self.l_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x, latent_w, noise):
        if self.use_noise:
            self.apply_noise(x, noise)
        
        x = self.l_relu(x)

        if self.use_instance_norm:
            x = self.instance_norm(x)

        if self.use_style:
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
        use_pixel_norm,
        use_instance_norm,
        use_style,
        use_noise,
        use_linear_upsampling,
        use_self_attention,
        bias = True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_linear_upsampling = use_linear_upsampling
        self.use_self_attention = use_self_attention
        
        self.conv_block_1 = EqualizedConv1d(in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
        self.conv_block_2 = EqualizedConv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, dilation = dilation, bias = bias)
        
        if self.use_linear_upsampling:
            self.linear_upsampling = nn.Upsample(scale_factor = self.scale_factor, mode = 'linear')

        self.layer_epilogue_1 = LayerEpilogue(
            use_pixel_norm = use_pixel_norm,
            use_noise = use_noise,
            use_instance_norm = use_instance_norm,
            use_style = use_style,
            channels = in_channels,
        )

        self.layer_epilogue_2 = LayerEpilogue(
            use_pixel_norm = use_pixel_norm,
            use_noise = use_noise,
            use_instance_norm = use_instance_norm,
            use_style = use_style,
            channels = out_channels,
        )
                
        if self.use_self_attention:
            self.self_attention = SelfAttention(in_channels)
    
    def forward(self, x, latent_w, noise):
        x = interpolate(x, scale_factor = self.scale_factor)

        # First block
        x = self.conv_block_1(x)
        x = self.layer_epilogue_1(x, latent_w[:, 0], noise)
        
        # Second block
        x = self.conv_block_2(x)
        x = self.layer_epilogue_2(x, latent_w[:, 1], noise)
        return x

# ------------------------------------------------------------
# Constant input.

class ConstantInput(nn.Module):
    def __init__(self, nf, start_size):
        super().__init__()

        self.constant_input = nn.Parameter(torch.randn(1, nf, start_size))
        self.bias = nn.Parameter(torch.zeros(nf))
    
    def forward(self, batch_size: int):
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
        use_pixel_norm = True,
        use_leaky_relu = True,
        lrmul = 0.01,
    ):
        super().__init__()
        
        self.use_pixel_norm = use_pixel_norm
        self.use_leaky_relu = use_leaky_relu
        self.broadcast = broadcast

        # Fully connected layers.
        self.layers = nn.ModuleList([])
        for l in range(depth - 1):
            self.layers.append(EqualizedLinear(in_channels = z_dim, out_channels = z_dim, lrmul = lrmul))
        
        self.layers.append(EqualizedLinear(in_channels = z_dim, out_channels = z_dim, lrmul = lrmul))

        # Normalization layer.
        if self.use_pixel_norm:
            self.pixel_norm = PixelNorm()
        
        # Non-linearity layer.
        if self.use_leaky_relu:
            self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        if self.use_pixel_norm:
            x = self.pixel_norm(x)

        for layer in self.layers:
            x = layer(x)
            if self.use_leaky_relu:
                x = self.leaky_relu(x)

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
        use_linear_upsampling,
        use_self_attention,
        use_pixel_norm,
        use_instance_norm,
        use_style,
        use_noise,
        start_size
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.nf = nf
        self.depth = depth
        self.num_channels = num_channels
        self.scale_factor = scale_factor
        self.use_linear_upsampling = use_linear_upsampling
        
        self.constant_input = ConstantInput(nf, start_size)

        self.truncation = Truncation(avg_latent = torch.zeros(z_dim))
        
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
                    scale_factor = float(self.scale_factor),
                    use_pixel_norm = use_pixel_norm,
                    use_instance_norm = use_instance_norm,
                    use_style = use_style,
                    use_noise = use_noise,
                    use_linear_upsampling = use_linear_upsampling,
                    use_self_attention = use_self_attention
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
        noise = torch.tensor(0) 
    ):     
        batch_size: int  = int(latent_z.size(0))

        x = self.constant_input(batch_size)
        
        latent_w = self.mapping_network(latent_z)
        latent_w = self.truncation(latent_w)
        
        # Layer 1
        x = self.layers[0](x, latent_w[:, 0:2], noise)
        skip = self.converters[0](x)
        out = skip
                 
        # Layer 2
        x = self.layers[1](x, latent_w[:, 2:4], noise) 
        skip = self.converters[1](x)
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        out = out + skip
        
        # Layer 3
        x = self.layers[2](x, latent_w[:, 4:6], noise) 
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        skip = self.converters[2](x)
        out = out + skip
        
        # Layer 4
        x = self.layers[3](x, latent_w[:, 6:8], noise)
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        skip = self.converters[3](x)
        out = out + skip
        
        # Layer 5
        x = self.layers[4](x, latent_w[:, 8:10], noise)
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        skip = self.converters[4](x)
        out = out + skip
        
        # Layer 6
        x = self.layers[5](x, latent_w[:, 10:12], noise)
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        skip = self.converters[5](x)
        out = out + skip
        
        # Layer 7
        x = self.layers[6](x, latent_w[:, 12:14], noise)
        out = F.upsample(out, scale_factor = float(self.scale_factor), mode = "linear")
        skip = self.converters[6](x)
        out = out + skip
         
        return out

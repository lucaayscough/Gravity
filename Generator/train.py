import os
import time
import copy
import math
import numpy as np

import torch
from torch import Tensor
import torchaudio
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

from model import Generator, Discriminator
from misc.utils import AudioFolder


class Train:
    def __init__(self,
        # Iteration
        restart: bool,
        iter_num: int,
        epochs: int,
        datadir: str,
        
        # Training
        batch_size: int,
        g_loss: str,
        d_loss: str,

        # Learning
        learning_rate: float,

        # Model        
        scale_factor: int,  
        depth: int,
        num_filters: int,
        start_size: int,
     
        # Setup
        num_channels: int,
        sample_rate: int,
        save_every: int,
        num_workers: int,
        device: str
    ):
        # Iteration
        self.restart = restart
        self.iter_num = iter_num
        self.epochs = epochs
        self.datadir = datadir
        
        # Training
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.g_loss = g_loss
        self.d_loss = d_loss

        # Generator
        self.ema_beta = 0.999
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_weight = 2

        # Discriminator
        self.r1_gamma = 10

        # Model
        self.G_reg = 8
        self.D_reg = 16
        self.z_dim = 512
        self.scale_factor = scale_factor
        self.depth = depth
        self.num_filters = num_filters
        self.start_size = start_size

        # Setup
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.save_every = save_every
        self.num_workers = num_workers
        self.device = device

        self.window = torch.hann_window(window_length=1024, device=self.device)
        
        # Initialization
        torch.backends.cudnn.benchmark = True
        #torch.autograd.set_detect_anomaly(True)
        
        self._init_models()
        self._init_optim()

        if self.restart == True:
            self._load_state()
        else:
            self.start_epoch = 1
            self.pl_mean = torch.tensor(0, dtype=torch.float, device=self.device).to(dtype=torch.float32)
        
        # Creates a shadow copy of the generator.
        self.netG_shadow = copy.deepcopy(self.netG)

        # Initialize the generator shadow weights equal to generator.
        self._update_average(beta = 0)

        self.dataset = AudioFolder(self.datadir)
        self._training_loop()


# ------------------------------------------------------------
# Initialization functions.

    def _init_models(self):
        self.netG = Generator(
            nf = self.num_filters,
            depth = self.depth,
            num_channels = self.num_channels,
            scale_factor = self.scale_factor,
            start_size = self.start_size
        ).to(self.device)
        
        self.netD = Discriminator(
            nf = self.num_filters / 2 ** self.depth,
            depth = self.depth,
            num_channels = self.num_channels,
            scale_factor = self.scale_factor,
            start_size = self.start_size,
        ).to(self.device)

    def _init_optim(self): 
        mb_ratio_g = self.G_reg / (self.G_reg + 1)
        mb_ratio_d = self.D_reg / (self.D_reg + 1)

        self.opt_gen = torch.optim.Adam(
            params = self.netG.parameters(),
            lr = self.learning_rate * mb_ratio_g,
            betas = (0.0 ** mb_ratio_g, 0.99 ** mb_ratio_g),
            eps = 1e-8
        )
        
        self.opt_dis = torch.optim.Adam(
            params = self.netD.parameters(),
            lr = self.learning_rate * mb_ratio_d, 
            betas = (0.0 ** mb_ratio_d, 0.99 ** mb_ratio_d),
            eps = 1e-8
        )

# ------------------------------------------------------------
# Model state functions.

    def _load_state(self):
        self._get_checkpoint()
        checkpointG = torch.load(
            'runs/iter_{iter_num}/model/checkpoint_{epoch}_G.pth.tar'.format(
                iter_num = self.iter_num,
                epoch = str(self.start_epoch).zfill(3)
            )
        )
        checkpointD = torch.load(
            'runs/iter_{iter_num}/model/checkpoint_{epoch}_D.pth.tar'.format(
                iter_num = self.iter_num,
                epoch = str(self.start_epoch).zfill(3)
            )
        )

        self.opt_gen.load_state_dict(checkpointG['optimizer'])
        self.opt_dis.load_state_dict(checkpointD['optimizer'])

        self.netG.load_state_dict(checkpointG['state_dict'])
        self.netD.load_state_dict(checkpointD['state_dict'])

        self.pl_mean = checkpointG['mean_pl']

    def _save_state(self, epoch): 
        checkpointD = {'state_dict': self.netD.state_dict(), 'optimizer': self.opt_dis.state_dict()}
        checkpointG = {'state_dict': self.netG.state_dict(), 'optimizer': self.opt_gen.state_dict(), "mean_pl": self.pl_mean}    
        
        torch.save(checkpointD,
            'runs/iter_{iter_num}/model/checkpoint_{epoch}_D.pth.tar'.format(
                iter_num = self.iter_num,
                epoch = str(epoch).zfill(3)
            )
        )        
        torch.save(checkpointG,
            'runs/iter_{iter_num}/model/checkpoint_{epoch}_G.pth.tar'.format(
                iter_num = self.iter_num,
                epoch = str(epoch).zfill(3)
            )
        )
        
    def _get_checkpoint(self):
        models = os.listdir('runs/iter_{}/model/'.format(self.iter_num))

        start = 0
        for model in models:
            if int(model.split("_")[1]) > start:
                start = int(model.split("_")[1])

        self.start_epoch = start

# ------------------------------------------------------------
# Model training methods.

# ------------------------------------------------------------
# Main training loop.

    def _training_loop(self):
        dataloader = DataLoader(
            self.dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers,
            drop_last = True,
            pin_memory = True
        )

        self.netG.requires_grad_(False)
        self.netD.requires_grad_(False)

        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()

            # Save State
            if epoch % self.save_every == 0:
                self._save_state(epoch)

            for idx, data in enumerate(dataloader):
                # Fetch real data.
                real = data[0].to(self.device)

                # Tranform data.
                real = real.reshape(self.batch_size, -1)
                real = torch.stft(real, n_fft=1024, hop_length=512, win_length=1024, window=self.window)
                real = torch.view_as_complex(real)
                mag = real.abs()
                ph = real.angle()
                real = torch.stack((mag, ph), dim=1)

                # Train generator.
                if self.g_loss == "ns":
                    self._train_generator_ns(idx)
                elif self.g_loss == "wgan":
                    self._train_generator_wgan()

                # Train discriminator.
                if self.d_loss == "r1":
                    self._train_discriminator_r1(real, idx)
                elif self.d_loss == "wgangp":
                    self._train_discriminator_wgangp(real)

                self._update_average(beta = self.ema_beta)

                print(
                    f'Epoch [{epoch}/{self.epochs}] \
                    Batch {idx} / {len(dataloader)} \
                    Loss D: {self.loss_disc:8f}, loss G: {self.loss_gen:8f}'
                )

                self._print_examples(idx, epoch)
                    
            print("Time elapsed: ", time.time() - start_time, " seconds.")

# ------------------------------------------------------------
# Generator training methods.

# ------------------------------------------------------------
# WGAN loss.

    def _train_generator_wgan(self):
        self.netG.requires_grad_(True)

        noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
        fake = self.netG(noise)
        output = self.netD(fake).reshape(-1)
        
        self.loss_gen = -torch.mean(output)
        self.loss_gen.backward()
        
        self.netG.requires_grad_(False)
        self.opt_gen.step()
        self.opt_gen.zero_grad(set_to_none=True)

# ------------------------------------------------------------
# Non-Saturating logistic loss with path lenght regularization.

    def _train_generator_ns(self, idx: int, G_reg: int=8):
        # Non-Saturating loss.
        self.netG.requires_grad_(True)

        noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
        fake = self.netG(noise)
        output = self.netD(fake)
        self.loss_gen = torch.nn.functional.softplus(-output).mean()

        self.loss_gen.backward()
        self.netG.requires_grad_(False)
        self.opt_gen.step()
        self.opt_gen.zero_grad(set_to_none=True)
        
        # Path-length regularization.
        if idx % G_reg == 0:
            self.netG.requires_grad_(True)

            batch_size = noise.shape[0] // self.pl_batch_shrink
            sounds, w_latents = self.netG(noise[:batch_size], return_w=True)

            pl_noise = torch.randn_like(sounds, device=self.device) / np.sqrt(sounds.shape[2] * sounds.shape[3])
            pl_grads = torch.autograd.grad(outputs=[(sounds*pl_noise).sum()], inputs=[w_latents], create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()

            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            (sounds[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(G_reg).backward()
            
            self.netG.requires_grad_(False)
            self.opt_gen.step()
            self.opt_gen.zero_grad(set_to_none=True)

# ------------------------------------------------------------
# Discriminator training methods.

# ------------------------------------------------------------
# Logistic loss with r1 regularization.

    def _train_discriminator_r1(self, real, idx):
        # Train on generated.
        self.netD.requires_grad_(True)

        noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
        gen_sounds = self.netG(noise)
        logits_fake = self.netD(gen_sounds)
        loss_fake = torch.nn.functional.softplus(logits_fake).mean()

        loss_fake.backward()
        self.netD.requires_grad_(False)
        self.opt_dis.step()
        self.opt_dis.zero_grad(set_to_none=True)

        # Train on real.
        self.netD.requires_grad_(True)
        real_tmp = real.detach().requires_grad_(idx % self.D_reg == 0)
        logits_real = self.netD(real_tmp)
        loss_real = torch.nn.functional.softplus(-logits_real).mean()

        # R1.
        if idx % self.D_reg == 0:
            r1_grads = torch.autograd.grad(outputs=loss_real.sum(), inputs=real_tmp, create_graph=True, only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1,2])
            loss_r1 = r1_penalty * (self.r1_gamma / 2)
            (logits_real * 0 + loss_real + loss_r1).mean().mul(self.D_reg).backward()
        else:
            (logits_real * 0 + loss_real).mean().backward()

        self.netD.requires_grad_(False)
        self.opt_dis.step()
        self.opt_dis.zero_grad(set_to_none=True)
        
        self.loss_disc = loss_real + loss_fake

# ------------------------------------------------------------
# WGAN-GP loss.
    
    def _train_discriminator_wgangp(self, real, gamma=10):
        self.netD.requires_grad_(True)

        noise = torch.randn((self.batch_size, self.z_dim), device=self.device)
        fake = self.netG(noise)
                    
        disc_real = self.netD(real)
        disc_fake = self.netD(fake)

        # Calculate gradient penalty.
        batch_size, channels, kf, kt = real.shape
        epsilon = torch.rand((batch_size, 1, 1, 1), device=self.device).repeat(1, channels, kf, kt)
        epsilon = (epsilon - 0.5) * 2

        real.requires_grad = True
        
        interpolated_sounds = real * epsilon + fake * (1 - epsilon)
        mixed_scores = self.netD(interpolated_sounds)

        gradient = torch.autograd.grad(
            inputs = interpolated_sounds,
            outputs = mixed_scores,
            grad_outputs = torch.ones_like(mixed_scores),
            create_graph = True,
            retain_graph = True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

        # Backpropagate and optimize.
        self.loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + gamma * gradient_penalty
        self.loss_disc.backward()

        self.netD.requires_grad_(False)
        self.opt_dis.step()
        self.opt_dis.zero_grad(set_to_none=True)

# ------------------------------------------------------------
# Helper functions.

    def _print_examples(self, idx, epoch): 
        if idx % 200 == 0:
            
            # TODO:
            # REMOVE THIS
            self._save_state(epoch)
            
            with torch.no_grad():
                fake_sample = self.netG(
                    torch.randn((8, self.z_dim), device=self.device),
                    self.depth
                )

                fake_sample = torch.swapaxes(fake_sample, 0, 1)
                mag = fake_sample[0]
                ph = fake_sample[1]
                fake_sample = mag * torch.exp(1.j * ph)
                fake_sample = torch.istft(fake_sample, n_fft=1024, hop_length=512, win_length=1024, window=self.window)
                fake_sample = fake_sample.reshape(fake_sample.size(0), 1, -1)

                for s in range(8):
                    # Print Fake Examples
                    torchaudio.save(
                        filepath = 'runs/iter_' + str(self.iter_num) + '/output/' + '_' + str(epoch).zfill(3) + '_' + str(s).zfill(3) + '.wav',
                        src = fake_sample[s].cpu(),
                        sample_rate = self.sample_rate
                    )
  
# ------------------------------------------------------------
# Update exponential moving average of generator.

    def _update_average(self, beta):
        self._set_grad_flag(self.netG_shadow, False)
        self._set_grad_flag(self.netG, False)

        param_dict_src = dict(self.netG.named_parameters())

        for p_name, p_tgt in self.netG_shadow.named_parameters():
            p_src = param_dict_src[p_name]
            assert (p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)

        # turn back on the gradient calculation
        self._set_grad_flag(self.netG_shadow, True)
        self._set_grad_flag(self.netG, True)
    
# ------------------------------------------------------------
# Switch grad caculations on and off.

    def _set_grad_flag(self, module, flag):
        for p in module.parameters():
            p.requires_grad = flag

    def _fast_zero_grad(self, model):
        for param in model.parameters():
            param.grad = None
import os
import time
import copy
import math
import numpy as np

import torch
import torchaudio
from torch.utils.data import DataLoader
import torch.autograd.profiler as profiler

from model import Generator, Discriminator
from utils import AudioFolder, gradient_penalty


class Train:
    def __init__(
        self,

        # Iteration
        restart: bool,
        iter_num: int,
        epochs: int,
        datadir: str,
        
        # Training
        batch_size: int,

        # Learning
        learning_rate,

        # Model        
        scale_factor,       
        depth,
        num_filters,
        start_size,
     
        # Setup
        num_channels,
        sample_rate,
        save_every,
        num_workers,
        device
    ):
        # Iteration
        self.restart = restart
        self.iter_num = iter_num
        self.epochs = epochs
        self.datadir = datadir
        
        # Training
        self.batch_size = batch_size
        
        # Learning
        self.learning_rate = learning_rate

        # Generator
        self.ema_beta = 0.999
        self.pl_batch_shrink = 2
        self.pl_decay = 0.01
        self.pl_weight = 2

        # Discriminator
        self.disc_loss_fn = "r1"
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
        
        # Initialization
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)
        
        self._init_models()
        self._init_optim()

        if self.restart == True:
            self._load_state()
        else:
            self.start_epoch = 1
            self.pl_mean = torch.tensor(0, dtype=torch.float, device=self.device)
        
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
            nf = self.num_filters // (2 ** self.depth),
            depth = self.depth,
            num_channels = self.num_channels,
            scale_factor = self.scale_factor,
            start_size = self.start_size,
        ).to(self.device)

    def _init_optim(self): 
        self.opt_gen = torch.optim.Adam(
            params = self.netG.parameters(),
            lr = self.learning_rate,
            betas = (0.0, 0.99),
            eps = 1e-8
        )
        
        self.opt_dis = torch.optim.Adam(
            params = self.netD.parameters(),
            lr = self.learning_rate, 
            betas = (0.0, 0.99),
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
# Model training functions.

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

        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()

            # Save State
            if epoch % self.save_every == 0:
                self._save_state(epoch)

            for idx, data in enumerate(dataloader):
                real = data[0].to(self.device)

                self._train_generator(idx)
                if self.disc_loss_fn == "r1":
                    self._train_discriminator_r1(real, idx)
                else:
                    self._train_discriminator_wgangp(real)

                self._update_average(beta = self.ema_beta)
                
                print(
                    f'Epoch [{epoch}/{self.epochs}] \
                    Batch {idx} / {len(dataloader)} \
                    Loss D: {self.loss_disc:.4f}, loss G: {self.loss_gen:.4f}'
                )

                self._print_examples(idx, epoch)
                    
            print("Time elapsed: ", time.time() - start_time, " seconds.")

    def _train_generator(self, idx):
        self.netG.requires_grad_(True)
        self.netD.requires_grad_(False)

        self.opt_gen.zero_grad(set_to_none=True)

        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        fake = self.netG(noise)
        output = self.netD(fake)
        self.loss_gen = torch.nn.functional.softplus(-output).mean()

        self.loss_gen.backward()
        self.opt_gen.step()

        if idx % self.G_reg == 0:
            self.opt_gen.zero_grad(set_to_none=True)

            batch_size = noise.shape[0] // self.pl_batch_shrink
            sounds, w_latents = self.netG(noise[:batch_size], return_w=True)

            pl_noise = torch.randn_like(sounds) / np.sqrt(sounds.shape[2])
            pl_grads = torch.autograd.grad(outputs=[(sounds*pl_noise).sum()], inputs=[w_latents], create_graph=True, only_inputs=True)[0]
            pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()

            pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
            self.pl_mean.copy_(pl_mean.detach())

            pl_penalty = (pl_lengths - pl_mean).square()
            loss_Gpl = pl_penalty * self.pl_weight

            (sounds[:, 0, 0] * 0 + loss_Gpl).mean().mul(self.G_reg).backward()
            self.opt_gen.step()
        
        

    def _train_discriminator_r1(self, real, idx):
        self.netG.requires_grad_(False)
        self.netD.requires_grad_(True)

        self.opt_dis.zero_grad(set_to_none=True)
        
        # Train on generated.
        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        gen_sounds = self.netG(noise)
        logits_fake = self.netD(gen_sounds)
        loss_fake = torch.nn.functional.softplus(logits_fake).mean()

        loss_fake.backward()
        self.opt_dis.step()

        # Train on real.
        self.opt_dis.zero_grad(set_to_none=True)

        logits_real = self.netD(real)
        loss_real = torch.nn.functional.softplus(-logits_real).mean()

        # R1.
        if idx % self.D_reg == 0:
            real.requires_grad = True
            r1_grads = torch.autograd.grad(outputs=loss_real.sum(), inputs=real, create_graph=True, only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1,2])
            loss_r1 = r1_penalty * (self.r1_gamma / 2)
            (logits_real * 0 + loss_real + loss_r1).mean().mul(self.D_reg).backward()
        else:
            (logits_real * 0 + loss_real).mean().backward()

        self.loss_disc = loss_real + loss_fake
        self.opt_dis.step()
    
    def _train_discriminator_wgangp(self, real):
        self.netG.requires_grad_(False)
        self.netD.requires_grad_(True)
    
        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        fake = self.netG(noise)
                    
        disc_real = self.netD(real)
        disc_fake = self.netD(fake)

        gp = gradient_penalty(self.netD, real, fake, device=self.device)

        self.loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + 10 * gp

        self.loss_disc.backward(retain_graph=True)

# ------------------------------------------------------------
# Helper functions.

    def _print_examples(self, idx, epoch): 
        if idx % 1000 == 0:
            
            # TODO:
            # REMOVE THIS
            self._save_state(epoch)
            
            with torch.no_grad():
                fake_sample = self.netG(
                    torch.randn((8, self.z_dim)).to(self.device),
                    self.depth
                )
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
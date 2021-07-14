
def _train_discriminator_r1(self, real, idx):
    self._set_grad_flag(self.netD, True)
    self._set_grad_flag(self.netG, False)
    
    self.netD.zero_grad()

    real.requires_grad = True

    noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
    fake = self.netG(noise)

    disc_real = self.netD(real)
    disc_fake = self.netD(fake)

    disc_real = F.softplus(-disc_real).mean()
    disc_fake = F.softplus(disc_fake).mean()

    self.loss_disc = disc_real + disc_fake
    self.loss_disc.backward(retain_graph = True)

    # R1.
    if idx % 16 == 0:
        self.netD.zero_grad()

        grad_real = torch.autograd.grad(outputs = disc_real.sum(), inputs = real, create_graph = True)[0]
        grad_penalty_real = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty_real = 10 / 2 * grad_penalty_real

        grad_penalty_real.backward()
    
    self.opt_dis.step()

def _train_generator_ns(self, idx):
    self.netG.zero_grad()

    self._set_grad_flag(self.netD, False)
    self._set_grad_flag(self.netG, True)

    noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
    fake = self.netG(noise)
    output = self.netD(fake)
    self.loss_gen = F.softplus(-output).mean()
    self.loss_gen.backward()

    if idx % 8 == 0:
        self.netG.zero_grad()

        path_batch_size = max(1, self.batch_size // 2)
        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        fake, latents = self.netG(noise, return_w = True)

        noise = torch.randn_like(fake) / math.sqrt(fake.shape[2])
        grad, = torch.autograd.grad(outputs = (fake * noise).sum(), inputs = latents, create_graph = True)

        path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
        path_mean = self.mean_path_length + 0.01 * (path_lengths.mean() - self.mean_path_length)
        path_loss = (path_lengths - path_mean).pow(2).mean()
        self.mean_path_length = path_mean.detach()

        weighted_path_loss = 2 * 8 * path_loss
        weighted_path_loss += 0 * fake[0, 0, 0]
        weighted_path_loss.backward()

    self.opt_gen.step()

##############################

def _train_discriminator_wgangp(self, real):
    noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
    fake = self.netG(noise)
                
    disc_real = self.netD(real)
    disc_fake = self.netD(fake)
    
    gp = gradient_penalty(self.netD, real, fake, device = self.device)

    self.loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + 10 * gp

    self.netD.zero_grad()
    self.loss_disc.backward(retain_graph = True)
    self.opt_dis.step()

def _train_generator_wgangp(self): 
    noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
    fake = self.netG(noise)
    output = self.netD(fake)
    
    self.loss_gen = -torch.mean(output)
    
    self.netG.zero_grad()
    self.loss_gen.backward()
    self.opt_gen.step()















































    def _train_discriminator(self, real, idx):
        self._set_grad_flag(self.netD, True)
        self._set_grad_flag(self.netG, False)
        
        self._fast_zero_grad(self.netD)
        
        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        fake = self.netG(noise)
                    
        disc_real = self.netD(real)
        disc_fake = self.netD(fake)
        
        gp = gradient_penalty(self.netD, real, fake, device = self.device)

        self.loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + 10 * gp

        self.loss_disc.backward(retain_graph = True)
        self.opt_dis.step()

    def _train_generator(self, idx):
        self._set_grad_flag(self.netD, False)
        self._set_grad_flag(self.netG, True)
        
        self._fast_zero_grad(self.netG)
    
        noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
        fake = self.netG(noise)
        output = self.netD(fake)
        
        self.loss_gen = -torch.mean(output)
        self.loss_gen.backward()
        
        if idx % 16 == 0:
            self._fast_zero_grad(self.netG)

            path_batch_size = max(1, self.batch_size // 2)
            noise = torch.randn((self.batch_size, self.z_dim)).to(self.device)
            fake, latents = self.netG(noise, return_w = True)

            noise = torch.randn_like(fake) / math.sqrt(fake.shape[2])
            grad, = torch.autograd.grad(outputs = (fake * noise).sum(), inputs = latents, create_graph = True)

            path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))
            path_mean = self.mean_path_length + 0.01 * (path_lengths.mean() - self.mean_path_length)
            path_loss = (path_lengths - path_mean).pow(2).mean()
            self.mean_path_length = path_mean.detach()

            weighted_path_loss = 2 * 8 * path_loss
            weighted_path_loss += 0 * fake[0, 0, 0]
            weighted_path_loss.backward()
       
        self.opt_gen.step()
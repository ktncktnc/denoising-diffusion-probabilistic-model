import torch.nn.functional as F
from torch import nn 
from .unet import UNet
from .utils import *


class DDPM(nn.Module):
    def __init__(self, img_channels, beta_start=0.0001, beta_end=0.02, timesteps=1000):
        super().__init__()
        self.img_channels = img_channels
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

        self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.unet = UNet(img_channels)

    def diffusion_process(self, x_0, t):
        noise = torch.rand_like(x_0)
        sqrt_alphas_cumprod_t = get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

        return sqrt_alphas_cumprod_t.to(x_0.device) * x_0 + sqrt_one_minus_alphas_cumprod_t.to(x_0.device) * noise.to(x_0.device), noise.to(x_0.device)
    
    def reversed_diffusion_process(self, x_noisy, t):
        noise_predicted = self.unet(x_noisy, t)
        return noise_predicted
    
    def sample_timestep(self, x, t):
        betas_t = get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = get_index_from_list(self.sqrt_recip_alphas, t, x.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t.to(x.device) * (
            x - betas_t.to(x.device) * self.unet(x, t).to(x.device) / sqrt_one_minus_alphas_cumprod_t.to(x.device)
        )
        posterior_variance_t = get_index_from_list(self.posterior_variance, t, x.shape).to(x.device)
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x).to(x.device)
            return model_mean + noise * torch.sqrt(posterior_variance_t)
        

if __name__ == '__main__':
    model = DDPM(3)
    x = torch.randn(1, 3, 256, 256)
    t = torch.randint(0, 100, (1,))

    print(model.diffusion_process(x, t)[0].shape)
    print(model.reversed_diffusion_process(x, t).shape)
    print(model.sample_timestep(x, t).shape)
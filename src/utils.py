import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, start=0.0001, end=0.02, device='cuda:0'):
    return torch.linspace(start, end, timesteps).to(device)


def get_index_from_list(vals, t, x_shape, device='cuda:0'):
    """
    Return a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = x_shape[0]
    out = vals.gather(-1, t.to(vals.device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


def forward_diffusion(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cuda:0"):
    """Take the original image and timestep and return the noisy version of it

    Args:
        x_0 (torch.Tensor): original image
        t (int): timestep 
        device (str, optional): device. Defaults to "cuda:0".
    """
    noise = torch.randn_like(x_0).to(device)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
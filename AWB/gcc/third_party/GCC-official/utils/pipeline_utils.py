import torch
import torch.nn.functional as F

def create_laplacian_pyramid_kernel(device):
    kernel = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]], device=device).unsqueeze(0).unsqueeze(0)
    return kernel

def apply_laplacian_highpass(latents, level=1):
    device = latents.device
    batch_size, channels, height, width = latents.size()
    base_kernel = create_laplacian_pyramid_kernel(device).to(dtype=latents.dtype)
    conv_kernel = base_kernel.repeat(channels, 1, 1, 1)
    highpass_latents = torch.zeros_like(latents)
    current_tensor = latents
    for l in range(level):
        blurred = F.conv2d(current_tensor, conv_kernel, padding=1, groups=channels)
        highfreq = current_tensor - blurred
        current_tensor = F.avg_pool2d(blurred, kernel_size=2)
        if l == 0:
            highpass_latents = highfreq
        else:
            highpass_latents += F.interpolate(highfreq, size=(height, width), mode='bilinear', align_corners=False)
    return highpass_latents

def encode_vae_mean(vae, org_image):
    with torch.no_grad():
        h = vae.encoder(org_image)
        moments = vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latent = mean * vae.config.scaling_factor
    return latent

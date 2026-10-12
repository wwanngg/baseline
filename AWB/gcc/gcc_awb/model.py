import gc
import time
import numpy as np
from PIL import Image
import torch
from .upstream import DEFAULT_WEIGHTS, modules, weights_path
PROMPT = 'a scene with a color checker that accurately reflects the ambient lighting of the scene.'

def resolve_device(requested):
    xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
    cuda = torch.cuda.is_available()
    if requested == 'auto':
        return 'xpu' if xpu else 'cuda' if cuda else 'cpu'
    if requested == 'xpu' and (not xpu):
        raise RuntimeError('XPU is unavailable in this conda env; no silent CPU fallback is used')
    if requested == 'cuda' and (not cuda):
        raise RuntimeError('CUDA is unavailable in this conda env')
    return requested

def synchronize(device):
    if device == 'xpu':
        torch.xpu.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()

class Predictor:

    def __init__(self, model='nus8', weights_dir=DEFAULT_WEIGHTS, device='xpu', dtype='auto'):
        from diffusers import DDIMScheduler
        from diffusers.models.attention_processor import AttnProcessor2_0
        self.device = resolve_device(device)
        self.dtype_name = ('float32' if self.device == 'cpu' else 'float16') if dtype == 'auto' else dtype
        self.dtype = {'float16': torch.float16, 'float32': torch.float32, 'bfloat16': torch.bfloat16}[self.dtype_name]
        self.weights = weights_path(weights_dir, model)
        self.model_name = model
        _, _, original = modules()
        scheduler = DDIMScheduler.from_pretrained(str(self.weights), subfolder='scheduler', local_files_only=True, timestep_spacing='trailing', prediction_type='v_prediction')
        scheduler.set_timesteps(1)
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError('Expected the released 1000-timestep GCC scheduler')
        self.pipeline = original.OneStepLaplacianInpaintPipeline.from_pretrained(str(self.weights), scheduler=scheduler, dtype=self.dtype, local_files_only=True, use_safetensors=True)
        if self.pipeline.unet.config.in_channels != 9:
            raise ValueError('Expected the released nine-channel GCC inpainting UNet')
        self.pipeline.unet.set_attn_processor(AttnProcessor2_0())
        self.pipeline.vae.set_attn_processor(AttnProcessor2_0())
        self.pipeline.to(self.device)
        self.pipeline.unet.eval()
        self.pipeline.vae.eval()
        self.pipeline.text_encoder.eval()
        self.pipeline.set_progress_bar_config(disable=True)
        synchronize(self.device)

    @torch.inference_mode()
    def predict(self, images, masks, seed=42):
        if len(images) != len(masks) or not images:
            raise ValueError('Expected paired nonempty image/mask batches')
        synchronize(self.device)
        started = time.perf_counter()
        generated = self.pipeline([PROMPT] * len(images), image=images, mask_image=masks, generator=torch.Generator(device='cpu').manual_seed(seed), num_inference_steps=1, single_timestep=1000, guidance_scale=0, strength=0.999, output_type='np').images
        synchronize(self.device)
        elapsed = time.perf_counter() - started
        if not np.isfinite(generated).all() or generated.shape != (len(images), 512, 512, 3):
            raise RuntimeError('GCC generated nonfinite values or unexpected image dimensions')
        outputs = [Image.fromarray((image * 255).round().astype(np.uint8)) for image in generated]
        return (outputs, elapsed)

    def close(self):
        if hasattr(self, 'pipeline'):
            del self.pipeline
        gc.collect()
        if self.device == 'xpu':
            torch.xpu.empty_cache()
        elif self.device == 'cuda':
            torch.cuda.empty_cache()

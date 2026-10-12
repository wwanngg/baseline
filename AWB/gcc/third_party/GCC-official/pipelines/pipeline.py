from diffusers import StableDiffusionInpaintPipeline
from typing import Dict, Union
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DiffusionPipeline, LCMScheduler, UNet2DConditionModel, DDPMScheduler
from diffusers.models import AsymmetricAutoencoderKL
from diffusers.utils import BaseOutput, deprecate
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from PIL import Image
from torchvision.transforms.functional import resize, pil_to_tensor
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import PIL.Image
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import torch.nn.functional as F
from utils import apply_laplacian_highpass

class OneStepLaplacianInpaintPipeline(StableDiffusionInpaintPipeline):

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        h = self.vae.encoder(image)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.vae.config.scaling_factor
        return rgb_latent

    def prepare_mask_latents(self, mask, masked_image, batch_size, height, width, dtype, device, generator, do_classifier_free_guidance):
        mask = torch.nn.functional.interpolate(mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor))
        mask = mask.to(device=device, dtype=dtype)
        masked_image = masked_image.to(device=device, dtype=dtype)
        if masked_image.shape[1] == 4:
            masked_image_latents = masked_image
        else:
            masked_image_latents = self.encode(masked_image)
        if mask.shape[0] < batch_size:
            if not batch_size % mask.shape[0] == 0:
                raise ValueError(f"The passed mask and the required batch size don't match. Masks are supposed to be duplicated to a total batch size of {batch_size}, but {mask.shape[0]} masks were passed. Make sure the number of masks that you pass is divisible by the total requested batch size.")
            mask = mask.repeat(batch_size // mask.shape[0], 1, 1, 1)
        if masked_image_latents.shape[0] < batch_size:
            if not batch_size % masked_image_latents.shape[0] == 0:
                raise ValueError(f"The passed images and the required batch size don't match. Images are supposed to be duplicated to a total batch size of {batch_size}, but {masked_image_latents.shape[0]} images were passed. Make sure the number of images that you pass is divisible by the total requested batch size.")
            masked_image_latents = masked_image_latents.repeat(batch_size // masked_image_latents.shape[0], 1, 1, 1)
        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        return (mask, masked_image_latents)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None, image=None, timestep=None, is_strength_max=True, return_noise=False, return_image_latents=False):
        shape = (batch_size, num_channels_latents, int(height) // self.vae_scale_factor, int(width) // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
        if (image is None or timestep is None) and (not is_strength_max):
            raise ValueError('Since strength < 1. initial latents are to be initialised as a combination of Image + Noise.However, either the image or the noise timestep has not been provided.')
        if return_image_latents or (latents is None and (not is_strength_max)):
            image = image.to(device=device, dtype=dtype)
            if image.shape[1] == 4:
                image_latents = image
            else:
                image_latents = self.encode(image=image)
                image_latents = apply_laplacian_highpass(image_latents, level=2)
            image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
        if latents is None:
            noise = torch.zeros_like(image_latents)
            latents = noise if is_strength_max else self.scheduler.add_noise(image_latents, noise, timestep)
            latents = latents * self.scheduler.init_noise_sigma if is_strength_max else latents
        else:
            noise = latents.to(device)
            latents = noise * self.scheduler.init_noise_sigma
        outputs = (latents,)
        if return_noise:
            outputs += (noise,)
        if return_image_latents:
            outputs += (image_latents,)
        return outputs

    @torch.no_grad()
    def __call__(self, prompt: Union[str, List[str]]=None, image: PipelineImageInput=None, mask_image: PipelineImageInput=None, masked_image_latents: torch.Tensor=None, canny_image: PipelineImageInput=None, height: Optional[int]=None, width: Optional[int]=None, padding_mask_crop: Optional[int]=None, strength: float=0.999, num_inference_steps: int=50, timesteps: List[int]=None, sigmas: List[float]=None, guidance_scale: float=7.5, negative_prompt: Optional[Union[str, List[str]]]=None, num_images_per_prompt: Optional[int]=1, eta: float=0.0, generator: Optional[Union[torch.Generator, List[torch.Generator]]]=None, latents: Optional[torch.Tensor]=None, prompt_embeds: Optional[torch.Tensor]=None, negative_prompt_embeds: Optional[torch.Tensor]=None, ip_adapter_image: Optional[PipelineImageInput]=None, ip_adapter_image_embeds: Optional[List[torch.Tensor]]=None, output_type: Optional[str]='pil', return_dict: bool=True, cross_attention_kwargs: Optional[Dict[str, Any]]=None, clip_skip: int=None, single_timestep: int=1000, callback_on_step_end: Optional[Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]]=None, callback_on_step_end_tensor_inputs: List[str]=['latents'], **kwargs):
        callback = kwargs.pop('callback', None)
        callback_steps = kwargs.pop('callback_steps', None)
        if callback is not None:
            deprecate('callback', '1.0.0', 'Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`')
        if callback_steps is not None:
            deprecate('callback_steps', '1.0.0', 'Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`')
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        self.check_inputs(prompt, image, mask_image, height, width, strength, callback_steps, output_type, negative_prompt, prompt_embeds, negative_prompt_embeds, ip_adapter_image, ip_adapter_image_embeds, callback_on_step_end_tensor_inputs, padding_mask_crop)
        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        text_encoder_lora_scale = cross_attention_kwargs.get('scale', None) if cross_attention_kwargs is not None else None
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, device, num_images_per_prompt, self.do_classifier_free_guidance, negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, lora_scale=text_encoder_lora_scale, clip_skip=self.clip_skip)
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(ip_adapter_image, ip_adapter_image_embeds, device, batch_size * num_images_per_prompt, self.do_classifier_free_guidance)
        timesteps = torch.tensor([single_timestep - 1])
        if num_inference_steps < 1:
            raise ValueError(f'After adjusting the num_inference_steps by strength parameter: {strength}, the number of pipelinesteps is {num_inference_steps} which is < 1 and not appropriate for this pipeline.')
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        is_strength_max = strength == 1.0
        if padding_mask_crop is not None:
            crops_coords = self.mask_processor.get_crop_region(mask_image, width, height, pad=padding_mask_crop)
            resize_mode = 'fill'
        else:
            crops_coords = None
            resize_mode = 'default'
        original_image = image
        init_image = self.image_processor.preprocess(image, height=height, width=width, crops_coords=crops_coords, resize_mode=resize_mode)
        init_image = init_image.to(dtype=torch.float32)
        num_channels_latents = self.vae.config.latent_channels
        num_channels_unet = self.unet.config.in_channels
        return_image_latents = num_channels_unet == 9
        latents_outputs = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents, image=init_image, timestep=latent_timestep, is_strength_max=is_strength_max, return_noise=True, return_image_latents=return_image_latents)
        if return_image_latents:
            latents, noise, image_latents = latents_outputs
        else:
            latents, noise = latents_outputs
        org_latents = latents
        mask_condition = self.mask_processor.preprocess(mask_image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords)
        if masked_image_latents is None:
            masked_image = init_image * (1 - mask_condition)
        else:
            masked_image = masked_image_latents
        mask, masked_image_latents = self.prepare_mask_latents(mask_condition, masked_image, batch_size * num_images_per_prompt, height, width, prompt_embeds.dtype, device, generator, self.do_classifier_free_guidance)
        if num_channels_unet == 9:
            num_channels_mask = mask.shape[1]
            num_channels_masked_image = masked_image_latents.shape[1]
            if num_channels_latents + num_channels_mask + num_channels_masked_image != self.unet.config.in_channels:
                raise ValueError(f'Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} + `num_channels_mask`: {num_channels_mask} + `num_channels_masked_image`: {num_channels_masked_image} = {num_channels_latents + num_channels_masked_image + num_channels_mask}. Please verify the config of `pipeline.unet` or your `mask_image` or `image` input.')
        elif num_channels_unet != 4 and canny_image is None:
            raise ValueError(f'The unet {self.unet.__class__} should have either 4 or 9 input channels, not {self.unet.config.in_channels}.')
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = {'image_embeds': image_embeds} if ip_adapter_image is not None or ip_adapter_image_embeds is not None else None
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim).to(device=device, dtype=latents.dtype)
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if num_channels_unet == 9:
                    latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = self.scheduler.step(noise_pred, t, latents)
                latents = latents.pred_original_sample
                if num_channels_unet == 4:
                    init_latents_proper = image_latents
                    if self.do_classifier_free_guidance:
                        init_mask, _ = mask.chunk(2)
                    else:
                        init_mask = mask
                    if i < len(timesteps) - 1:
                        noise_timestep = timesteps[i + 1]
                        init_latents_proper = self.scheduler.add_noise(init_latents_proper, noise, torch.tensor([noise_timestep]))
                    latents = (1 - init_mask) * init_latents_proper + init_mask * latents
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop('latents', latents)
                    prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop('negative_prompt_embeds', negative_prompt_embeds)
                    mask = callback_outputs.pop('mask', mask)
                    masked_image_latents = callback_outputs.pop('masked_image_latents', masked_image_latents)
                if i == len(timesteps) - 1 or (i + 1 > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, 'order', 1)
                        callback(step_idx, t, latents)
        if not output_type == 'latent':
            condition_kwargs = {}
            if isinstance(self.vae, AsymmetricAutoencoderKL):
                init_image = init_image.to(device=device, dtype=masked_image_latents.dtype)
                init_image_condition = init_image.clone()
                init_image = self.encode(init_image)
                mask_condition = mask_condition.to(device=device, dtype=masked_image_latents.dtype)
                condition_kwargs = {'image': init_image_condition, 'mask': mask_condition}
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None
        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        if padding_mask_crop is not None:
            image = [self.image_processor.apply_overlay(mask_image, original_image, i, crops_coords) for i in image]
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image, has_nsfw_concept)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

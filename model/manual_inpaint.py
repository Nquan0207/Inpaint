from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from diffusers import DDIMScheduler
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from .inpainting import preprocess_img_and_mask, get_sd_device_and_dtype


# Simple cache to avoid reloading weights on repeated calls
_CACHED = {
    "model_id": None,
    "tokenizer": None,
    "text_encoder": None,
    "vae": None,
    "unet": None,
    "scheduler": None,
    "device": None,
    "dtype": None,
}


def _load_components(model_id: str):
    device, dtype = get_sd_device_and_dtype()

    if (
        _CACHED["model_id"] == model_id
        and _CACHED["tokenizer"] is not None
        and _CACHED["text_encoder"] is not None
        and _CACHED["vae"] is not None
        and _CACHED["unet"] is not None
        and _CACHED["scheduler"] is not None
    ):
        return (
            _CACHED["tokenizer"],
            _CACHED["text_encoder"],
            _CACHED["vae"],
            _CACHED["unet"],
            _CACHED["scheduler"],
            device,
            dtype,
        )

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

    # Use a clean DDIM scheduler by default (robust for img2img/inpaint manual loop)
    try:
        scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    except Exception:
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

    tokenizer = tokenizer
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    unet = unet.to(device)

    # dtype handling: prefer float16 on CUDA, float32 otherwise
    if dtype == torch.float16:
        text_encoder = text_encoder.half()
        vae = vae.half()
        unet = unet.half()

    text_encoder.eval()
    vae.eval()
    unet.eval()

    _CACHED.update(
        dict(
            model_id=model_id,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )
    )

    return tokenizer, text_encoder, vae, unet, scheduler, device, dtype


def _encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
    dtype: torch.dtype,
):
    # Conditional
    text_inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # Unconditional (for CFG)
    uncond_inputs = tokenizer(
        [negative_prompt] if negative_prompt is not None else [""],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        cond = text_encoder(text_inputs.input_ids.to(device))[0]
        uncond = text_encoder(uncond_inputs.input_ids.to(device))[0]

    cond = cond.to(dtype)
    uncond = uncond.to(dtype)
    # Shape: [2, seq, hidden]
    return torch.cat([uncond, cond], dim=0)


def _pil_to_tensor_bchw(img: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0  # HWC in [0,1]
    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    t = torch.from_numpy(arr)[None, ...]  # BCHW
    t = t.to(device=device, dtype=torch.float32)
    return t.to(dtype)


@torch.no_grad()
def manual_sd15_inpaint(
    img_pil: Image.Image,
    mask_pil: Image.Image,
    prompt: str = "photo of a cat or dog, realistic",
    negative_prompt: str = "nsfw, watermark, text",
    model_id: Optional[str] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
    invert_mask: bool = False,
) -> Image.Image:
    """
    Manual Stable Diffusion 1.5 inpainting loop (no high-level Pipeline).

    - Uses UNet(9ch), VAE, CLIP text encoder, and DDIM scheduler.
    - White(255) in mask = region to inpaint.
    - Preserves unmasked region each step.
    """
    model_id = model_id or "runwayml/stable-diffusion-inpainting"

    tokenizer, text_encoder, vae, unet, scheduler, device, dtype = _load_components(model_id)

    # Preprocess image and mask (binary, padded to multiple of 8, size-matched)
    img_proc, mask_proc = preprocess_img_and_mask(img_pil, mask_pil, invert_mask=invert_mask)
    orig_w, orig_h = img_pil.size

    # To tensors
    image = _pil_to_tensor_bchw(img_proc, device, dtype)  # [B,3,H,W] in [0,1]
    mask = _pil_to_tensor_bchw(mask_proc, device, dtype)   # [B,1,H,W] in [0,1]
    mask = mask[:, :1, ...]  # ensure single channel

    # Encode prompts (Classifier-Free Guidance)
    text_embeds = _encode_prompt(tokenizer, text_encoder, prompt, negative_prompt, device, dtype)

    # Normalize to [-1,1] before VAE
    image = image * 2.0 - 1.0

    # VAE encode image to latents
    posterior = vae.encode(image).latent_dist
    latents = posterior.sample()
    latents = latents * vae.config.scaling_factor  # typically 0.18215

    # Prepare mask and masked image latents in latent space (H/8, W/8)
    latent_h, latent_w = latents.shape[-2:]
    mask_latents = F.interpolate(mask, size=(latent_h, latent_w), mode="nearest")  # [B,1,h/8,w/8] in {0,1}
    masked_image_latents = latents * (1.0 - mask_latents)

    # Scheduler setup and add noise according to strength (img2img style)
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Choose start step by strength (e.g., 1.0 -> full noise, 0.0 -> no noise)
    start_idx = int(len(timesteps) * strength)
    start_idx = min(max(start_idx, 0), len(timesteps) - 1)
    timesteps = timesteps[start_idx:]

    noise = torch.randn_like(latents, device=device, dtype=dtype)
    if len(timesteps) > 0:
        latents = scheduler.add_noise(latents, noise, timesteps[0])

    # Denoising loop
    for t in timesteps:
        # Prepare model input: concat [latents, mask, masked_image_latents] along channel dim
        latent_model_input = torch.cat([latents, mask_latents, masked_image_latents], dim=1)

        # Classifier-Free Guidance: duplicate for [uncond, cond]
        latent_in = torch.cat([latent_model_input, latent_model_input], dim=0)
        noise_pred = unet(latent_in, t, encoder_hidden_states=text_embeds).sample

        # Split and apply guidance
        noise_uncond, noise_cond = noise_pred.chunk(2, dim=0)
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

        # Step
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        # Preserve unmasked (known) region in latent space each step
        latents = masked_image_latents + latents * mask_latents

    # Decode via VAE
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents).sample
    image = (image / 2.0 + 0.5).clamp(0.0, 1.0)

    # To PIL
    image = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()[0]
    out = Image.fromarray((image * 255).astype(np.uint8))

    # Resize back to original if needed
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)
    return out


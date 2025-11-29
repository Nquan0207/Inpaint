from PIL import Image
import numpy as np

import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from typing import Optional

def get_sd_device_and_dtype():
    if torch.backends.mps.is_available():
        # MPS is still finicky with float16; stay on float32 to avoid NaNs
        return torch.device("mps"), torch.float32
    elif torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    else:
        return torch.device("cpu"), torch.float32

def load_sd_inpaint(model_id: Optional[str] = None):
    """
    Load an inpainting pipeline. Supports SD 1.5 and SDXL.

    If `model_id` points to an SDXL base repo (e.g. "stabilityai/stable-diffusion-xl-base-1.0"),
    we map it to the SDXL inpainting checkpoint for compatibility.
    """
    device, dtype = get_sd_device_and_dtype()

    # Default to SD 1.5 inpainting if nothing provided
    model_id = model_id or "runwayml/stable-diffusion-inpainting"

    # Map common aliases to correct inpainting repos
    alias_map = {
        # User requested SDXL base → map to SDXL inpainting weights
        "stabilityai/stable-diffusion-xl-base-1.0": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        # Sometimes people reference refiner for inpainting by mistake
        "stabilityai/stable-diffusion-xl-refiner-1.0": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    }
    resolved_id = alias_map.get(model_id, model_id)

    # Choose pipeline class based on model family
    if "stable-diffusion-xl" in resolved_id:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            resolved_id,
            dtype=dtype,
            use_safetensors=True,
            safety_checker=None,
        ).to(device)
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            resolved_id,
            dtype=dtype,
            safety_checker=None,  # disable at load-time (diffusers >= 0.27)
        ).to(device)
    # If your diffusers version doesn't accept safety_checker=None, use the monkey-patch below:
    if getattr(pipe, "safety_checker", None) is not None:
        def _dummy_checker(images, **kwargs):
            # return images, list_of_flags
            return images, [False] * len(images)
        pipe.safety_checker = _dummy_checker

    pipe.enable_attention_slicing()
    return pipe, device

def _to_multiple_of_8(w, h):
    return (w - w % 8), (h - h % 8)

def preprocess_img_and_mask(img_pil: Image.Image, mask_pil: Image.Image, invert_mask: bool = False):
    # Convert modes
    img = img_pil.convert("RGB")
    mask = mask_pil.convert("L")  # white(255)=to inpaint, black(0)=keep

    # Force binary mask (remove gray pixels)
    mask_np = np.array(mask)
    if invert_mask:
        mask_np = 255 - mask_np
    mask_bin = (mask_np > 127).astype(np.uint8) * 255
    mask = Image.fromarray(mask_bin, mode="L")

    # Match sizes
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)

    # Pad to multiple of 8 (avoid shrinking that can leave visible borders)
    W, H = img.size
    pad_w = (8 - (W % 8)) % 8
    pad_h = (8 - (H % 8)) % 8
    if pad_w or pad_h:
        import cv2
        img_np = np.array(img)
        m_np = np.array(mask)
        img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)
        m_np = cv2.copyMakeBorder(m_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        img = Image.fromarray(img_np)
        mask = Image.fromarray(m_np, mode="L")

    # Guard: mask not all-zeros or all-ones
    m = np.array(mask)
    h, w = m.shape[:2]
    if m.max() == 0:
        # Nothing to inpaint → nudge a pixel (rare)
        m[h//2, w//2] = 255
        mask = Image.fromarray(m, mode="L")
    elif m.min() == 255:
        # Inpainting whole image causes poor results; erode a bit
        import cv2
        m = cv2.erode(m, np.ones((7,7), np.uint8), iterations=1)
        mask = Image.fromarray(m, mode="L")

    return img, mask

def sd_inpaint(
    img_pil,
    mask_pil,
    prompt: str = "photo of a cat or dog, realistic",
    negative_prompt: str = "nsfw, nude, explicit, watermark, text",
    model_id: Optional[str] = None,
    num_inference_steps: int = 40,
    guidance_scale: float = 8.0,
    strength: float = 1.0,
    invert_mask: bool = False,
):
    pipe, device = load_sd_inpaint(model_id=model_id)
    orig_w, orig_h = img_pil.size
    img, mask = preprocess_img_and_mask(img_pil, mask_pil, invert_mask=invert_mask)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img,
        mask_image=mask,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
    )

    # For SDXL, pass composition-preserving size hints
    # SDXL composition hints are typically used for text2img; keep disabled for inpaint

    with torch.no_grad():
        out = pipe(**call_kwargs).images[0]

    # Resize back to original size if dimensions differ
    if out.size != (orig_w, orig_h):
        out = out.resize((orig_w, orig_h), Image.LANCZOS)
    return out

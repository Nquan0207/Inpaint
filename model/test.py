from diffusers import DiffusionPipeline
from PIL import Image
import torch

def get_device_and_dtype():
    if torch.backends.mps.is_available():
        # MPS is more stable with float32
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32

device, dtype = get_device_and_dtype()

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# If you hit OOM on MPS/CPU, try lowering steps or switch to SD 1.5.
# Optionally, if you have accelerate installed and want CPU offload instead:
# (comment out .to(device) above before enabling the next line)
# pipe.enable_sequential_cpu_offload()

prompt = "An astronaut riding a dark horse"
image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.0).images[0]
image.save("outputs/sdxl_test.png")

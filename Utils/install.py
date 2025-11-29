from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import os

root = "./data"
ds = OxfordIIITPet(
    root=root,
    download=True,
    target_types="segmentation",
    transform=transforms.Resize((512,512)),   # normalize to 512x512
)
print("Total images:", len(ds), "-> data at:", os.path.join(root, "oxford-iiit-pet"))
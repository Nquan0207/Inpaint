import torch, torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import numpy as np

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

_device = get_device()

# Use weights enum (DEFAULT gives latest recommended weights)
_model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
_model = _model.to(_device).eval()

@torch.no_grad()
def predict_pet_mask(img_pil):
    tr = torchvision.transforms.ToTensor()
    x = tr(img_pil).unsqueeze(0).to(_device)
    out = _model(x)[0]
    # COCO: 17=cat, 18=dog
    keep = [(int(l)==17 or int(l)==18) and float(s)>0.5 for l,s in zip(out["labels"], out["scores"])]
    if not any(keep):
        w,h = img_pil.size
        return np.zeros((h,w), np.uint8)
    masks = out["masks"][keep].squeeze(1) > 0.5
    if masks.ndim==2:
        m = masks
    else:
        m = torch.any(masks, dim=0)
    m = m.float().cpu().numpy()
    return (m*255).astype(np.uint8)
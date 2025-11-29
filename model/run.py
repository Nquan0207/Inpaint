# model/run.py
from PIL import Image
import cv2, numpy as np
from pathlib import Path

from .seg_maskcnn import get_device, _model as seg_model  # after you refactor as in section 1
from .inpainting import sd_inpaint
import torchvision, torch
from torchvision.transforms import ToTensor

def _ensure_nonempty_mask(mask_img, min_white=1000, ksize=15, invert=False):
    m = np.array(mask_img)
    if invert:
        m = 255 - m
    unique, counts = np.unique(m, return_counts=True)
    print("mask stats:", dict(zip(unique.tolist(), counts.tolist())))
    white = int((m == 255).sum())
    if white < min_white:
        print(f"mask too small ({white}px); dilating {ksize}x{ksize}")
        m = cv2.dilate(m, np.ones((ksize, ksize), np.uint8), iterations=1)
    return Image.fromarray(m, "L")


def predict_pet_mask_pil(img_pil, score_thresh: float = 0.7):
    tr = ToTensor()
    x = tr(img_pil).unsqueeze(0).to(get_device())
    with torch.no_grad():
        out = seg_model(x)[0]
    labels = out["labels"].tolist()
    scores = out["scores"].tolist()
    print("Predicted labels and scores:", list(zip(labels, scores)))
    masks  = out["masks"]
    keep = [ (l in [17,18]) and (s >= score_thresh) for l,s in zip(labels, scores) ]
    kept_labels = [ l for l,s in zip(labels, scores) if s >= score_thresh ]
    print(f"Using score_thresh >= {score_thresh}; kept pet detections mask flags: {keep}")
    print("Final kept labels (thresholded):", kept_labels)
    if not any(keep):
        w,h = img_pil.size
        return Image.fromarray(np.zeros((h,w), np.uint8), "L"), kept_labels
    m = (masks[keep].squeeze(1) > 0.5).any(dim=0).float().cpu().numpy()
    return Image.fromarray((m*255).astype(np.uint8), "L"), kept_labels

def main():
    img_path = Path("data/occluded") / "Bengal_102.jpg"          # adjust
    occ_path = Path("data/occluded") / "Bengal_102_mask.png"     # adjust
    img = Image.open(img_path).convert("RGB")
    mask_occ = Image.open(occ_path).convert("L")
    mask_pred, kept_labels = predict_pet_mask_pil(img, score_thresh=0.7)
    # Baseline ổn định: dùng trực tiếp occlusion mask và nới rộng để che kín viền
    k = 31
    m_occ = cv2.dilate(np.array(mask_occ), np.ones((k,k), np.uint8), iterations=1)
    mask_inpaint = Image.fromarray(m_occ, "L")

    # In thống kê để debug nhanh
    for name, arr in (
        ("pred", np.array(mask_pred)),
        ("occ", np.array(mask_occ)),
        ("inpaint", np.array(mask_inpaint)),
    ):
        u, c = np.unique(arr, return_counts=True)
        print(f"mask {name} stats:", dict(zip(u.tolist(), c.tolist())))
    # Fallback: nếu vẫn quá nhỏ thì dùng trực tiếp mask_occ để test pipeline
    m_arr = np.array(mask_inpaint)
    if (m_arr == 255).sum() < 1000:
        print("fallback to mask_occ (test pipeline)")
        mask_inpaint = _ensure_nonempty_mask(mask_occ, min_white=1000, ksize=15, invert=False)

    # Chọn prompt theo nhãn dự đoán (COCO: 17=cat, 18=dog) sau khi lọc theo score
    print("Final predicted labels (thresholded):", kept_labels)
    has_cat = 17 in kept_labels
    has_dog = 18 in kept_labels
    if has_cat and not has_dog:
        prompt = "restore occluded cat fur, keep same cat identity, realistic photo"
        negative = "dog, canine, snout, muzzle, watermark, text, nsfw"
    elif has_dog and not has_cat:
        prompt = "restore occluded dog fur, keep same dog identity, realistic photo"
        negative = "cat, feline, whiskers, watermark, text, nsfw"
    else:
        prompt = "restore occluded pet fur, keep same identity, realistic photo"
        negative = "watermark, text, nsfw"

    out = sd_inpaint(
        img,
        mask_inpaint,
        prompt=prompt,
        negative_prompt=negative,
        model_id="stabilityai/stable-diffusion-xl-base-1.0",
        num_inference_steps=40,
        guidance_scale=7.5,
        strength=1.0,
    )
    out.save("outputs/result.png")
    mask_pred.save("outputs/mask_pred.png")
    mask_inpaint.save("outputs/mask_inpaint.png")
    print("Done → outputs/result.png")

if __name__ == "__main__":
    main()

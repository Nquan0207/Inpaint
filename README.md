DeepRestore — Inpainting Pipeline (Scratch + Diffusion)

Quickstart

- Create/activate a Python 3.10+ env and install deps:
  - `pip install -r requirements.txt`
- Download Oxford‑IIIT Pet (for clean targets):
  - `python Utils/install.py`
- Generate occluded images + masks (white=hole, black=keep):
  - `python Utils/gen_occ.py --num 100 --size 0.18,0.30`
  - Outputs to `data/occluded/` as `<name>.jpg` and `<name>_mask.png`
- Train from scratch in the notebook:
  - Open `model/model.ipynb`
  - Ensure paths in the config cell point to:
    - `base_img_dir="../data/oxford-iiit-pet/images"`
    - `occluded_dir="../data/occluded"`
  - Run cells top → bottom. Results saved to `model/outputs/`.

What’s Included

- Scratch model training (Simple U‑Net)
  - Input: 3‑channel occluded RGB + 1‑channel mask
  - Loss: Masked L1 in the hole + identity outside the mask
  - Blended previews: `pred = y_hat*mask + occ_rgb*(1-mask)` to preserve context
  - Metrics on masked region: PSNR, SSIM, LPIPS

- Data utilities
  - `Utils/install.py`: downloads Oxford‑IIIT Pet under `data/oxford-iiit-pet/`
  - `Utils/gen_occ.py`: generates occluded images and masks
    - Key args:
      - `--num N` pick N images (default: all)
      - `--size lo,hi` rectangle side as fraction of min(H,W)
      - `--per-image K` create K variations per source image
      - `--src-dir`, `--out-dir`, `--seed`

- Diffusion inpainting (manual loop)
  - `model/manual_inpaint.py`: Stable Diffusion 1.5 inpainting loop without Pipeline
  - Uses CLIP text encoder, VAE, UNet, DDIM scheduler (via diffusers)
  - Example use inside Python:
    - `from model.manual_inpaint import manual_sd15_inpaint`

Run SD inpainting demo (optional)

- The `model/run.py` demo uses segmentation to pick prompts and SDXL/SD‑1.5 inpaint.
- Run from repo root as a package module:
  - `python -m model.run`

Notebook Tips

- Paths
  - The notebook lives in `model/`, so relative paths use `../data/...` by default.
  - You can pass absolute paths in `TrainConfig`.
- Batch/epochs
  - Default `batch_size=8`, `epochs=2` (for a quick smoke test). Increase for quality.
- More data
  - Generate more occlusions: `python Utils/gen_occ.py --num 1000 --size 0.18,0.30`
- Speed
  - Apple M‑series (MPS) and CUDA are auto‑detected. CPU will be slower.

Troubleshooting

- “No such file or directory: data/occluded”
  - Generate occlusions first: `python Utils/gen_occ.py --num 50`
- Masks look inverted or predictions still gray
  - Use the “Debug mask polarity” cell in the notebook; it writes `debug_*` images to `model/outputs/`.
  - Ensure previews are saved from blended output, not the raw network output.
- Mismatched names between clean and occluded
  - Each occluded file `<stem>.jpg` must map to clean `<stem>.jpg` in `oxford-iiit-pet/images` and to mask `<stem>_mask.png` in `occluded`.
  - If you used `--per-image > 1` (e.g., `_occ1` suffix), adjust the mapping in the Dataset as noted in the notebook.
- OpenCV read error in `gen_occ.py`
  - The script now skips unreadable files and supports `.jpg/.jpeg/.png` (any case). Verify `--src-dir` points to images.

Repository Map

- `model/model.ipynb` — end‑to‑end training/eval notebook (recommended entry point)
- `Utils/install.py` — downloads Oxford‑IIIT Pet
- `Utils/gen_occ.py` — occlusion + mask generation
- `model/inpainting.py` — SD inpaint pipeline helpers + preprocessing (mask binarize, pad, guard)
- `model/manual_inpaint.py` — manual SD 1.5 inpainting loop (no Pipeline)
- `model/run.py` — demo mixing segmentation + SD inpaint

Notes on Masks

- Convention: white (255) = region to inpaint; black (0) = keep.
- Preprocessing pads to multiples of 8 and replicates image borders (avoids edge artifacts).
- The notebook zeroes the hole in the network input so the model can’t “copy” the occluder.

License

- Academic/experimental use. See dataset license for Oxford‑IIIT Pet.

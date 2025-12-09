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

Related work:

- DeepGIN: https://arxiv.org/abs/2008.07173
- Edge connect: https://arxiv.org/abs/1901.00212
- Generative Image Inpainting with Contextual Attention: https://arxiv.org/pdf/1801.07892
- Globally and Locally Consistent Image Completion: https://github.com/satoshiiizuka/siggraph2017_inpainting
- Globally and Locally Consistent Image Completion: https://iizuka.cs.tsukuba.ac.jp/projects/completion/data/completion_sig2017.pdf
- v2: https://github.com/JiahuiYu/generative_inpainting
- diffusion-based image inpainting pipeline: https://arxiv.org/pdf/2403.16016
- Exemplar-based Image Editing with Diffusion Models: https://arxiv.org/pdf/2211.13227
- U-Net: https://arxiv.org/pdf/1505.04597
- contextencoders: https://arxiv.org/pdf/1604.07379
- Generative Adversarial Nets: https://arxiv.org/pdf/1406.2661
- Conditional Adversarial Networks: https://arxiv.org/pdf/1611.07004
- RePaint: Inpainting using Denoising Diffusion Probabilistic Models: https://arxiv.org/pdf/2201.09865
- High-Resolution Image Synthesis with Latent Diffusion Models: https://arxiv.org/pdf/2112.10752
- Denoising Diffusion Probabilistic Models: https://arxiv.org/abs/2006.11239

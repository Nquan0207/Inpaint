import cv2, os, random, numpy as np, argparse
from glob import glob

def parse_size_range(s):
    try:
        a, b = s.split(",")
        return float(a), float(b)
    except Exception:
        raise argparse.ArgumentTypeError("--size phải ở dạng 'min,max', ví dụ: 0.08,0.16")


parser = argparse.ArgumentParser(description="Tạo ảnh occlusion nhỏ ở giữa ảnh")
parser.add_argument("--src-dir", default="./data/oxford-iiit-pet/images", help="Thư mục ảnh nguồn")
parser.add_argument("--out-dir", default="./data/occluded", help="Thư mục lưu ảnh occluded")
parser.add_argument("--num", type=int, default=None, help="Số lượng ảnh nguồn sẽ tạo (mặc định: tất cả)")
parser.add_argument("--per-image", type=int, default=1, help="Số biến thể occlusion cho mỗi ảnh")
parser.add_argument("--size", type=parse_size_range, default=(0.12, 0.22), help="Khoảng tỉ lệ kích thước mask theo cạnh ngắn, ví dụ 0.12,0.22")
parser.add_argument("--seed", type=int, default=42, help="Seed cho random để tái lập")
args = parser.parse_args()

src_dir = args.src_dir
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
random.seed(args.seed)

# Collect images robustly (handle different extensions/case)
patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
imgs = []
for pat in patterns:
    imgs.extend(sorted(glob(os.path.join(src_dir, pat))))
# de-duplicate while preserving order
imgs = list(dict.fromkeys(imgs))
if not imgs:
    raise SystemExit(f"No images found under: {src_dir}")

if args.num is not None:
    take_n = min(args.num, len(imgs))
    chosen = random.sample(imgs, take_n)
else:
    chosen = imgs

for img_path in chosen:
    img = cv2.imread(img_path)
    if img is None:
        print("[warn] skip unreadable:", img_path)
        continue
    h,w = img.shape[:2]
    mask = np.zeros((h,w), np.uint8)
    name = os.path.basename(img_path)
    stem = os.path.splitext(name)[0]

    for k in range(args.per_image):
        mask[:] = 0
        # vẽ 1 hình chữ nhật nhỏ che gần giữa ảnh
        min_frac, max_frac = args.size
        size = max(5, int(random.uniform(min_frac, max_frac) * min(h, w)))

        # chọn tâm hình chữ nhật trong vùng giữa (35% - 65%)
        cx_min, cx_max = int(0.35 * w), int(0.65 * w)
        cy_min, cy_max = int(0.35 * h), int(0.65 * h)
        cx = random.randint(cx_min, cx_max)
        cy = random.randint(cy_min, cy_max)

        # suy ra góc trên-trái từ tâm, đảm bảo không vượt biên
        x1 = max(0, min(w - size, cx - size // 2))
        y1 = max(0, min(h - size, cy - size // 2))
        x2, y2 = x1 + size, y1 + size
        cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)

        occ = img.copy()
        occ[mask>0] = (127,127,127)

        if args.per_image > 1:
            out_name = f"{stem}_occ{k+1}.jpg"
            mask_name = f"{stem}_occ{k+1}_mask.png"
        else:
            out_name = f"{stem}.jpg"
            mask_name = f"{stem}_mask.png"

        cv2.imwrite(os.path.join(out_dir, out_name), occ)
        cv2.imwrite(os.path.join(out_dir, mask_name), mask)

print("Đã tạo occlusions ở:", out_dir)

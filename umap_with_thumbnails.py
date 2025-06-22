import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
from PIL import Image
import numpy as np

# === Configuration ===
PATCH_DIR = "sample_patches"
CSV_PATH = "patch_clusters.csv"
OUTPUT_IMG = "umap_thumbnails.png"
THUMB_SIZE = 64  # size of thumbnail images in the UMAP

# === Load Data ===
print("📄 Loading patch metadata...")
df = pd.read_csv(CSV_PATH)

# === Set up plot ===
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("🧬 Patch UMAP with Thumbnails")
ax.set_xlabel("UMAP-1")
ax.set_ylabel("UMAP-2")

# === Normalize coordinates to improve layout ===
x = df["umap_x"].values
y = df["umap_y"].values

x_norm = (x - x.min()) / (x.max() - x.min()) * 10
y_norm = (y - y.min()) / (y.max() - y.min()) * 10

# === Plot each patch as a thumbnail ===
for i, row in df.iterrows():
    file_path = os.path.join(PATCH_DIR, row["filename"])
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.Resampling.LANCZOS)
        imagebox = OffsetImage(np.array(img), zoom=1)
        ab = AnnotationBbox(imagebox, (x_norm[i], y_norm[i]), frameon=True, pad=0.2, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    except Exception as e:
        print(f"⚠️ Could not load {file_path}: {e}")

ax.set_xlim(x_norm.min() - 1, x_norm.max() + 1)
ax.set_ylim(y_norm.min() - 1, y_norm.max() + 1)
plt.tight_layout()
plt.savefig(OUTPUT_IMG, dpi=300)
print(f"✅ Saved UMAP with thumbnails to: {OUTPUT_IMG}")

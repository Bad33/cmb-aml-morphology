import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import umap

from torchvision import transforms
from transformers import AutoImageProcessor, ViTModel

# === Config ===
PATCH_DIR = "clean_patches"
MODEL_NAME = "owkin/phikon-v2"
DEVICE = torch.device("cpu")
N_CLUSTERS = 3
OUT_CSV = "clean_patch_clusters.csv"
OUT_IMG = "clean_umap_clustered.png"

# === Load Model ===
print("🔄 Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# === Load Patches ===
patch_files = sorted([
    os.path.join(PATCH_DIR, f) for f in os.listdir(PATCH_DIR) if f.endswith(".png")
])
print(f"📦 Found {len(patch_files)} clean patches.")

# === Extract Features ===
embeddings = []
filenames = []

with torch.no_grad():
    for f in tqdm(patch_files, desc="🧠 Extracting embeddings"):
        img = Image.open(f).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        feat = outputs.pooler_output.cpu().squeeze().numpy()
        embeddings.append(feat)
        filenames.append(f)

embeddings = np.array(embeddings)

# === Clustering ===
print("🔹 Running KMeans...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

# === UMAP ===
print("🔻 Running UMAP...")
umap_proj = umap.UMAP(
    n_neighbors=min(10, len(filenames) - 1),
    min_dist=0.1,
    random_state=42
).fit_transform(embeddings)

# === Plot ===
plt.figure(figsize=(10, 7))
plt.scatter(
    umap_proj[:, 0], umap_proj[:, 1],
    c=cluster_labels, cmap="tab10", s=60, edgecolors='k'
)
plt.title("🧬 UMAP of Clean Tissue Patches (Clustered)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=300)
print(f"✅ Saved UMAP plot: {OUT_IMG}")

# === Save Metadata ===
df = pd.DataFrame({
    "filename": [os.path.basename(f) for f in filenames],
    "slide_id": [os.path.basename(f).split("_")[0] for f in filenames],
    "cluster": cluster_labels,
    "umap_x": umap_proj[:, 0],
    "umap_y": umap_proj[:, 1],
})
df.to_csv(OUT_CSV, index=False)
print(f"📄 Saved metadata: {OUT_CSV}")

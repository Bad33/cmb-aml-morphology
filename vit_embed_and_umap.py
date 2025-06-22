import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # <- must come before pyplot
import matplotlib.pyplot as plt


from torchvision import transforms
from transformers import AutoImageProcessor, ViTModel
import umap

# === Configuration ===
PATCH_DIR = "sample_patches"   # Folder where patch PNGs are saved
DEVICE = torch.device("cpu")  # ✅ use CPU on macOS to avoid MPS bus error

MODEL_NAME = "owkin/phikon-v2"
N_PATCHES = 100  # Use all or limit for quick test

# === Load Pretrained Phikon-ViT ===
print("🔄 Loading Phikon-ViT...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = ViTModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

# === Load Patches ===
patch_files = [os.path.join(PATCH_DIR, f) for f in os.listdir(PATCH_DIR) if f.endswith(".png")]
patch_files = patch_files[:N_PATCHES]

print(f"📦 Found {len(patch_files)} patches")

# === Preprocess Images ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT expected input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# === Extract Embeddings ===
embeddings = []

with torch.no_grad():
    for f in tqdm(patch_files, desc="🧠 Extracting embeddings"):
        img = Image.open(f).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        feat = outputs.pooler_output.cpu().squeeze().numpy()  # 768-dim
        embeddings.append(feat)

embeddings = np.array(embeddings)

# === UMAP Dimensionality Reduction ===
print("🔻 Running UMAP...")
umap_proj = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(embeddings)

# === Visualize Clusters ===
plt.figure(figsize=(8,6))
plt.scatter(umap_proj[:,0], umap_proj[:,1], s=50, c='dodgerblue', alpha=0.7, edgecolors='k')
plt.title("🧬 Patch Embeddings (UMAP Projection)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig("umap_plot.png", dpi=300)
print("✅ Saved UMAP plot to umap_plot.png")


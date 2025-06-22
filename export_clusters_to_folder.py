import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# === Config ===
CSV_PATH = "clean_patch_clusters.csv"
PATCH_DIR = "clean_patches"
OUT_DIR = "clusters_export"

# === Load cluster info ===
df = pd.read_csv(CSV_PATH)
unique_clusters = sorted(df["cluster"].unique())

# === Create folders ===
for cluster_id in unique_clusters:
    cluster_folder = os.path.join(OUT_DIR, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)

# === Copy patches into cluster folders ===
for _, row in tqdm(df.iterrows(), total=len(df), desc="📦 Exporting"):
    src_path = os.path.join(PATCH_DIR, row["filename"])
    dst_path = os.path.join(OUT_DIR, f"cluster_{row['cluster']}", row["filename"])
    try:
        Image.open(src_path).save(dst_path)
    except Exception as e:
        print(f"⚠️ Failed to save {row['filename']}: {e}")

print(f"✅ Exported clusters to: {OUT_DIR}/cluster_*/")

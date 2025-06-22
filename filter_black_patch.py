import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

# === Config ===
PATCH_DIR = "sample_patches"
CSV_PATH = "patch_clusters.csv"
OUT_CSV = "patch_clusters_clean.csv"
CLEAN_PATCH_DIR = "clean_patches"
BRIGHTNESS_THRESHOLD = 30  # below this = "black patch"

# === Setup output folder ===
os.makedirs(CLEAN_PATCH_DIR, exist_ok=True)

# === Load CSV ===
df = pd.read_csv(CSV_PATH)

# === Measure average brightness for each patch ===
brightness_list = []
for fname in tqdm(df["filename"], desc="📷 Checking brightness"):
    path = os.path.join(PATCH_DIR, fname)
    try:
        img = Image.open(path).convert("L")  # grayscale
        avg_brightness = np.array(img).mean()
    except Exception as e:
        print(f"⚠️ Error reading {fname}: {e}")
        avg_brightness = 0
    brightness_list.append(avg_brightness)

df["brightness"] = brightness_list

# === Filter out dark patches ===
clean_df = df[df["brightness"] > BRIGHTNESS_THRESHOLD].copy()
print(f"✅ Kept {len(clean_df)} of {len(df)} patches")

# === Save filtered CSV ===
clean_df.to_csv(OUT_CSV, index=False)
print(f"📄 Saved: {OUT_CSV}")

# === Optionally copy cleaned patches ===
for fname in tqdm(clean_df["filename"], desc="📦 Copying clean patches"):
    src = os.path.join(PATCH_DIR, fname)
    dst = os.path.join(CLEAN_PATCH_DIR, fname)
    Image.open(src).save(dst)

print(f"✅ Copied clean patches to: {CLEAN_PATCH_DIR}")

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === Config ===
CSV_PATH = "clean_patch_clusters.csv"
OUT_IMG = "clean_umap_by_slide.png"

# === Load Data ===
df = pd.read_csv(CSV_PATH)

# === Unique slide color mapping ===
unique_slides = sorted(df["slide_id"].unique())
palette = sns.color_palette("husl", len(unique_slides))
color_map = {slide_id: palette[i] for i, slide_id in enumerate(unique_slides)}
df["color"] = df["slide_id"].map(color_map)

# === Plot ===
plt.figure(figsize=(10, 7))
for slide_id in unique_slides:
    subset = df[df["slide_id"] == slide_id]
    plt.scatter(
        subset["umap_x"], subset["umap_y"],
        label=slide_id,
        color=color_map[slide_id],
        s=60, edgecolor='k', alpha=0.8
    )

plt.legend(title="Slide ID", loc="best", fontsize=8)
plt.title("🧬 UMAP Colored by Slide ID")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=300)
print(f"✅ Saved slide-colored UMAP plot: {OUT_IMG}")

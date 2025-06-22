import openslide
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
# === Configuration ===
SLIDE_FILES = [
    "CMB-AML/MSB-01723-04-02.svs",
    "CMB-AML/MSB-01723-09-06.svs",
    "CMB-AML/MSB-01723-10-02.svs",
]
PATCH_SIZE = 512  # patch size (in pixels)
LEVEL = 0         # resolution level in WSI (0 is highest)
STRIDE_FACTOR = 4  # stride = PATCH_SIZE * STRIDE_FACTOR
MAX_PATCHES_PER_SLIDE = 100  # limit patches to avoid OOM
SAVE_PATCHES = True         # save a few sample patches to disk
SAVE_DIR = "sample_patches"  # output directory

# === Helper Functions ===
def is_tissue(patch, sat_thresh=15):
    hsv = np.array(Image.fromarray(patch).convert("HSV"))
    saturation = hsv[..., 1]
    return np.mean(saturation) > sat_thresh


def extract_patches(slide_path, max_patches):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.dimensions
    stride = PATCH_SIZE * STRIDE_FACTOR

    patches = []
    coords = []
    count = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            region = slide.read_region((x, y), LEVEL, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
            img_np = np.array(region)

            if is_tissue(img_np):
                patches.append(img_np)
                coords.append((x, y))
                count += 1

                if count >= max_patches:
                    return patches, coords
    return patches, coords


def visualize_patch_locations(slide_path, coords, patch_size=PATCH_SIZE):
    slide = openslide.OpenSlide(slide_path)
    
    # Create thumbnail image for overview
    thumb_size = (2048, 2048)
    thumb = slide.get_thumbnail(thumb_size)
    thumb_w, thumb_h = thumb.size
    full_w, full_h = slide.dimensions

    # Compute scale ratio between full image and thumbnail
    scale_x = thumb_w / full_w
    scale_y = thumb_h / full_h

    # Plot patch locations
    plt.figure(figsize=(10, 10))
    plt.imshow(thumb)
    for (x, y) in coords:
        rect = plt.Rectangle(
            (x * scale_x, y * scale_y),
            patch_size * scale_x,
            patch_size * scale_y,
            edgecolor='red',
            facecolor='none',
            linewidth=1
        )
        plt.gca().add_patch(rect)

    plt.title(f"Patch Locations: {os.path.basename(slide_path)}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# === Main Logic ===
all_patches = []
all_coords = []

if SAVE_PATCHES:
    os.makedirs(SAVE_DIR, exist_ok=True)

for slide_path in SLIDE_FILES:
    print(f"\n📂 Processing slide: {slide_path}")
    patches, coords = extract_patches(slide_path, MAX_PATCHES_PER_SLIDE)
    print(f"✅ Extracted {len(patches)} tissue patches.")

    all_patches.extend(patches)
    all_coords.extend([(slide_path, c) for c in coords])  # retain slide ID

    if SAVE_PATCHES:
        base_name = os.path.splitext(os.path.basename(slide_path))[0]
        for i, patch in enumerate(patches[:5]):
            out_path = os.path.join(SAVE_DIR, f"{base_name}_patch_{i}.png")
            Image.fromarray(patch).save(out_path)
            print(f"🖼️ Saved patch to: {out_path}")

    # 🧭 Visual debug
    visualize_patch_locations(slide_path, coords)





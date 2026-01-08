#!/usr/bin/env python
"""
Extract diverse sample patches from a WSI for reward function testing.

Automatically extracts patches with different characteristics:
- Blank/white patches
- Tissue-rich patches
- Stained patches
- Edge patches (tissue boundary)
- Dense tissue patches
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image
from src.utils.wsi import WSI
import argparse


def is_mostly_white(patch, threshold=240):
    """Check if patch is mostly white/blank."""
    arr = np.array(patch)
    mean_intensity = arr.mean()
    return mean_intensity > threshold


def has_stain(patch):
    """Check if patch has purple/pink staining (H&E)."""
    arr = np.array(patch)
    # H&E: purple (hematoxylin) and pink (eosin)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Purple detection: high blue, low green
    purple_mask = (b > 150) & (g < 100) & (r < 150)
    purple_ratio = purple_mask.sum() / purple_mask.size

    # Pink detection: high red, moderate green, low blue
    pink_mask = (r > 150) & (g > 80) & (g < 180) & (b < 150)
    pink_ratio = pink_mask.sum() / pink_mask.size

    # Need reasonable amount of staining
    return (purple_ratio > 0.05) or (pink_ratio > 0.05)


def tissue_density(patch):
    """Measure tissue density (non-white pixels)."""
    arr = np.array(patch)
    mean_intensity = arr.mean(axis=2)
    tissue_mask = mean_intensity < 220
    return tissue_mask.sum() / tissue_mask.size


def variance_score(patch):
    """Measure visual complexity/variance."""
    arr = np.array(patch).astype(float)
    return arr.std()


def extract_diverse_patches(
    wsi_path, output_dir, level=None, num_samples=20, samples_per_type=5
):
    """
    Extract diverse patches from WSI covering different tissue types.

    Args:
        wsi_path: Path to WSI file
        output_dir: Directory to save sample patches
        level: WSI pyramid level (None = uses multiple levels)
        num_samples: Number of random patches to sample before filtering
        samples_per_type: Number of samples to extract for each patch type
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading WSI: {wsi_path}")
    wsi = WSI(wsi_path, patch_size=512)

    print(f"WSI has {wsi.max_level + 1} levels (0 to {wsi.max_level})")

    # Sample patches from ALL levels for diversity
    candidates = []
    print(f"Sampling candidate patches from all levels...")

    for current_level in range(wsi.max_level + 1):
        width, height = wsi.levels_info[current_level]["size"]

        # Sample more patches per level to ensure we have enough candidates
        samples_this_level = (
            num_samples * 2 if current_level == wsi.max_level else num_samples
        )

        step_x = max(width // 12, wsi.patch_size)  # Denser grid
        step_y = max(height // 12, wsi.patch_size)

        sampled_this_level = 0
        for y in range(0, height - wsi.patch_size, step_y):
            for x in range(0, width - wsi.patch_size, step_x):
                if sampled_this_level >= samples_this_level:
                    break

                try:
                    patch = wsi.get_patch(current_level, x, y)

                    # Compute characteristics
                    white = is_mostly_white(patch, threshold=245)  # Stricter for blank
                    stain = has_stain(patch)
                    density = tissue_density(patch)
                    variance = variance_score(patch)

                    candidates.append(
                        {
                            "level": current_level,
                            "x": x,
                            "y": y,
                            "patch": patch,
                            "white": white,
                            "stain": stain,
                            "density": density,
                            "variance": variance,
                        }
                    )
                    sampled_this_level += 1

                except Exception as e:
                    continue

            if sampled_this_level >= samples_this_level:
                break

        print(f"  Level {current_level}: sampled {sampled_this_level} patches")

    print(f"Total collected: {len(candidates)} candidate patches")

    # Select exactly 5 patches for each required type: dense, sparse, stained, blank
    samples = {}

    # 1. Dense: High variance + high density (really complex tissue)
    dense_patches = [c for c in candidates if c["variance"] > 50 and c["density"] > 0.4]
    if len(dense_patches) < samples_per_type:
        # Relax criteria progressively
        dense_patches = [
            c for c in candidates if c["variance"] > 40 and c["density"] > 0.3
        ]
    if len(dense_patches) < samples_per_type:
        dense_patches = [c for c in candidates if not c["white"]]
    dense_patches.sort(key=lambda x: (-x["variance"], -x["density"]))
    for i in range(min(samples_per_type, len(dense_patches))):
        samples[f"dense_{i+1}"] = dense_patches[i]
    print(
        f"✓ Found {len([k for k in samples if k.startswith('dense')])}/5 dense patches"
    )

    # 2. Sparse: Low density tissue
    low_density = [
        c for c in candidates if not c["white"] and 0.05 < c["density"] < 0.3
    ]
    if len(low_density) < samples_per_type:
        low_density = [c for c in candidates if not c["white"] and c["density"] < 0.4]
    if len(low_density) < samples_per_type:
        low_density = [c for c in candidates if not c["white"]]
    low_density.sort(key=lambda x: x["density"])
    for i in range(min(samples_per_type, len(low_density))):
        samples[f"sparse_{i+1}"] = low_density[i]
    print(
        f"✓ Found {len([k for k in samples if k.startswith('sparse')])}/5 sparse patches"
    )

    # 3. Stained: Small spot of staining on mostly white background
    # Look for patches with low density but some variance (indicates spots)
    stained_candidates = [
        c
        for c in candidates
        if 0.01 < c["density"] < 0.15 and c["variance"] > 10 and c["variance"] < 50
    ]
    if len(stained_candidates) < samples_per_type:
        # Fallback: any patch with very low density but not completely white
        stained_candidates = [
            c for c in candidates if 0.005 < c["density"] < 0.2 and not c["white"]
        ]
    # Sort by density - prefer patches with just a small amount of content
    stained_candidates.sort(key=lambda x: x["density"])
    for i in range(min(samples_per_type, len(stained_candidates))):
        samples[f"stained_{i+1}"] = stained_candidates[i]
    print(
        f"✓ Found {len([k for k in samples if k.startswith('stained')])}/5 stained patches"
    )

    # 4. Blank: Very white, minimal content
    white_patches = [c for c in candidates if c["white"] and c["density"] < 0.05]
    if len(white_patches) < samples_per_type:
        white_patches = [c for c in candidates if c["white"]]
    if len(white_patches) < samples_per_type:
        # Just get the whitest patches
        all_by_whiteness = sorted(candidates, key=lambda x: x["density"])
        white_patches = all_by_whiteness[: samples_per_type * 2]
    white_patches.sort(key=lambda x: x["density"])  # Whitest first
    for i in range(min(samples_per_type, len(white_patches))):
        samples[f"blank_{i+1}"] = white_patches[i]
    print(
        f"✓ Found {len([k for k in samples if k.startswith('blank')])}/5 blank patches"
    )

    # Save patches
    print(f"\nSaving {len(samples)} sample patches to {output_dir}/")

    # Also save patch info as JSON for programmatic access
    import json

    patch_info = {}

    for name, info in samples.items():
        filepath = output_dir / f"{name}.png"
        info["patch"].save(filepath)
        print(
            f"  {name}: level={info['level']}, density={info['density']:.2f}, variance={info['variance']:.1f}"
        )

        # Store info (exclude PIL image object)
        patch_info[name] = {
            "filename": f"{name}.png",
            "level": info["level"],
            "x": info["x"],
            "y": info["y"],
            "density": float(info["density"]),
            "variance": float(info["variance"]),
            "is_white": bool(info["white"]),
            "has_stain": bool(info["stain"]),
        }

    # Save as JSON
    info_path = output_dir / "patch_info.json"
    with open(info_path, "w") as f:
        json.dump(patch_info, f, indent=2)
    print(f"\nPatch info saved to: {info_path}")

    # Save metadata
    metadata_path = output_dir / "metadata.txt"
    with open(metadata_path, "w") as f:
        f.write(f"WSI: {wsi_path}\n")
        f.write(f"Patch size: {wsi.patch_size}\n")
        f.write(f"Levels: 0 to {wsi.max_level}\n\n")
        for name, info in samples.items():
            f.write(f"{name}:\n")
            f.write(f"  Level: {info['level']}\n")
            f.write(f"  Position: ({info['x']}, {info['y']})\n")
            f.write(f"  Density: {info['density']:.3f}\n")
            f.write(f"  Variance: {info['variance']:.1f}\n")
            f.write(f"  Mostly white: {info['white']}\n")
            f.write(f"  Has stain: {info['stain']}\n\n")

    print(f"\nMetadata saved to {metadata_path}")
    print(f"\n✓ Done! Extracted {len(samples)} diverse sample patches")
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract diverse sample patches from WSI"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="./data/example_images/test_img_1.svs",
        help="Path to WSI file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./src/reward_distinction/img",
        help="Output directory for sample patches",
    )
    parser.add_argument(
        "--level", type=int, default=None, help="WSI pyramid level (default: max level)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of candidate patches to sample",
    )
    parser.add_argument(
        "--samples-per-type",
        type=int,
        default=5,
        help="Number of samples to extract for each patch type",
    )

    args = parser.parse_args()
    extract_diverse_patches(
        args.image, args.output, args.level, args.num_samples, args.samples_per_type
    )

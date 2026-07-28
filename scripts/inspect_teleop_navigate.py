"""Inspect teleop_navigate field statistics in lerobot_0510 dataset."""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path("/cpfs_infra/shared/xiaoxinyu/DiT4DiT/datasets/lerobot_0510/data/chunk-000")


def main():
    parquet_files = sorted(DATA_DIR.glob("episode_*.parquet"))
    print(f"Total episodes: {len(parquet_files)}")

    all_vals = []
    nonzero_counts = [0, 0, 0]
    total_frames = 0

    for pf in tqdm(parquet_files, desc="Loading"):
        df = pd.read_parquet(pf, columns=["teleop_navigate"])
        arr = np.stack(df["teleop_navigate"].values)  # (T, 3)
        all_vals.append(arr)
        total_frames += len(arr)
        for dim in range(3):
            nonzero_counts[dim] += int(np.count_nonzero(arr[:, dim]))

    all_vals = np.concatenate(all_vals, axis=0)  # (N, 3)
    print(f"\nTotal frames: {total_frames:,}")
    print(f"Shape: {all_vals.shape}")

    print("\n" + "=" * 60)
    print("teleop_navigate statistics (full dataset)")
    print("=" * 60)

    dim_names = ["dim0", "dim1", "dim2"]
    for i, name in enumerate(dim_names):
        col = all_vals[:, i]
        print(f"\n--- {name} ---")
        print(f"  min:    {col.min():.6f}")
        print(f"  max:    {col.max():.6f}")
        print(f"  mean:   {col.mean():.6f}")
        print(f"  std:    {col.std():.6f}")
        print(f"  median: {np.median(col):.6f}")
        print(f"  nonzero ratio (|x|>1e-6): {nonzero_counts[i] / total_frames * 100:.2f}%")
        for p in [1, 5, 25, 50, 75, 95, 99]:
            print(f"  p{p:2d}: {np.percentile(col, p):.6f}")

    print("\n" + "=" * 60)
    print("Per-frame L2 norm: sqrt(dim0^2 + dim1^2 + dim2^2)")
    print("=" * 60)
    norms = np.linalg.norm(all_vals, axis=1)
    print(f"  min:    {norms.min():.6f}")
    print(f"  max:    {norms.max():.6f}")
    print(f"  mean:   {norms.mean():.6f}")
    print(f"  std:    {norms.std():.6f}")
    print(f"  median: {np.median(norms):.6f}")
    print(f"  all-zero ratio (norm<1e-6): {(norms < 1e-6).sum() / total_frames * 100:.2f}%")

    mask = norms > 1e-6
    if mask.any():
        print(f"\nNon-zero sample examples (first 10):")
        idx = np.where(mask)[0][:10]
        for i in idx:
            print(f"  frame {i}: {all_vals[i]}")


if __name__ == "__main__":
    main()

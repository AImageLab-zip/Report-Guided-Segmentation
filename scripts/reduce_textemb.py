"""
============================
Removes the top-1 principal component from pre-computed report embeddings.

Expected input structure:
    <input_folder>/
        clinical/   *.npz
        generated/  *.npz
        concat/     *.npz

Output structure (mirrors input):
    <output_folder>/
        clinical/   *.npz
        generated/  *.npz
        concat/     *.npz

Each .npz file must contain an "embedding" key with a 1-D float32 array.

Usage:
    python reduce_report_embeddings.py \
        --input_folder  /path/to/report_RG_biobert \
        --output_folder /path/to/report_RG_biobert_reduced

    # Process only specific modes:
    python reduce_report_embeddings.py \
        --input_folder  /path/to/report_RG_biobert \
        --output_folder /path/to/report_RG_biobert_reduced \
        --modes clinical generated
"""

import argparse
import glob
import os
import pathlib

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# PC removal
# ---------------------------------------------------------------------------

def remove_top1_pc(X: np.ndarray) -> np.ndarray:
    """
    Remove the top-1 principal component from a matrix of embeddings.

    For each embedding x_i, computes:
        x_i_reduced = x_i - (x_i · u) * u
    where u is the first principal component of X (unit vector).

    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Matrix of N embeddings of dimension D.

    Returns
    -------
    X_reduced : np.ndarray, shape (N, D)
        Embeddings with the top PC projected out, as float32.
    """
    pca = PCA(n_components=1)
    pca.fit(X)
    u = pca.components_  # shape (1, D)

    # Project out: X_reduced = X - X @ u.T @ u
    projections = X @ u.T          # (N, 1)
    X_reduced = X - projections @ u  # (N, D)

    return X_reduced.astype(np.float32)


# ---------------------------------------------------------------------------
# Per-mode processing
# ---------------------------------------------------------------------------

def process_mode(mode: str, input_folder: str, output_folder: str, overwrite: bool) -> None:
    # mode="." means flat layout — use the folders directly, no subfolder appended.
    input_dir  = input_folder  if mode == "." else os.path.join(input_folder,  mode)
    output_dir = output_folder if mode == "." else os.path.join(output_folder, mode)

    if not os.path.isdir(input_dir):
        print(f"[WARN] mode='{mode}': input directory not found at '{input_dir}', skipping.")
        return

    files = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not files:
        print(f"[WARN] mode='{mode}': no .npz files found in '{input_dir}', skipping.")
        return

    # Check for existing outputs if not overwriting
    if not overwrite:
        existing = [f for f in files if os.path.exists(
            os.path.join(output_dir, os.path.basename(f))
        )]
        if existing:
            print(
                f"[WARN] mode='{mode}': {len(existing)}/{len(files)} output files already exist. "
                f"Pass --overwrite to replace them."
            )

    print(f"[INFO] mode='{mode}': loading {len(files)} embeddings from '{input_dir}'")

    # Load all embeddings in deterministic (sorted) order
    embeddings = [np.load(f)["embedding"] for f in files]
    X = np.stack(embeddings)  # (N, D)
    print(f"[INFO] mode='{mode}': embedding matrix shape = {X.shape}")

    # Fit PCA and remove top-1 PC
    X_reduced = remove_top1_pc(X)
    print(f"[INFO] mode='{mode}': PC removal done. Output shape = {X_reduced.shape}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    skipped = 0
    for file_path, emb in zip(files, X_reduced):
        out_path = os.path.join(output_dir, os.path.basename(file_path))
        if (not overwrite) and os.path.exists(out_path):
            skipped += 1
            continue
        np.savez_compressed(out_path, embedding=emb)
        written += 1

    print(
        f"[DONE] mode='{mode}': written={written}, skipped_existing={skipped}, "
        f"output_dir='{output_dir}'"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove top-1 principal component from report embeddings."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Root folder containing 'clinical', 'generated', 'concat' subfolders.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Root folder where reduced embeddings will be saved (mirrors input structure).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["clinical", "generated", "concat"],
        help="Which report modes to process. Default: clinical generated concat.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def is_flat_layout(folder: str) -> bool:
    """
    Returns True if `folder` contains .npz files directly (flat layout).
    Returns False if it contains subfolders (multi-mode layout).
    A folder with both is treated as flat — the .npz files take priority.
    """
    return any(pathlib.Path(folder).glob("*.npz"))


def main() -> None:
    args = parse_args()

    print(f"[INFO] input_folder  = '{args.input_folder}'")
    print(f"[INFO] output_folder = '{args.output_folder}'")

    os.makedirs(args.output_folder, exist_ok=True)

    if is_flat_layout(args.input_folder):
        # ── Flat layout: .npz files sit directly in input_folder ─────────────
        print("[INFO] layout = flat (no subfolders)")
        print()
        process_mode(
            mode=".",           # "." means use the folder itself, no subfolder appended
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            overwrite=args.overwrite,
        )
    else:
        # ── Multi-mode layout: clinical/ generated/ concat/ subfolders ────────
        print(f"[INFO] layout = multi-mode, modes = {args.modes}")
        for mode in args.modes:
            print()
            process_mode(
                mode=mode,
                input_folder=args.input_folder,
                output_folder=args.output_folder,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
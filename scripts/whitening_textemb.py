"""
============================
Applies Whitening-k to pre-computed report embeddings organised in
report-mode subfolders (clinical / generated / concat).

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

Whitening params are saved per-mode under:
    <params_folder>/
        clinical.npz
        generated.npz
        concat.npz

Algorithm (Whitening-k):
    'Whitening Sentence Representations for Better Semantics and Faster Retrieval'
    x_tilde = (x - mu) @ W
    where W = (U * sqrt(Lambda^{-1}))[:, :k]  and  SVD(Sigma) = U Lambda U^T

Usage:
    python whiten_report_embeddings.py \
        --input_folder  /path/to/report_RG_biobert \
        --output_folder /path/to/report_RG_biobert_whitened \
        --params_folder /path/to/report_RG_biobert_whitened/params \
        --k 256

    # Process only specific modes:
    python whiten_report_embeddings.py \
        --input_folder  /path/to/report_RG_biobert \
        --output_folder /path/to/report_RG_biobert_whitened \
        --params_folder /path/to/report_RG_biobert_whitened/params \
        --modes clinical generated
"""

import argparse
import os
import pathlib

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Whitening-k
# ---------------------------------------------------------------------------

@dataclass
class WhiteningK:
    """
    Whitening-k as in:
    'Whitening Sentence Representations for Better Semantics and Faster Retrieval'
    Algorithm 1: x_tilde = (x - mu) @ W
    where W = (U * sqrt(Lambda^{-1}))[:, :k] and SVD(Sigma) = U Lambda U^T
    """
    k: int | None = None
    eps: float = 1e-12
    mu_: np.ndarray | None = None
    W_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "WhiteningK":
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [N, D], got {X.shape}")

        X64 = X.astype(np.float64, copy=False)
        N, D = X64.shape
        k = D if self.k is None else int(self.k)
        if not (1 <= k <= D):
            raise ValueError(f"k must be in [1, {D}], got {k}")

        # (1) Mean
        mu = X64.mean(axis=0)  # [D]

        # (2) Covariance with 1/N scaling
        Xc = X64 - mu
        Sigma = (Xc.T @ Xc) / N  # [D, D]

        # (3) SVD of covariance
        U, s, _ = np.linalg.svd(Sigma, full_matrices=True)

        # (4) W = (U * sqrt(s^{-1}))[:, :k]
        inv_sqrt = 1.0 / np.sqrt(s + self.eps)       # [D]
        W_full = U * inv_sqrt[np.newaxis, :]          # scale columns of U
        W = W_full[:, :k]                             # [D, k]

        self.mu_ = mu.astype(np.float32)
        self.W_ = W.astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.W_ is None:
            raise RuntimeError("Call fit() before transform().")
        if X.ndim != 2:
            raise ValueError(f"Expected X shape [N, D], got {X.shape}")
        X32 = X.astype(np.float32, copy=False)
        return (X32 - self.mu_) @ self.W_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        if self.mu_ is None or self.W_ is None:
            raise RuntimeError("Nothing to save; call fit() first.")
        np.savez(path, mu=self.mu_, W=self.W_, k=np.array([self.W_.shape[1]], dtype=np.int32))

    @staticmethod
    def load(path: str) -> "WhiteningK":
        d = np.load(path)
        obj = WhiteningK(k=int(d["k"][0]))
        obj.mu_ = d["mu"]
        obj.W_ = d["W"]
        return obj


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_npz_dir(in_dir: str, emb_key: str = "embedding") -> tuple[np.ndarray, list[str]]:
    """
    Load all .npz files from `in_dir`, each assumed to hold a single 1-D
    embedding under `emb_key`.  Returns:
        X     – float32 array of shape [N, D], rows ordered by sorted filename
        names – list of N bare stems (e.g. "BraTS-GLI-00000-000") in the same order
    """
    paths = sorted(pathlib.Path(in_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {in_dir!r}")

    embeddings, names = [], []
    for p in paths:
        data = np.load(p)
        if emb_key not in data:
            raise KeyError(f"{p.name}: expected key {emb_key!r}, got {list(data.keys())}")
        emb = data[emb_key]
        if emb.ndim != 1:
            raise ValueError(f"{p.name}: embedding must be 1-D, got shape {emb.shape}")
        embeddings.append(emb.astype(np.float32))
        names.append(p.stem)

    X = np.stack(embeddings, axis=0)   # [N, D]
    return X, names


def save_npz_dir(out_dir: str, Xw: np.ndarray, names: list[str], emb_key: str = "embedding") -> None:
    """
    Save each row of `Xw` as an individual .npz file named ``<stem>.npz``
    under `out_dir`, mirroring the layout of the input directory.
    """
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for name, vec in zip(names, Xw):
        np.savez_compressed(out_path / f"{name}.npz", **{emb_key: vec})


# ---------------------------------------------------------------------------
# Per-mode processing
# ---------------------------------------------------------------------------

def process_mode(
    mode: str,
    input_folder: str,
    output_folder: str,
    params_folder: str,
    k: int | None,
    eps: float,
    emb_key: str,
    overwrite: bool,
) -> None:
    # mode="." means flat layout — use the folders directly, no subfolder appended.
    input_dir   = input_folder  if mode == "." else os.path.join(input_folder,  mode)
    output_dir  = output_folder if mode == "." else os.path.join(output_folder, mode)
    params_path = os.path.join(params_folder, "params.npz" if mode == "." else f"{mode}.npz")

    if not os.path.isdir(input_dir):
        print(f"[WARN] mode='{mode}': input directory not found at '{input_dir}', skipping.")
        return

    # Skip if outputs already exist and overwrite is off
    if (not overwrite) and os.path.isdir(output_dir) and any(pathlib.Path(output_dir).glob("*.npz")):
        print(
            f"[WARN] mode='{mode}': output directory '{output_dir}' already contains .npz files. "
            f"Pass --overwrite to replace them. Skipping."
        )
        return

    # Load
    print(f"[INFO] mode='{mode}': loading embeddings from '{input_dir}'")
    X, names = load_npz_dir(input_dir, emb_key=emb_key)
    print(f"[INFO] mode='{mode}': loaded {len(names)} embeddings, shape={X.shape}")

    # Fit + transform
    wk = WhiteningK(k=k, eps=eps)
    Xw = wk.fit_transform(X)
    print(f"[INFO] mode='{mode}': whitening done. Output shape={Xw.shape}")

    # Save params
    os.makedirs(params_folder, exist_ok=True)
    wk.save(params_path)
    print(f"[INFO] mode='{mode}': params saved to '{params_path}'  (mu={wk.mu_.shape}, W={wk.W_.shape})")

    # Save whitened embeddings
    save_npz_dir(output_dir, Xw.astype(np.float32), names, emb_key=emb_key)
    print(f"[DONE] mode='{mode}': {len(names)} whitened embeddings saved to '{output_dir}'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply Whitening-k to report embeddings organised in mode subfolders."
    )
    parser.add_argument(
        "--input_folder",
        required=True,
        help="Root folder containing 'clinical', 'generated', 'concat' subfolders.",
    )
    parser.add_argument(
        "--output_folder",
        required=True,
        help="Root folder where whitened embeddings will be saved (mirrors input structure).",
    )
    parser.add_argument(
        "--params_folder",
        required=True,
        help="Folder where per-mode whitening params (.npz) will be saved.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["clinical", "generated", "concat"],
        help="Which report modes to process. Default: clinical generated concat.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Whitening-k output dimension (default: keep full D).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-12,
        help="Numerical stability epsilon for SVD inversion (default: 1e-12).",
    )
    parser.add_argument(
        "--emb_key",
        default="embedding",
        help="Key inside each .npz file (default: 'embedding').",
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
    print(f"[INFO] params_folder = '{args.params_folder}'")
    print(f"[INFO] k             = {args.k if args.k is not None else 'full D'}")

    os.makedirs(args.output_folder, exist_ok=True)

    if is_flat_layout(args.input_folder):
        # ── Flat layout: .npz files sit directly in input_folder ─────────────
        print("[INFO] layout = flat (no subfolders)")
        print()
        process_mode(
            mode=".",
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            params_folder=args.params_folder,
            k=args.k,
            eps=args.eps,
            emb_key=args.emb_key,
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
                params_folder=args.params_folder,
                k=args.k,
                eps=args.eps,
                emb_key=args.emb_key,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import argparse
import os
import numpy as np
from dataclasses import dataclass


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

        # (2) Covariance with 1/N scaling (as in the paper)
        Xc = X64 - mu
        Sigma = (Xc.T @ Xc) / N  # [D, D]

        # (3) SVD of covariance
        U, s, _ = np.linalg.svd(Sigma, full_matrices=True)

        # (4) W = (U * sqrt(s^{-1}))[:, :k]
        inv_sqrt = 1.0 / np.sqrt(s + self.eps)       # [D]
        W_full = U * inv_sqrt[np.newaxis, :]         # scale columns of U
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_npy", required=True, help="Path to input embeddings .npy (shape [N, D])")
    ap.add_argument("--out_npy", required=True, help="Path to output whitened embeddings .npy (shape [N, k])")
    ap.add_argument("--out_params", required=True, help="Path to output whitening params .npz (mu, W, k)")
    ap.add_argument("--k", type=int, default=None, help="Whitening-k dimension (default: keep full D)")
    ap.add_argument("--eps", type=float, default=1e-12, help="Numerical stability epsilon")
    args = ap.parse_args()

    X = np.load(args.in_npy)
    if X.ndim != 2:
        raise ValueError(f"{args.in_npy} must contain a 2D array [N, D], got {X.shape}")

    wk = WhiteningK(k=args.k, eps=args.eps)
    Xw = wk.fit_transform(X)

    os.makedirs(os.path.dirname(args.out_npy) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_params) or ".", exist_ok=True)

    np.save(args.out_npy, Xw.astype(np.float32))
    wk.save(args.out_params)

    print(f"Loaded:        {args.in_npy} shape={X.shape} dtype={X.dtype}")
    print(f"Whitened:      {args.out_npy} shape={Xw.shape} dtype=float32")
    print(f"Params saved:  {args.out_params} (mu: {wk.mu_.shape}, W: {wk.W_.shape})")


if __name__ == "__main__":
    main()
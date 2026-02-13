import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from datasets.DatasetFactory import DatasetFactory
from utils.textemb_BioBERT import hash_report, mean_pool, precompute_unique_report_embeddings
from transforms.TransformsFactory import TransformsFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute report text embeddings.")
    parser.add_argument(
        "--mode",
        default="brats3d",
        choices=["brats3d", "qatacov"],
        help="Extraction mode. Use 'brats3d' to generate per-case .npz files for BraTS3DText.",
    )
    parser.add_argument(
        "--model_name",
        default="dmis-lab/biobert-base-cased-v1.1",
        help="Model card to be used for computing embeddings. Default: BioBERT"
    )
    parser.add_argument(
        "--config",
        default="./config/config_brats3d.json",
        help="Path to the config JSON file.",
    )
    parser.add_argument(
        "--save_directory",
        default=None,
        help="Directory where text embeddings will be saved (QaTaCov mode only).",
    )
    parser.add_argument(
        "--reports_path",
        default=None,
        help=(
            "BraTS reports source. Can be either: (1) a directory of <case_id>.txt files, "
            "or (2) a JSON file mapping case_id -> report text."
        ),
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="Optional split file override for BraTS. If omitted, uses config.dataset.split_file.",
    )
    parser.add_argument(
        "--report_folder",
        default="rep_RG",
        help=(
            "BraTS output folder name under dataset root where .npz embeddings are saved. "
            "Must match config.dataset.report_folder used by BraTS3DText."
        ),
    )
    parser.add_argument(
        "--pooling",
        default="mean",
        choices=["mean", "max", "cls"],
        help="Pooling strategy for text embeddings.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum token length for text embeddings.",
    )
    parser.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Disable loading Hugging Face models from safetensors weights.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing embeddings if present.",
    )
    return parser.parse_args()


def _brats_case_id_from_image_path(image_path: str) -> str:
    # e.g. BraTS-GLI-00000-000_vol.nii.gz -> BraTS-GLI-00000-000
    return Path(image_path).stem.removesuffix("_vol.nii")


def _load_brats_case_ids(dataset_root: str, split_file: str) -> List[str]:
    vol_dir = os.path.join(dataset_root, "vol")
    images = glob(os.path.join(vol_dir, "*.nii.gz"))
    images_by_id = {_brats_case_id_from_image_path(p): p for p in images}

    with open(split_file, "r") as f:
        split_data = json.load(f)

    ordered_ids: List[str] = []
    for split in ("train", "val", "test"):
        for case_id in split_data.get(split, []):
            if case_id in images_by_id:
                ordered_ids.append(case_id)

    return ordered_ids


def _load_brats_reports_map(reports_path: str) -> Dict[str, str]:
    if os.path.isdir(reports_path):
        mapping: Dict[str, str] = {}
        for txt_path in glob(os.path.join(reports_path, "*.txt")):
            case_id = Path(txt_path).stem
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                mapping[case_id] = f.read()
        return mapping

    if reports_path.lower().endswith(".json"):
        with open(reports_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("BraTS reports JSON must be a dict mapping case_id -> report text.")
        return {str(k): str(v) for k, v in data.items()}

    raise ValueError("--reports_path must be a directory of .txt files or a JSON mapping file.")


@torch.no_grad()
def _encode_unique_reports(
    reports: List[str],
    model_name: str,
    max_length: int,
    pooling: str,
    device: str,
    use_safetensors: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModel.from_pretrained(model_name, use_safetensors=use_safetensors)
    except ValueError as exc:
        raise ValueError(
            "Failed to load model weights. If you are on torch<2.6, set "
            "use_safetensors=True or upgrade torch to >=2.6."
        ) from exc

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    bs = 64
    all_emb = []
    token_counts = []
    raw_token_counts = []
    num_truncated_reports = 0
    for i in range(0, len(reports), bs):
        chunk = reports[i : i + bs]

        # Raw lengths (without truncation) to estimate truncation statistics.
        raw_tok = tokenizer(
            chunk,
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
        raw_chunk_counts = [len(ids) for ids in raw_tok["input_ids"]]
        raw_token_counts.extend(raw_chunk_counts)

        tok = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        chunk_token_counts = tok["attention_mask"].sum(dim=1).detach().cpu().numpy().astype(np.int32)
        token_counts.extend(chunk_token_counts.tolist())
        num_truncated_reports += int(np.sum(np.array(raw_chunk_counts) > max_length))

        out = model(**tok)
        last_hidden = out.last_hidden_state  # [b, L, H]

        if pooling == "cls":
            pooled = last_hidden[:, 0, :]
        elif pooling == "mean":
            pooled = mean_pool(last_hidden, tok["attention_mask"])
        elif pooling == "max":
            expanded_mask = tok["attention_mask"].unsqueeze(-1).bool()
            masked = last_hidden.masked_fill(~expanded_mask, torch.finfo(last_hidden.dtype).min)
            pooled = masked.max(dim=1).values
        else:
            raise ValueError("pooling must be one of: 'mean', 'max', 'cls'")

        all_emb.append(pooled.detach().cpu().numpy().astype(np.float32))

    reports_emb = np.concatenate(all_emb, axis=0)

    token_counts_np = np.asarray(token_counts, dtype=np.int32)
    raw_token_counts_np = np.asarray(raw_token_counts, dtype=np.int32)
    token_stats = {
        "avg_num_tokens": float(np.mean(token_counts_np)) if token_counts_np.size > 0 else 0.0,
        "median_num_tokens": float(np.median(token_counts_np)) if token_counts_np.size > 0 else 0.0,
        "min_num_tokens": int(np.min(token_counts_np)) if token_counts_np.size > 0 else 0,
        "max_num_tokens": int(np.max(token_counts_np)) if token_counts_np.size > 0 else 0,
        "std_num_tokens": float(np.std(token_counts_np)) if token_counts_np.size > 0 else 0.0,
        "total_num_tokens": int(np.sum(token_counts_np)) if token_counts_np.size > 0 else 0,
        "avg_raw_num_tokens": float(np.mean(raw_token_counts_np)) if raw_token_counts_np.size > 0 else 0.0,
        "num_truncated_reports": int(num_truncated_reports),
        "pct_truncated_reports": (
            float(100.0 * num_truncated_reports / len(reports)) if len(reports) > 0 else 0.0
        ),
    }

    return reports_emb, token_stats


def _run_brats3d_extraction(args: argparse.Namespace, config: Config) -> None:
    dataset_root = config.dataset["path"]
    split_file = args.split_file or config.dataset.get("split_file")
    if not split_file:
        raise ValueError("BraTS mode requires a split file (pass --split_file or set config.dataset.split_file).")

    reports_path = args.reports_path or config.dataset.get("reports_path")
    if not reports_path:
        raise ValueError(
            "BraTS mode requires --reports_path (or config.dataset.reports_path) to locate text reports."
        )

    report_folder = args.report_folder or config.dataset.get("report_folder", "rep_RG")
    output_dir = os.path.join(dataset_root, report_folder)
    os.makedirs(output_dir, exist_ok=True)

    case_ids = _load_brats_case_ids(dataset_root, split_file)
    reports_map = _load_brats_reports_map(reports_path)

    unique_reports: List[str] = []
    report_hash_to_idx: Dict[str, int] = {}
    case_to_report_idx: Dict[str, int] = {}
    missing_reports_count = 0
    empty_reports_count = 0

    for case_id in case_ids:
        txt = reports_map.get(case_id)
        if txt is None:
            missing_reports_count += 1
            continue
        txt = txt.strip()
        if not txt:
            empty_reports_count += 1
            continue

        h = hash_report(txt)
        if h not in report_hash_to_idx:
            report_hash_to_idx[h] = len(unique_reports)
            unique_reports.append(txt)
        case_to_report_idx[case_id] = report_hash_to_idx[h]

    if len(unique_reports) == 0:
        raise ValueError("No valid BraTS reports found for selected split IDs.")

    print(f"[INFO] BraTS cases in split (with images): {len(case_ids)}")
    print(f"[INFO] BraTS cases with available non-empty report: {len(case_to_report_idx)}")
    print(f"[INFO] Unique reports to encode: {len(unique_reports)}")

    reports_emb, token_stats = _encode_unique_reports(
        reports=unique_reports,
        model_name=args.model_name,
        max_length=args.max_len,
        pooling=args.pooling,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_safetensors=not args.no_safetensors,
    )

    words_per_report = [len(r.split()) for r in unique_reports]
    chars_per_report = [len(r) for r in unique_reports]

    written = 0
    skipped_existing = 0
    for case_id, emb_idx in case_to_report_idx.items():
        out_path = os.path.join(output_dir, f"{case_id}.npz")
        if (not args.overwrite) and os.path.exists(out_path):
            skipped_existing += 1
            continue
        np.savez_compressed(out_path, embedding=reports_emb[emb_idx])
        written += 1

    meta = {
        "model_name": args.model_name,
        "max_length": int(args.max_len),
        "pooling": args.pooling,
        "embedding_dim": int(reports_emb.shape[1]),
        "num_cases_in_split": int(len(case_ids)),
        "num_cases_with_report": int(len(case_to_report_idx)),
        "num_cases_missing_report": int(missing_reports_count),
        "num_cases_empty_report": int(empty_reports_count),
        "num_unique_reports": int(len(unique_reports)),
        "avg_num_words": float(np.mean(words_per_report)) if words_per_report else 0.0,
        "avg_num_characters": float(np.mean(chars_per_report)) if chars_per_report else 0.0,
        "median_num_characters": float(np.median(chars_per_report)) if chars_per_report else 0.0,
        "output_dir": output_dir,
        "report_folder": report_folder,
        "report_hash_to_idx": report_hash_to_idx,
        **token_stats,
    }

    with open(os.path.join(output_dir, "avg_num_tokens.txt"), "w") as f:
        f.write(str(token_stats["avg_num_tokens"]))
    with open(os.path.join(output_dir, "total_num_tokens.txt"), "w") as f:
        f.write(str(token_stats["total_num_tokens"]))
    with open(os.path.join(output_dir, "token_stats.json"), "w") as f:
        json.dump(token_stats, f)

    with open(os.path.join(output_dir, "reports_meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(output_dir, "case_to_report_idx.json"), "w") as f:
        json.dump(case_to_report_idx, f)

    print(
        "[DONE] BraTS embeddings saved. "
        f"written={written}, skipped_existing={skipped_existing}, output_dir={output_dir}"
    )


def _run_qatacov_extraction(args: argparse.Namespace, config: Config) -> None:
    if not args.save_directory:
        raise ValueError("QaTaCov mode requires --save_directory.")

    dataset_factory = DatasetFactory()

    transforms_config_path = config.dataset["transforms"]
    with open(transforms_config_path, "r") as f:
        transforms_config = json.load(f)

    augmentation_transforms = TransformsFactory.create_instance(
        transforms_config.get("preprocessing", []), backend="monai"
    )

    if augmentation_transforms:
        train_transforms = augmentation_transforms
        test_transforms = None
    else:
        train_transforms = None
        test_transforms = None

    qatacov = dataset_factory.create_instance(
        config=config,
        validation=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms,
    )

    train_loader = qatacov.get_loader("train")

    precompute_unique_report_embeddings(
        dataloader=train_loader,
        save_directory=args.save_directory,
        model_name=args.model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_len,
        pooling=args.pooling,
        use_safetensors=not args.no_safetensors,
        overwrite=args.overwrite,
    )


def main() -> None:
    args = parse_args()
    config = Config(args.config)
    print(config)

    if args.mode == "brats3d":
        _run_brats3d_extraction(args, config)
    elif args.mode == "qatacov":
        _run_qatacov_extraction(args, config)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()

import os, json, hashlib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def image_stem(image_path: str) -> str:
    return os.path.splitext(os.path.basename(image_path))[0]


def hash_report(text: str) -> str:
    # Keep identical behaviour to your BioBERT util
    t = (text or "").strip().lower()
    t = " ".join(t.split())
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


@torch.no_grad()
def precompute_unique_report_embeddings_medsiglip(
    dataloader,
    save_directory: str,
    model_name: str = "google/medsiglip-448",
    max_length: int = 64,          # MedSigLIP: max 64 text tokens
    device: str = "cuda",
    overwrite: bool = False,
    use_safetensors: bool = True,  # kept for compatibility; may or may not be used depending on HF version
    batch_size: int = 64,
    normalize: bool = True,
    use_amp: bool = True,
):
    os.makedirs(save_directory, exist_ok=True)
    emb_path = os.path.join(save_directory, "reports_emb.npy")
    meta_path = os.path.join(save_directory, "reports_meta.json")
    map_path  = os.path.join(save_directory, "image_to_report_idx.json")

    if (not overwrite) and os.path.exists(emb_path) and os.path.exists(meta_path) and os.path.exists(map_path):
        print("[INFO] Found existing files; set overwrite=True to regenerate.")
        return

    # Enforce MedSigLIP context length
    max_length = int(min(max_length, 64))

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

    report_hash_to_idx = {}
    unique_reports = []
    image_to_idx = {}

    # 1) First pass: collect unique reports + image mapping
    for batch in dataloader:
        if isinstance(batch, list):
            texts = [b["text"] for b in batch]
            paths = [b["image_path"] for b in batch]
        else:
            texts = list(batch["text"]) if isinstance(batch["text"], (list, tuple)) else [batch["text"]]
            paths = list(batch["image_path"]) if isinstance(batch["image_path"], (list, tuple)) else [batch["image_path"]]

        for txt, pth in zip(texts, paths):
            h = hash_report(txt)
            if h not in report_hash_to_idx:
                report_hash_to_idx[h] = len(unique_reports)
                unique_reports.append(txt if txt is not None else "")
            image_to_idx[image_stem(str(pth))] = report_hash_to_idx[h]

    print(f"[INFO] Unique reports: {len(unique_reports)} (from {len(image_to_idx)} images)")

# --- 2) Encode unique reports in batches ---
    from tqdm import tqdm

    all_emb = []
    token_counts = []

    is_cuda = device.startswith("cuda") and torch.cuda.is_available()
    amp_enabled = bool(use_amp and is_cuda)
    did_print_token_debug = False

    num_batches = (len(unique_reports) + batch_size - 1) // batch_size
    print(f"[MedSigLIP] Encoding {len(unique_reports)} reports in {num_batches} batches (bs={batch_size}, device={device})")

    for start in tqdm(
        range(0, len(unique_reports), batch_size),
        total=num_batches,
        desc="MedSigLIP text embeddings",
    ):
        chunk = unique_reports[start:start + batch_size]

        tok = tokenizer(
            chunk,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}

        # token stats (post-tokenization, non-pad tokens)
        attention_mask = tok.get("attention_mask", None)
        if attention_mask is None:
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            attention_mask = (tok["input_ids"] != pad_id).long()

        # token stats (non-pad tokens)
        seq_counts = attention_mask.sum(dim=1).detach().cpu().numpy().tolist()
        token_counts.extend(seq_counts)


        # one-time token debug on the first report only
        if not did_print_token_debug:
            raw_tok = tokenizer(
                chunk[0],
                padding=False,
                truncation=False,
                return_tensors="pt",
            )
            raw_len = raw_tok["input_ids"].shape[1]
            used_len = int(attention_mask[0].sum().item())
            was_truncated = raw_len > max_length

            print(
                "\n[Token Debug – first report only]"
                f"\n  Raw token length        : {raw_len}"
                f"\n  Used token length       : {used_len}"
                f"\n  Max allowed length      : {max_length}"
                f"\n  Truncated               : {was_truncated}"
            )
            preview = chunk[0][:200].replace("\n", " ")
            print(f"  Text preview            : \"{preview}...\"")
            did_print_token_debug = True

        # forward: text features only (no pixel_values)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            out = model.get_text_features(
                input_ids=tok["input_ids"],
                attention_mask=attention_mask,
            )

            # HF versions differ: sometimes returns Tensor, sometimes a ModelOutput
            if torch.is_tensor(out):
                pooled = out
            elif hasattr(out, "text_embeds") and out.text_embeds is not None:
                pooled = out.text_embeds
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                pooled = out.pooler_output
            else:
                raise RuntimeError(f"Unexpected get_text_features output: {type(out)}")

        if normalize:
            pooled = torch.nn.functional.normalize(pooled, dim=-1)

        # lightweight debug once
        if start == 0:
            mean_norm = float(pooled.norm(dim=1).mean().item())
            print(f"[DEBUG] pooled shape={tuple(pooled.shape)} dtype={pooled.dtype} mean_norm={mean_norm:.3f}")

        all_emb.append(pooled.detach().float().cpu().numpy().astype(np.float32))

    if len(all_emb) == 0:
        raise RuntimeError("No reports were found in dataloader; cannot compute embeddings.")

    reports_emb = np.concatenate(all_emb, axis=0)  # [N_unique, D]
    np.save(emb_path, reports_emb)

    # Save average/total token count
    avg_tokens = None
    total_tokens = None
    if len(token_counts) > 0:
        avg_tokens = float(np.mean(token_counts))
        total_tokens = int(np.sum(token_counts))
        with open(os.path.join(save_directory, "avg_num_tokens.txt"), "w") as f:
            f.write(str(avg_tokens))
        with open(os.path.join(save_directory, "total_num_tokens.txt"), "w") as f:
            f.write(str(total_tokens))

    meta = {
        "model_name": model_name,
        "max_length": max_length,
        "embedding_dim": int(reports_emb.shape[1]),
        "num_unique_reports": int(reports_emb.shape[0]),
        "avg_num_tokens": (avg_tokens if avg_tokens is not None else None),
        "report_hash_to_idx": report_hash_to_idx,
        "notes": "Embeddings computed from MedSigLIP text projection space (model.get_text_features).",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with open(map_path, "w") as f:
        json.dump(image_to_idx, f)

    print(f"[DONE] Saved:\n- {emb_path}\n- {meta_path}\n- {map_path}")

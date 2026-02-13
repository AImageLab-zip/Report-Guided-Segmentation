# Scripts

Utility scripts for preprocessing and auxiliary workflows.

## 📄 [`extract_textemb_biobert.py`](extract_textemb_biobert.py)
Precomputes BioBERT report embeddings.

The script supports two modes:

- `--mode brats3d` (default): writes one `.npz` embedding per BraTS case into
  `<dataset_root>/<report_folder>/<case_id>.npz`, matching
  `datasets/BraTS3DText.py` lookup logic.
- `--mode qatacov`: legacy QaTaCov export to a shared embedding directory.

**Outputs:**
- `reports_emb.npy`
- `reports_meta.json`
- `image_to_report_idx.json`

**Typical usage (BraTS3D / BraTS3DText):**

```bash
python scripts/extract_textemb_biobert.py \
  --mode brats3d \
  --config ./config/config_brats3d.json \
  --reports_path /path/to/brats_reports_txt_or_json \
  --report_folder rep_RG \
  --pooling mean \
  --max_len 256 \
  --overwrite
```

`--reports_path` can be either:

- a directory with `<BraTS-case-id>.txt` files, or
- a JSON file mapping `case_id -> report_text`.

**Typical usage (QaTaCov, legacy):**

```bash
python scripts/extract_textemb_biobert.py \
  --mode qatacov \
  --config ./config/config_qatacov2d.json \
  --save_directory /path/to/Text_Embeddings/BioBERT \
  --pooling mean \
  --max_len 256
```

To run on a Slurm cluster, see [`jobs/extract_text_emb.sh`](../jobs/extract_text_emb.sh).

---

## 📄 [`preprocess_qatacov.py`](preprocess_qatacov.py)
Offline preprocessing utility for QaTaCov that resizes images/masks, saves them into
`Images/` and `Ground-truths/`, and produces a `qatacov_split.xlsx` containing image
names, splits, and report text.

**Typical usage:**

```bash
python scripts/preprocess_qatacov.py \
  --root /path/to/QaTa-COV19-v2 \
  --out /path/to/QaTa-COV19-v2_preprocessed \
  --image_size 224 224 \
  --validation
```

You can also point to a config file and store preprocess settings under a `preprocess`
section (e.g., `preprocess.output_path`, `preprocess.use_test_transforms`).
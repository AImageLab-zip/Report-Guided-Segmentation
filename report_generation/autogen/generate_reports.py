import torch
import pandas as pd
import json
from tqdm import tqdm
import shutil as sh
from report_generation.utils.data import JsonDataset
from pathlib import Path

from argparse import ArgumentParser
"""
Generate Brain2Text (B2T) structured and textual reports from precomputed geometric datasets.

This script builds a B2T dataset by combining:
  - Tumor geometric features from BraTS_CC (output of run_cc_segmentation.py),
  - Atlas-based regional annotations from BraTS_atlas (output of generate_atlas.py),
  - Eloquent cortex masks from BraTS_eloquent (output of generate_eloquent.py),
  - A postprocessed anatomical legend (CSV).

For each specified threshold value, a new dataset directory is created:
    <output-root>/BraTS_B2T_t{threshold*100}

Within each subject folder, two files are generated:
    - generated.json   Structured representation of extracted features.
    - generated.txt    Textual description derived from the structured data.

Processing pipeline:
  1) Instantiate `JsonDataset` with dataset paths and threshold.
  2) Iterate through subjects using a PyTorch DataLoader (parallelized).
  3) Save per-subject structured JSON and generated text output.

The `threshold` parameter controls filtering or spatial inclusion criteria
inside `JsonDataset` (e.g., region overlap threshold).

Command-line arguments:
  --output-root      Root directory where B2T datasets will be created.
  --thresholds       One or more threshold values (e.g., 0.0 0.2 0.5).
  --num-workers      Number of DataLoader workers (default: 8).

  --cc-path          Path to BraTS_CC directory (output of run_cc_segmentation.py).
  --atlas-path       Path to BraTS_atlas directory (output of generate_atlas.py).
  --legend-path      Path to legend_postprocessed.csv.
  --eloquent-path    Path to BraTS_eloquent directory (output of generate_eloquent.py).
"""
parser = ArgumentParser()
parser.add_argument("--output-root", type=Path, required=True,
                    help="Root directory where B2T datasets will be generated.")
parser.add_argument("--legend-path", type=Path, required=True,
                    help="Path to legend_postprocessed.csv.")
parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0],
                    help="List of thresholds (e.g. --thresholds 0.0 0.2 0.5).")
parser.add_argument("--num-workers", type=int, default=8,
                    help="Number of DataLoader workers.")


parser.add_argument("--cc-path", type=Path, required=True,
                    help="Path to BraTS_CC dataset directory.")
parser.add_argument("--atlas-path", type=Path, required=True,
                    help="Path to BraTS_atlas directory (output of atlas generation).")
parser.add_argument("--legend-path", type=Path, required=True,
                    help="Path to legend_postprocessed.csv.")
parser.add_argument("--eloquent-path", type=Path, required=True,
                    help="Path to BraTS_eloquent directory.")

args = parser.parse_args()

pbar = tqdm(args.thresholds,desc='Starting')
for th in pbar:
    pbar.set_description(f'Generating B2T dataset with threshold: {th}')
    dataset = JsonDataset(cc_path=args.cc_path,
                          atlas_path=args.atlas_path,
                          legend_path=args.legend_path,
                          eloquent_path=args.eloquent_path,
                          threshold = th,
                          )
    loader = torch.utils.data.DataLoader(dataset,batch_size=None,num_workers=args.num_workers,shuffle=False)
    output_path = args.output_root / f"BraTS_B2T_t{int(th * 100)}"
    if output_path.exists():
        while True:
            answer = input(
                f'I found a preexisting output directory: {output_path}\n'
                'Do you want to delete it? y/n\n'
            ).strip().lower()

            if answer == 'y':
                sh.rmtree(output_path)
                output_path.mkdir(parents=True, exist_ok=False)
                break
            elif answer == 'n':
                raise RuntimeError("Aborting to avoid overwriting existing output directory.")
    else:
        output_path.mkdir(parents=True, exist_ok=False)
    legend = pd.read_csv(args.legend_path)

    for element in tqdm(loader,unit='brains'):
        subj_dir = output_path / element['name']
        subj_dir.mkdir(exist_ok=True)
        with open(subj_dir / 'generated.json','w') as f:
            json.dump(element['data'],f,indent=3)
        with open(subj_dir/'generated.txt','w') as f:
            f.write(element['txt'])

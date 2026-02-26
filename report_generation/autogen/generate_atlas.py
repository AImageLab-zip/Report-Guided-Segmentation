import argparse
import ants
import os
from pathlib import Path
from tqdm import tqdm
import shutil
"""
Register a T1 atlas to each subject T1 and warp a labeled brain-areas atlas into subject space.

This script iterates over a dataset organized as one folder per subject (ex: Brats2023). For each subject, it:
  1) Loads the subject T1 image (<subject>-t1n.nii.gz).
  2) Computes a deformable registration (SyN) between the subject T1 (fixed) and the T1 atlas (moving).
  3) Applies the resulting forward transforms (atlas -> subject) to the labeled atlas (areas atlas),
     using nearest-neighbor interpolation to preserve integer labels.
  4) Writes the warped label map to the subject output folder.
  5) Copies forward and inverse transform files to an output transforms/ directory for reproducibility.

Outputs are written under:
  <output-dir>/<subject>/
    - <subject>_t1n.nii.gz          (copy of subject T1 for debugging/viewing)
    - areas.nii.gz                  (warped areas atlas in subject space)
    - transforms/
        fwd_*.mat / fwd_*.nii.gz    (atlas -> subject transforms)
        inv_*.mat / inv_*.nii.gz    (subject -> atlas transforms)

Command-line arguments:
  --input-dir         Path to directory containing subject subfolders (BraTS2023 works straight out of the box).
  --output-dir        Path to output directory (created; if it exists, user is prompted to delete it).
  --t1-atlas-path     Path to the T1 atlas NIfTI file (intensity image).
  --areas-atlas-path  Path to the labeled areas atlas NIfTI file (integer labels).
"""
parser = argparse.ArgumentParser()
parser.add_argument("--input-dir",type=Path)
parser.add_argument("--output-dir",type=Path)
parser.add_argument("--t1-atlas-path",type=Path)
parser.add_argument("--areas-atlas-path",type=Path)

args = parser.parse_args()
base_path=Path(__file__).parent.parent.parent

if os.path.exists(args.output_dir):
    while True:
        answer = input(f'I found a preexisting output directory:{args.output_dir}.\n' +
                       'Do you want to delete it? y/n\n').strip().lower()
        if answer == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir,exist_ok=False)
            break
else:
    os.makedirs(args.output_dir,exist_ok=False)
print(f'Starting the iteration over {args.input_dir}')

t1_atlas = ants.image_read(str(args.t1_atlas_path))
complete_atlas = ants.image_read(str(args.areas_atlas_path))

args = [(sub,args.output_dir) for sub in args.input_dir.iterdir()]

atlas = ants.resample_image_to_target(
    image=complete_atlas,
    target=t1_atlas,
    interp_type="nearestNeighbor"
)


for sub in tqdm(list(args.input_dir.iterdir())):
    t1_subj  = ants.image_read(str(sub/f'{sub.name}-t1n.nii.gz'))

    tx_dir = args.output_dir / sub.name / "transforms"
    tx_dir.mkdir(parents=True, exist_ok=True)

    reg = ants.registration(
        fixed=t1_subj,
        moving=t1_atlas,
        type_of_transform="SyN",
        verbose=False
    )

    fwd_transforms = reg["fwdtransforms"]      # atlas->subject
    inv_transforms = reg["invtransforms"]      # subject->atlas

    # Saving the T1 for debugging and viewing
    ants.image_write(t1_subj, str(args.output_dir / sub.name/ f"{sub.name}_t1n.nii.gz"))
    atlas = ants.apply_transforms(
        fixed=t1_subj,
        moving=complete_atlas,
        transformlist=fwd_transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write(atlas, str(args.output_dir / sub.name/ f"areas.nii.gz"))

    # Save forward (atlas -> subject)
    fwd_saved = []
    for i, tx in enumerate(reg["fwdtransforms"]):
        dst = tx_dir / f"fwd_{i}_{Path(tx).name}"
        shutil.copy(tx, dst)
        fwd_saved.append(str(dst))

    # Save inverse (subject -> atlas)
    inv_saved = []
    for i, tx in enumerate(reg["invtransforms"]):
        dst = tx_dir / f"inv_{i}_{Path(tx).name}"
        shutil.copy(tx, dst)
        inv_saved.append(str(dst))


from pathlib import Path
import ants
import shutil as sh
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser

"""
Warp eloquent-cortex ROI masks from atlas space to each subject using precomputed transforms.

This script applies *atlas -> subject* forward transforms (saved per subject) to a set of
binary ROI masks (motor, speech-motor, speech-receptive, vision). It is intended to be run
after `generate_atlas.py`, and expects as input the directory produced by that script.

In particular, `--atlas-path` must point to the output directory created by `generate_atlas.py`,
which contains (for each subject) the subject T1 image and a `transforms/` folder with files:
  - fwd_* : forward transforms (atlas -> subject) produced by ANTs registration.

For each subject directory in `--atlas-path`, the script:
  1) Loads the subject T1 image `<subject>-t1n.nii.gz` as the fixed image.
  2) Loads and sorts the forward transforms `transforms/fwd_*` by index.
  3) Applies those transforms to each ROI mask provided via CLI.
  4) Writes the warped ROI masks to:
       <output-path>/<subject>/{motor,speech_motor,speech_receptive,vision}.nii.gz

Command-line arguments:
  --atlas-path              Path to the directory produced by `generate_atlas.py`.
  --output-path             Path to the output directory (created; if it exists, the user is
                           prompted to delete it).

  --motor-path              Path to motor ROI mask in atlas space (NIfTI).
  --speech-motor-path       Path to speech-motor ROI mask in atlas space (NIfTI).
  --speech-receptive-path   Path to speech-receptive ROI mask in atlas space (NIfTI).
  --vision-path             Path to vision ROI mask in atlas space (NIfTI).
"""
parser = ArgumentParser()
parser.add_argument('--atlas-path',type=Path,required=True)
parser.add_argument('--output-path',type=Path,required=True)

parser.add_argument("--motor-path", type=str, required=True)
parser.add_argument("--speech-motor-path", type=str, required=True)
parser.add_argument("--speech-receptive-path", type=str, required=True)
parser.add_argument("--vision-path", type=str, required=True)
args = parser.parse_args()

atlas_path = args.atlas_path
output_path = args.output_path

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

# Load atlas masks once (shared across threads)
motor = ants.image_read(str(args.motor_path))
speech_motor = ants.image_read(str(args.speech_motor_path))
speech_receptive = ants.image_read(str(args.speech_receptive_path))
vision = ants.image_read(str(args.vision_path))


def process_subject(sub):
    transform_path = sub / 'transforms'

    transforms = sorted(
        transform_path.glob("fwd_*"),
        key=lambda x: int(x.name.split("_")[1])
    )
    transforms = [str(t) for t in transforms]

    fixed = ants.image_read(str(sub / f'{sub.name}-t1n.nii.gz'))

    output_sub = output_path / sub.name
    output_sub.mkdir()

    warped_motor = ants.apply_transforms(
        fixed=fixed,
        moving=motor,
        transformlist=transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write(warped_motor, str(output_sub / 'motor.nii.gz'))

    warped_speech_motor = ants.apply_transforms(
        fixed=fixed,
        moving=speech_motor,
        transformlist=transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write(warped_speech_motor, str(output_sub / 'speech_motor.nii.gz'))

    warped_speech_receptive = ants.apply_transforms(
        fixed=fixed,
        moving=speech_receptive,
        transformlist=transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write(warped_speech_receptive, str(output_sub / 'speech_receptive.nii.gz'))

    warped_vision = ants.apply_transforms(
        fixed=fixed,
        moving=vision,
        transformlist=transforms,
        interpolator="nearestNeighbor"
    )
    ants.image_write(warped_vision, str(output_sub / 'vision.nii.gz'))


subjects = list(atlas_path.iterdir())

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_subject, sub) for sub in subjects]

    for _ in tqdm(as_completed(futures), total=len(futures)):
        pass

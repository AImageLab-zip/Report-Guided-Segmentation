from src.utils.geometries import make_cc_labels

from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import shutil as sh
import nibabel as nib
import torch
import numpy as np
from argparse import ArgumentParser
class CCDataset(torch.utils.data.Dataset):
    def __init__(self, image_path:Path, output_path:Path):

        self.data_list = sorted(list(image_path.iterdir()))
        self.data_list = [path / f'{path.name}-seg.nii.gz' for path in self.data_list]
        self.image_path = image_path

        if output_path.exists():
            sh.rmtree(output_path)
        output_path.mkdir()

        self.output_path = output_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if isinstance(idx,str):
            path = self.image_path /idx/ f'{idx}-seg.nii.gz'
        else:
            path = self.data_list[idx]

        segmentation = nib.loadsave.load(path)
        affine = segmentation.affine
        header = segmentation.header
        segmentation = segmentation.get_fdata().astype(np.uint8)
        name = path.parent.name
        (self.output_path / name).mkdir()
        save_path = self.output_path / name / f'{name}-cc.nii.gz'
        new_segmentation = make_cc_labels(segmentation)

        img = nib.Nifti1Image(new_segmentation, affine=affine, header=header)
        nib.save(img, filename=save_path)
        return 0

parser = ArgumentParser()
parser.add_argument(
    "--image-path",
    type=Path,
    required=True,
    help="Path to BraTS dataset directory containing subject folders with <subject>-seg.nii.gz.",
)
parser.add_argument(
    "--output-path",
    type=Path,
    required=True,
    help="Output directory where connected-component segmentations will be saved.",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=8,
    help="Number of DataLoader workers.",
)

args = parser.parse_args()

dataset = CCDataset(
    image_path=args.image_path,
    output_path=args.output_path)
loader = DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, drop_last=False, shuffle=True)

for element in tqdm(loader, total=len(loader)):
    pass





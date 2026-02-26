import numpy as np
from scipy import ndimage as ndi
import torch
from pathlib import Path
import nibabel as nib
from tqdm import tqdm
from argparse import ArgumentParser

"""
Fill single-voxel holes in atlas label maps within a brain mask.

This script processes a dataset organized as one folder per subject. For each subject, it:
  1) Loads the subject T1 image (<subject>-t1n.nii.gz) and derives a brain mask
     by thresholding non-zero intensities.
  2) Finds all atlas label maps (.nii.gz) in the same folder, excluding:
       - the T1 image (*-t1n.nii.gz)
       - already processed files containing "fill" in the filename.
  3) Fills holes defined as voxels:
       - with label 0 (background),
       - inside the brain mask,
       - at exact Euclidean distance 1 from the nearest labeled voxel.
  4) Assigns each such voxel the label of its nearest neighbor.
  5) Saves the corrected atlas as "<original_name>-fill.nii.gz".


Input structure (per subject folder):
  <subject>/
    - <subject>-t1n.nii.gz
    - <atlas_1>.nii.gz
    - <atlas_2>.nii.gz
    - ...

Output (per atlas):
  <atlas_name>-fill.nii.gz


Arguments:
  --input-path    Path containing subject subdirectories.
"""

def fill_holes(atlas, brain_mask):
    # Holes we want to consider (inside brain mask)
    holes = (atlas == 0) & brain_mask
    if not np.any(holes):
        return atlas

    # Labeled voxels
    labeled = atlas != 0

    # Zeros = features for EDT (labeled voxels)
    edt_input = (~labeled).astype(np.uint8)

    # Distance transform with distances and nearest indices
    distances, indices = ndi.distance_transform_edt(
        edt_input, return_indices=True
    )

    # Only holes at distance exactly 1
    holes_dist1 = holes & (distances == 1)

    if not np.any(holes_dist1):
        return atlas

    # Nearest labels
    nearest_labels = atlas[tuple(indices)]

    # Fill only distance-1 holes
    filled = atlas.copy()
    filled[holes_dist1] = nearest_labels[holes_dist1]

    return filled

class AtlasDataset(torch.utils.data.Dataset):
    def __init__(self, image_path:Path):

        self.data_list = sorted(list(image_path.iterdir()))
        self.data_list = [path for path in self.data_list]
        self.image_path = image_path

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path = self.data_list[idx]
        ref = nib.loadsave.load(path / f'{path.name}-t1n.nii.gz').get_fdata()
        ref = np.clip(ref,0,None)
        ref = ref != 0

        list_of_atlases = [atlas for atlas in path.iterdir() if atlas.name.endswith('.nii.gz') and not atlas.name.endswith('t1n.nii.gz') and not 'fill' in atlas.name]
        for file in list_of_atlases:
            new_name = file.name.split('.')[0] + '-fill.nii.gz'
            save_path = path / new_name

            img = nib.loadsave.load(file)
            affine = img.affine
            header = img.header
            atlas = img.get_fdata().astype(np.uint16)

            new_atlas = fill_holes(atlas,ref)

            new_img = nib.Nifti1Image(dataobj=new_atlas,affine = affine,header = header)
            nib.save(new_img,save_path)

        return 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input-path',type=Path,required=True)
    args = parser.parse_args()
    dataset = AtlasDataset(image_path=args.input_path)
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         shuffle=False,
                                         num_workers=8,
                                         drop_last=False,
                                         pin_memory=False,
                                         batch_size=1)

    for _ in tqdm(loader,total=len(loader)):
        pass

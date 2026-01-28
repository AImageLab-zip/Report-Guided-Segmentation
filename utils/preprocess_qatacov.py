import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets import DatasetFactory
from datasets.QaTaCov import QaTaCov2DSet
from transforms import QaTaCov as QaTaCovTransforms
from types import SimpleNamespace
from config import Config


def build_config(root_path: str, image_size: list[int]):
    return SimpleNamespace(
        dataset={
            'name': 'QaTaCov',
            'path': root_path,
            'batch_size': 1,
            'image_size': image_size,
        }
    )


def save_sample(sample, out_dir: str, split: str, index: int, preserve_filenames: bool):
    image = sample['image']
    label = sample['label']
    image_path = sample.get('image_path', '')
    label_path = sample.get('label_path', '')

    image = image.squeeze(0).cpu().numpy()
    label = label.squeeze(0).cpu().numpy()

    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image = (image * 255).astype(np.uint8)
    label = (label * 255).astype(np.uint8)

    split_dir = os.path.join(out_dir, split)
    images_dir = os.path.join(split_dir, 'Images')
    masks_dir = os.path.join(split_dir, 'Ground-truths')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    if preserve_filenames and image_path and label_path:
        image_name = os.path.basename(image_path)
        mask_name = os.path.basename(label_path)
    else:
        image_name = f"{index:06d}.png"
        mask_name = f"{index:06d}.png"

    image_path = os.path.join(images_dir, image_name)
    mask_path = os.path.join(masks_dir, mask_name)

    Image.fromarray(image).save(image_path)
    Image.fromarray(label).save(mask_path)


def main():
    parser = argparse.ArgumentParser(description='Offline preprocessing for QaTaCov')
    parser.add_argument(
        '--root',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'QaTaCov19-v2', 'QaTa-COV19-v2'),
        help='Path to QaTa-COV19-v2 dataset root'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Optional config JSON to load dataset settings (overrides --root/--image_size)'
    )
    parser.add_argument('--out', type=str, default=None, help='Output folder for preprocessed data')
    parser.add_argument('--image_size', nargs=2, type=int, default=[224, 224])
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--max_items', type=int, default=None, help='Optional limit per split')
    parser.add_argument('--use_test_transforms', action='store_true', help='Use test transforms for all splits')
    parser.add_argument('--preserve_filenames', action='store_true', help='Preserve original filenames')
    args = parser.parse_args()

    if args.config:
        config = Config(args.config)
        args.root = config.dataset['path']
        args.image_size = config.dataset.get('image_size', args.image_size)
        if hasattr(config, 'preprocess'):
            args.out = config.preprocess.get('output_path', args.out)
            args.use_test_transforms = config.preprocess.get('use_test_transforms', args.use_test_transforms)
            args.preserve_filenames = config.preprocess.get('preserve_filenames', args.preserve_filenames)
    else:
        config = build_config(args.root, args.image_size)
    if not args.out:
        raise ValueError("Output path not provided. Use --out or set preprocess.output_path in config.")

    dataset = DatasetFactory.create_instance(config, args.validation)

    if args.use_test_transforms:
        image_size = args.image_size
        test_transforms = QaTaCovTransforms(mode='test').transform(image_size=image_size)
        train_text_map = dataset.train_set.text_map
        test_text_map = dataset.test_set.text_map
        dataset.train_set = QaTaCov2DSet(dataset.train_images, dataset.train_labels, train_text_map, test_transforms)
        if args.validation:
            dataset.val_set = QaTaCov2DSet(dataset.val_images, dataset.val_labels, train_text_map, test_transforms)
        dataset.test_set = QaTaCov2DSet(dataset.test_images, dataset.test_labels, test_text_map, test_transforms)

    for split in ['train', 'val', 'test']:
        if split == 'val' and not args.validation:
            continue
        data = getattr(dataset, f"{split}_set")
        max_items = args.max_items or len(data)

        for idx in tqdm(range(max_items), desc=f"Saving {split}"):
            sample = data[idx]
            save_sample(sample, args.out, split, idx, args.preserve_filenames)


if __name__ == '__main__':
    main()
import argparse
import json
import os
import sys
import torch
from itertools import chain

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import Config
from datasets.DatasetFactory import DatasetFactory
from utils.textemb_MedSigLIP import precompute_unique_report_embeddings_medsiglip
from transforms.TransformsFactory import TransformsFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute report text embeddings.")
    parser.add_argument(
        "--config",
        default="./config/config_qatacov2d.json",
        help="Path to the config JSON file.",
    )
    parser.add_argument(
        "--save_directory",
        default="/leonardo_work/IscrC_narc2/reports_project/QaTa-COV19/Text_Embeddings/MedSigLip",
        help="Directory where text embeddings will be saved.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=64,
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


def main() -> None:
    args = parse_args()

    dataset_factory = DatasetFactory()
    config = Config(args.config)
    print(config)

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
    val_loader = qatacov.get_loader("val")

    precompute_unique_report_embeddings_medsiglip(
        dataloader=chain(train_loader, val_loader),
        save_directory=args.save_directory,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_length=args.max_len,          # will be clamped to 64
        overwrite=args.overwrite,
        use_safetensors=not args.no_safetensors,
    )


if __name__ == "__main__":
    main()

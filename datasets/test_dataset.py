import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import datasets
from DatasetFactory import DatasetFactory #togli il . se non usi pytest
from config import Config
import torchio as tio
from tqdm import tqdm
import json
from transforms import TransformsFactory

from PIL import Image

def test_atlas():
    d = DatasetFactory()
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_atlas.json")
    print(c)
    # Instantiate the Transforms
    transforms_config = c.dataset['transforms']
    with open(transforms_config, 'r') as f:
        transforms_config = json.load(f)  
    preprocessing_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []))
    augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('augmentations', []))

    if preprocessing_transforms and augmentation_transforms:
        train_transforms = tio.Compose([preprocessing_transforms, augmentation_transforms])
        test_transforms = preprocessing_transforms
    elif preprocessing_transforms:
        train_transforms = preprocessing_transforms
        test_transforms = preprocessing_transforms
    elif augmentation_transforms:
        train_transforms = augmentation_transforms
        test_transforms = None
    else:
        train_transforms = None
        test_transforms = None

    atlas = d.create_instance(
        config=c,
        validation=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms
    )

    train_loader = atlas.get_loader('train')
    val_loader = atlas.get_loader('val')
    test_loader = atlas.get_loader('test')

    for idx, sample in tqdm(enumerate(train_loader), desc=f'Epoch 0', total=len(train_loader)):
        image = sample['image'][tio.DATA].float()
        label = sample['label'][tio.DATA].float()
        assert image.shape == label.shape == (c.dataset['batch_size'], 1, c.dataset['patch_size'], c.dataset['patch_size'], c.dataset['patch_size'])
        print(image.shape, label.shape)

    # for idx, sample in tqdm(enumerate(test_loader), desc=f'Epoch 0', total=len(test_loader)):
    #     image = sample['image'][tio.DATA].float()
    #     label = sample['label'][tio.DATA].float()
    #     assert image.shape == label.shape == (c.dataset['batch_size'], 1, c.dataset['patch_size'], c.dataset['patch_size'], c.dataset['patch_size'])
    #     print(image.shape, label.shape)

    #a = ATLAS("/work/grana_neuro/Brain-Segmentation/config.json", "train", "/work/grana_neuro/nnUNet_raw/Dataset102_ATLAS")


def test_BraTS2D():
    d = DatasetFactory()
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_brats2d.json")
    print(c)
    # Instantiate the Transforms
    transforms_config = c.dataset['transforms']
    with open(transforms_config, 'r') as f:
        transforms_config = json.load(f)  

    #preprocessing_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []))
    augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('augmentations', []))

    if augmentation_transforms:
        train_transforms = augmentation_transforms
        test_transforms = None
    else:
        train_transforms = None
        test_transforms = None

    brats_2d = d.create_instance(
        config=c,
        validation=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms
    )

    #train_loader = brats_2d.get_loader('train')
    #val_loader = brats_2d.get_loader('val')
    test_loader = brats_2d.get_loader('test')
    sample = test_loader.dataset[0]


def test_brats3d():
    d = DatasetFactory()
    c = Config("/work/grana_neuro/Brain-Segmentation/config/config_brats3d.json")
    print(c)
    # Instantiate the Transforms
    transforms_config = c.dataset['transforms']
    with open(transforms_config, 'r') as f:
        transforms_config = json.load(f)  
    preprocessing_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []))
    augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('augmentations', []))

    if preprocessing_transforms and augmentation_transforms:
        train_transforms = tio.Compose([preprocessing_transforms, augmentation_transforms])
        test_transforms = preprocessing_transforms
    elif preprocessing_transforms:
        train_transforms = preprocessing_transforms
        test_transforms = preprocessing_transforms
    elif augmentation_transforms:
        train_transforms = augmentation_transforms
        test_transforms = None
    else:
        train_transforms = None
        test_transforms = None

    brats = d.create_instance(
        config=c,
        validation=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms
    )

    train_loader = brats.get_loader('train')
    val_loader = brats.get_loader('val')
    test_loader = brats.get_loader('test')

    for idx, sample in tqdm(enumerate(train_loader), desc=f'Epoch 0', total=len(train_loader)):
        image = sample['image'][tio.DATA].float()
        label = sample['label'][tio.DATA].float()
        assert image.shape == label.shape == (c.dataset['batch_size'], 4, c.dataset['patch_size'], c.dataset['patch_size'], c.dataset['patch_size'])
        print(image.shape, label.shape)


def test_qatacov():
    d = DatasetFactory()
    c = Config("./config/config_qatacov2d.json")
    print(c)

    transforms_config = c.dataset['transforms']
    with open(transforms_config, 'r') as f:
        transforms_config = json.load(f)

    augmentation_transforms = TransformsFactory.create_instance(transforms_config.get('preprocessing', []), backend='monai')

    if augmentation_transforms:
        train_transforms = augmentation_transforms
        test_transforms = None
    else:
        train_transforms = None
        test_transforms = None

    qatacov = d.create_instance(
        config=c,
        validation=True,
        train_transforms=train_transforms,
        test_transforms=test_transforms
    )

    train_loader = qatacov.get_loader('train')
    val_loader = qatacov.get_loader('val')
    test_loader = qatacov.get_loader('test')

    sample = next(iter(train_loader))
    image = sample['image']
    label = sample['label']
    text = sample['text']
    assert image.shape[0] == c.dataset['batch_size']
    assert image.shape == label.shape

    image = (image[0].numpy() * 255).astype('uint8').squeeze()
    label = (label[0].numpy() * 255).astype('uint8').squeeze()

    #save the first image and label
    Image.fromarray(image).save("test_image_0.png")
    Image.fromarray(label).save("test_label_0.png")

    print(image.shape, label.shape)
    print(text[0])

if __name__ == '__main__':
    #test_atlas()
    #test_BraTS2D()
    #test_brats3d()
    test_qatacov()
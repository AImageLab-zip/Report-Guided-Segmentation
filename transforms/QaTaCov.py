from typing import List
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RandZoomd,
    Resized,
    NormalizeIntensityd,
    ToTensord
)
from base.base_transforms_2d import BaseTransforms2D


class QaTaCov(BaseTransforms2D):
    """
    Transforms for QaTaCov X-ray dataset for COVID-19 segmentation.
    
    This class implements transformations specifically designed for 2D X-ray 
    images using dictionary-based MONAI transforms. Includes data augmentation
    for training mode (random zoom) and standard preprocessing for all modes.
    """
    
    def __init__(self, mode: str = 'train'):
        """
        Initialize the QaTaCov transforms.
        
        Args:
            mode (str): Mode of operation ('train', 'val', or 'test').
        """
        super().__init__(mode)
    
    def transform(self, image_size: List[int] = [224, 224]):
        """
        Define the transformation pipeline for QaTaCov X-ray images.
        
        Args:
            image_size (List[int]): Desired output image size [height, width]. 
                                    Default is [224, 224].
            
        Returns:
            Compose: MONAI Compose object containing the transformation pipeline.
            
        Pipeline:
            - LoadImaged: Load image and ground truth using PIL reader
            - EnsureChannelFirstd: Ensure channel-first format (C, H, W)
            - RandZoomd (train only): Random zoom augmentation (0.95-1.2x)
            - Resized: Resize to target image size
            - NormalizeIntensityd: Channel-wise intensity normalization
            - ToTensord: Convert to PyTorch tensors
        """
        trans = Compose([
                LoadImaged(["image", "gt"], reader='PILReader', image_only=False),
                EnsureChannelFirstd(["image", "gt"]),
                Resized(["image"], spatial_size=image_size, mode='bicubic'),
                Resized(["gt"], spatial_size=image_size, mode='nearest'),
                NormalizeIntensityd(['image'], channel_wise=True),
                ToTensord(["image", "gt"]),
            ])
        return trans

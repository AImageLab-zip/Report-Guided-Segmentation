from abc import ABC, abstractmethod
from typing import List


class BaseTransforms2D(ABC):
    """
    Abstract base class for 2D image transforms.
    
    All 2D transform classes should inherit from this class and implement
    the abstract method `transform`. This class provides a consistent interface
    for creating transformation pipelines for 2D medical image segmentation.
    """
    
    def __init__(self, mode: str = 'train'):
        """
        Initialize the transforms.
        
        Args:
            mode (str): Mode of operation ('train', 'val', or 'test').
                       Training mode typically includes data augmentation,
                       while validation/test modes only include preprocessing.
        """
        if mode not in ['train', 'val', 'test']:
            raise ValueError(f"Mode must be 'train', 'val', or 'test', got '{mode}'")
        self.mode = mode
    
    @abstractmethod
    def transform(self, image_size: List[int] = [224, 224]):
        """
        Define the transformation pipeline.
        
        This method must be implemented by all subclasses to define
        the specific transformation pipeline for the dataset.
        
        Args:
            image_size (List[int]): Desired output image size [height, width].
            
        Returns:
            Compose: A MONAI Compose object containing the transformation pipeline.
        """
        raise NotImplementedError("Subclasses must implement the transform method")
    
    def __call__(self, image_size: List[int] = [224, 224]):
        """
        Make the class callable to get transforms directly.
        
        Args:
            image_size (List[int]): Desired output image size [height, width].
            
        Returns:
            Compose: A MONAI Compose object containing the transformation pipeline.
        """
        return self.transform(image_size)

import torch
import torch.nn.functional as F

def _onehot_enc(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
           Function which return a one-hot encoded mask

           :param mask: mask --> (B, 1, D, H, W)
           :return: A one-hot encoded mask --> (B, C, D, H, W)
    """
    # Remove the singleton channel dimension
    mask = mask.squeeze(1)  # Shape: (B, D, H, W)

    # Apply one-hot encoding
    # The one_hot function expects input of shape (N, ...), where N is batch size
    # Output shape will be (B, D, H, W, num_classes)
    one_hot_mask = F.one_hot(mask, num_classes=num_classes)  # Labels should be of type torch.int64

    # 3. Permute dimensions to bring 'num_classes' to the channel dimension
    # Current shape: (B, D, H, W, num_classes)
    # Desired shape: (B, num_classes, D, H, W)
    one_hot_mask = one_hot_mask.permute(0, 4, 1, 2, 3).float()

    return one_hot_mask


def _onehot_enc_2d(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    One-hot encode 2D segmentation masks.

    :param mask: Tensor of shape (B, 1, H, W) or (B, H, W)
    :return: One-hot encoded mask of shape (B, C, H, W)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.squeeze(1).long()
    one_hot_mask = F.one_hot(mask, num_classes=num_classes)  # (B, H, W, C)
    one_hot_mask = one_hot_mask.permute(0, 3, 1, 2).float()
    return one_hot_mask
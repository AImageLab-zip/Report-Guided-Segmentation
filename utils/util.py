import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def _save_visualization_2d(self, image, label, prediction, phase, epoch, idx=0):
    """
    Save visualization of the first batch of predictions

    :param image: Input image tensor
    :param label: Ground truth label tensor
    :param prediction: Model prediction tensor
    :param phase: 'val' or 'test'
    :param epoch: Current epoch number
    :param idx: Sample index in the current batch
    """

    # Move tensors to CPU and convert to numpy
    image_np = image[idx].cpu().numpy().squeeze()
    label_np = label[idx].cpu().numpy().squeeze()
    prediction_np = prediction[idx].cpu().numpy().squeeze()

    pred = torch.softmax(torch.tensor(prediction_np), dim=0)
    prediction_np = torch.argmax(pred, dim=0).numpy()

    # Create a figure with subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image_np, cmap='gray')
    axs[0].set_title('Input Image')
    axs[0].axis('off')

    axs[1].imshow(label_np, cmap='jet', alpha=0.5)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    axs[2].imshow(prediction_np, cmap='jet', alpha=0.5)
    axs[2].set_title('Prediction')
    axs[2].axis('off')

    plt.suptitle(f'{phase.capitalize()} Epoch {epoch} Visualization')
    plt.tight_layout()
        
    # Create directory if it doesn't exist
    vis_dir = os.path.join(self.save_path, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Save the figure
    plt.savefig(os.path.join(vis_dir, f'{phase}_epoch_{epoch}_visualization_{idx}.png'))
    plt.close()
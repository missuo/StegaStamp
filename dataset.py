import glob
import os
import random
from os.path import join

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class StegaStampDataset(Dataset):
    """
    Dataset for StegaStamp training.

    Loads images from a directory and generates random binary secrets.
    Images are resized to 400x400 and normalized to [0, 1].
    """

    def __init__(self, data_dir, secret_size=100, image_size=(400, 400)):
        """
        Args:
            data_dir: Path to directory containing images
            secret_size: Number of bits in the secret (default: 100)
            image_size: Target image size (default: (400, 400))
        """
        self.data_dir = data_dir
        self.secret_size = secret_size
        self.image_size = image_size

        # Get list of image files
        self.files_list = glob.glob(join(data_dir, "**/*"), recursive=True)
        # Filter out directories and non-image files
        self.files_list = [
            f for f in self.files_list
            if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ]

        if len(self.files_list) == 0:
            raise ValueError(f"No images found in {data_dir}")

        print(f"Dataset initialized with {len(self.files_list)} images from {data_dir}")

    def __len__(self):
        # Return a large number since we sample randomly with replacement
        # This allows for unlimited epochs
        return 100000

    def __getitem__(self, idx):
        """
        Returns:
            image: [3, H, W] - image tensor in [0, 1]
            secret: [secret_size] - binary secret
        """
        # Randomly select an image
        img_path = random.choice(self.files_list)

        # Load and process image
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize/crop to target size (ImageOps.fit centers and crops)
            img = ImageOps.fit(img, self.image_size)
            # Convert to numpy array and normalize to [0, 1]
            img = np.array(img, dtype=np.float32) / 255.0
        except Exception as e:
            # If image loading fails, return black image
            print(f"Warning: Failed to load {img_path}: {e}")
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.float32)

        # Convert from HWC to CHW (PyTorch format)
        img = torch.from_numpy(img).permute(2, 0, 1)  # [3, H, W]

        # Generate random binary secret
        secret = np.random.binomial(1, 0.5, self.secret_size).astype(np.float32)
        secret = torch.from_numpy(secret)

        return img, secret


if __name__ == '__main__':
    print("Testing StegaStamp Dataset...")

    # Test with a dummy directory structure
    import tempfile
    import shutil

    # Create temporary directory with dummy images
    temp_dir = tempfile.mkdtemp()
    print(f"Created temp directory: {temp_dir}")

    try:
        # Create some dummy images
        for i in range(5):
            dummy_img = Image.new('RGB', (500, 500), color=(i*50, i*50, i*50))
            dummy_img.save(join(temp_dir, f"test_{i}.png"))

        # Test dataset
        dataset = StegaStampDataset(temp_dir, secret_size=100, image_size=(400, 400))

        print(f"\nDataset length: {len(dataset)}")

        # Test loading a few samples
        for i in range(3):
            img, secret = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Image shape: {img.shape}")
            print(f"  Image min/max: {img.min().item():.3f}/{img.max().item():.3f}")
            print(f"  Secret shape: {secret.shape}")
            print(f"  Secret sum: {secret.sum().item():.0f}/100")

        # Test DataLoader
        from torch.utils.data import DataLoader

        print("\n\nTesting DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        batch_img, batch_secret = next(iter(dataloader))
        print(f"Batch image shape: {batch_img.shape}")
        print(f"Batch secret shape: {batch_secret.shape}")

        print("\n✓ Dataset tests passed!")

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")

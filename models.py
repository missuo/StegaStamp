import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class StegaStampEncoder(nn.Module):
    """
    U-Net encoder that embeds a secret into an image by generating a residual.

    Architecture:
    - Secret processing: Dense layer + reshape to 50x50x3, then upsample to 400x400x3
    - Encoder path: 4 downsampling conv blocks (32->32->64->128->256 channels)
    - Decoder path: 4 upsampling blocks with skip connections (256->128->64->32->32)
    - Output: 3-channel residual to add to input image

    Input: (secret: [B, 100], image: [B, 3, 400, 400])
    Output: residual [B, 3, 400, 400]
    """

    def __init__(self, height=400, width=400):
        super(StegaStampEncoder, self).__init__()

        # Secret processing layer
        self.secret_dense = nn.Linear(100, 7500)

        # Encoder path (downsampling)
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)  # 6 channels: 3 from secret_enlarged + 3 from image
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder path (upsampling with skip connections)
        # Note: In TensorFlow version, UpSampling2D is used before Conv2D with padding='same'
        # After upsampling, apply conv with same padding
        self.up6 = nn.Conv2d(256, 128, kernel_size=2, padding='same')
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding='same')  # 128 from up6 + 128 from conv4

        self.up7 = nn.Conv2d(128, 64, kernel_size=2, padding='same')
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding='same')  # 64 from up7 + 64 from conv3

        self.up8 = nn.Conv2d(64, 32, kernel_size=2, padding='same')
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding='same')  # 32 from up8 + 32 from conv2

        self.up9 = nn.Conv2d(32, 32, kernel_size=2, padding='same')
        self.conv9 = nn.Conv2d(70, 32, kernel_size=3, padding='same')  # 32 from up9 + 32 from conv1 + 6 from inputs

        self.conv10 = nn.Conv2d(32, 32, kernel_size=3, padding='same')

        # Final residual output (no activation, can be positive or negative)
        self.residual = nn.Conv2d(32, 3, kernel_size=1, padding='same')

        # Initialize weights with He normal initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, secret, image):
        """
        Args:
            secret: [B, 100] - binary secret bits
            image: [B, 3, H, W] - input image in range [0, 1]

        Returns:
            residual: [B, 3, H, W] - residual to add to image
        """
        # Center inputs around 0
        secret = secret - 0.5
        image = image - 0.5

        # Process secret: dense layer -> reshape -> upsample
        secret_features = F.relu(self.secret_dense(secret))  # [B, 7500]
        secret_features = secret_features.view(-1, 3, 50, 50)  # [B, 3, 50, 50] (PyTorch uses NCHW)
        secret_enlarged = F.interpolate(secret_features, size=(400, 400), mode='nearest')  # [B, 3, 400, 400]

        # Concatenate secret and image
        inputs = torch.cat([secret_enlarged, image], dim=1)  # [B, 6, 400, 400]

        # Encoder path with skip connections
        conv1 = F.relu(self.conv1(inputs))  # [B, 32, 400, 400]
        conv2 = F.relu(self.conv2(conv1))   # [B, 32, 200, 200]
        conv3 = F.relu(self.conv3(conv2))   # [B, 64, 100, 100]
        conv4 = F.relu(self.conv4(conv3))   # [B, 128, 50, 50]
        conv5 = F.relu(self.conv5(conv4))   # [B, 256, 25, 25]

        # Decoder path with skip connections (U-Net style)
        # Use scale_factor=2 for upsampling, then apply conv, matching TF's UpSampling2D(2,2)
        up6 = F.relu(self.up6(F.interpolate(conv5, scale_factor=2, mode='nearest')))  # [B, 128, 50, 50]
        merge6 = torch.cat([conv4, up6], dim=1)  # [B, 256, 50, 50]
        conv6 = F.relu(self.conv6(merge6))  # [B, 128, 50, 50]

        up7 = F.relu(self.up7(F.interpolate(conv6, scale_factor=2, mode='nearest')))  # [B, 64, 100, 100]
        merge7 = torch.cat([conv3, up7], dim=1)  # [B, 128, 100, 100]
        conv7 = F.relu(self.conv7(merge7))  # [B, 64, 100, 100]

        up8 = F.relu(self.up8(F.interpolate(conv7, scale_factor=2, mode='nearest')))  # [B, 32, 200, 200]
        merge8 = torch.cat([conv2, up8], dim=1)  # [B, 64, 200, 200]
        conv8 = F.relu(self.conv8(merge8))  # [B, 32, 200, 200]

        up9 = F.relu(self.up9(F.interpolate(conv8, scale_factor=2, mode='nearest')))  # [B, 32, 400, 400]
        merge9 = torch.cat([conv1, up9, inputs], dim=1)  # [B, 70, 400, 400]
        conv9 = F.relu(self.conv9(merge9))  # [B, 32, 400, 400]

        conv10 = F.relu(self.conv10(conv9))  # [B, 32, 400, 400]

        # Generate residual (no activation, can be positive or negative)
        residual = self.residual(conv10)  # [B, 3, 400, 400]

        return residual


class StegaStampDecoder(nn.Module):
    """
    Decoder with Spatial Transformer Network for extracting secret from transformed images.

    Architecture:
    - STN parameter network: 3 conv layers + dense layer -> 6 affine parameters
    - STN: applies learned affine transformation to align the image
    - Decoder network: 5 conv layers + dense layers -> 100-bit secret

    Input: image [B, 3, 400, 400]
    Output: secret_logits [B, 100] (apply sigmoid to get probabilities)
    """

    def __init__(self, secret_size=100, height=400, width=400):
        super(StegaStampDecoder, self).__init__()
        self.height = height
        self.width = width
        self.secret_size = secret_size

        # Spatial Transformer Network - parameter estimation
        self.stn_params = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 50 * 50, 128),  # After 3 stride-2 convs: 400->200->100->50
            nn.ReLU(inplace=True)
        )

        # STN affine transformation parameters (initialize to identity)
        self.fc_loc = nn.Linear(128, 6)
        # Initialize to identity transformation
        self.fc_loc.weight.data.zero_()
        self.fc_loc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 13 * 13, 512),  # After 5 stride-2 convs: 400->200->100->50->25->13
            nn.ReLU(inplace=True),
            nn.Linear(512, secret_size)
        )

    def stn(self, x):
        """
        Spatial Transformer Network

        Args:
            x: [B, 3, H, W] - input image

        Returns:
            transformed: [B, 3, H, W] - spatially transformed image
        """
        # Get transformation parameters
        xs = self.stn_params(x)  # [B, 128]
        theta = self.fc_loc(xs)  # [B, 6]
        theta = theta.view(-1, 2, 3)  # [B, 2, 3] - affine transformation matrix

        # Generate sampling grid
        grid = F.affine_grid(theta, x.size(), align_corners=False)

        # Sample from input using grid
        x_transformed = F.grid_sample(x, grid, align_corners=False)

        return x_transformed

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] - input image in range [0, 1]

        Returns:
            secret_logits: [B, secret_size] - logits for secret bits (apply sigmoid for probabilities)
        """
        # Center input around 0
        image = image - 0.5

        # Apply spatial transformation
        transformed_image = self.stn(image)

        # Decode secret
        secret_logits = self.decoder(transformed_image)

        return secret_logits


class Discriminator(nn.Module):
    """
    WGAN-style discriminator for adversarial training.

    Architecture:
    - 5 convolutional layers with stride-2 downsampling
    - Returns both scalar output (mean of feature map) and full heatmap

    Input: image [B, 3, 400, 400]
    Output: (scalar_output [1], heatmap [B, 1, 25, 25])
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )

    def forward(self, image):
        """
        Args:
            image: [B, 3, H, W] - input image in range [0, 1]

        Returns:
            output: scalar - mean of heatmap
            heatmap: [B, 1, H/16, W/16] - discriminator feature map
        """
        # Center input around 0
        x = image - 0.5

        # Generate heatmap
        heatmap = self.model(x)  # [B, 1, 25, 25] for 400x400 input

        # Compute scalar output as mean
        output = torch.mean(heatmap)

        return output, heatmap


def get_secret_acc(secret_true, secret_pred):
    """
    Calculate bit accuracy and string accuracy for secret recovery.

    Args:
        secret_true: [B, secret_size] - ground truth secret bits
        secret_pred: [B, secret_size] - predicted secret logits

    Returns:
        bit_acc: scalar - fraction of correctly predicted bits
        str_acc: scalar - fraction of perfectly recovered secrets (all bits correct)
    """
    with torch.no_grad():
        secret_pred_binary = torch.round(torch.sigmoid(secret_pred))
        correct_bits = (secret_pred_binary == secret_true).sum(dim=1)  # [B]

        # String accuracy: fraction of samples with all bits correct
        total_bits = secret_true.shape[1]
        str_acc = (correct_bits == total_bits).float().mean()

        # Bit accuracy: fraction of all bits that are correct
        bit_acc = correct_bits.float().mean() / total_bits

        return bit_acc, str_acc


if __name__ == '__main__':
    # Test model shapes
    print("Testing StegaStamp PyTorch models...")

    batch_size = 4
    secret = torch.rand(batch_size, 100)
    image = torch.rand(batch_size, 3, 400, 400)

    # Test Encoder
    print("\n1. Testing Encoder...")
    encoder = StegaStampEncoder()
    residual = encoder(secret, image)
    print(f"   Input: secret {secret.shape}, image {image.shape}")
    print(f"   Output: residual {residual.shape}")
    assert residual.shape == (batch_size, 3, 400, 400), "Encoder output shape mismatch!"

    # Test Decoder
    print("\n2. Testing Decoder...")
    decoder = StegaStampDecoder()
    secret_logits = decoder(image)
    print(f"   Input: image {image.shape}")
    print(f"   Output: secret_logits {secret_logits.shape}")
    assert secret_logits.shape == (batch_size, 100), "Decoder output shape mismatch!"

    # Test Discriminator
    print("\n3. Testing Discriminator...")
    discriminator = Discriminator()
    output, heatmap = discriminator(image)
    print(f"   Input: image {image.shape}")
    print(f"   Output: scalar {output.item():.4f}, heatmap {heatmap.shape}")
    assert heatmap.shape[0] == batch_size and heatmap.shape[1] == 1, "Discriminator heatmap shape mismatch!"

    # Test accuracy function
    print("\n4. Testing accuracy calculation...")
    secret_true = torch.randint(0, 2, (batch_size, 100)).float()
    secret_pred = torch.randn(batch_size, 100)
    bit_acc, str_acc = get_secret_acc(secret_true, secret_pred)
    print(f"   Bit accuracy: {bit_acc:.4f}, String accuracy: {str_acc:.4f}")

    # Test gradient flow
    print("\n5. Testing gradient flow...")
    encoded_image = image + residual
    secret_recovered = decoder(encoded_image)
    loss = F.binary_cross_entropy_with_logits(secret_recovered, secret)
    loss.backward()
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Encoder has gradients: {encoder.conv1.weight.grad is not None}")
    print(f"   Decoder has gradients: {decoder.decoder[0].weight.grad is not None}")

    print("\n✓ All tests passed!")

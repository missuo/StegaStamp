import cv2
import itertools
import numpy as np
import random
import torch
import torch.nn.functional as F


def random_blur_kernel(probs, N_blur, sigrange_gauss, sigrange_line, wmin_line, device='cpu'):
    """
    Generate a random blur kernel for data augmentation.

    Args:
        probs: [p_gauss, p_line] - probabilities for Gaussian and line blur
        N_blur: kernel size (typically 7)
        sigrange_gauss: [min, max] - sigma range for Gaussian blur
        sigrange_line: [min, max] - sigma range for line blur
        wmin_line: minimum width for line blur
        device: torch device

    Returns:
        kernel: [N_blur, N_blur, 3, 3] - separable blur kernel for each channel
    """
    N = N_blur

    # Create coordinate grid
    coords_1d = torch.arange(N_blur, dtype=torch.float32, device=device) - (0.5 * (N - 1))
    coords = torch.stack(torch.meshgrid(coords_1d, coords_1d, indexing='ij'), dim=-1)  # [N, N, 2]
    manhat = torch.sum(torch.abs(coords), dim=-1)  # [N, N]

    # Identity kernel (no blur)
    vals_nothing = (manhat < 0.5).float()

    # Gaussian blur kernel
    sig_gauss = torch.rand(1, device=device) * (sigrange_gauss[1] - sigrange_gauss[0]) + sigrange_gauss[0]
    vals_gauss = torch.exp(-torch.sum(coords ** 2, dim=-1) / (2 * sig_gauss ** 2))

    # Line blur kernel
    theta = torch.rand(1, device=device) * 2 * np.pi
    v = torch.stack([torch.cos(theta), torch.sin(theta)], dim=0).squeeze()  # [2]
    dists = torch.sum(coords * v, dim=-1)  # [N, N]

    sig_line = torch.rand(1, device=device) * (sigrange_line[1] - sigrange_line[0]) + sigrange_line[0]
    w_line = torch.rand(1, device=device) * (0.5 * (N - 1) + 0.1 - wmin_line) + wmin_line

    vals_line = torch.exp(-dists ** 2 / (2 * sig_line ** 2)) * (manhat < w_line).float()

    # Random selection
    t = torch.rand(1, device=device)
    if t < probs[0]:
        vals = vals_gauss
    elif t < probs[0] + probs[1]:
        vals = vals_line
    else:
        vals = vals_nothing

    # Normalize
    vals = vals / torch.sum(vals)

    # Create separable kernel for 3 channels
    z = torch.zeros_like(vals)
    # Stack as [v, 0, 0, 0, v, 0, 0, 0, v] and reshape to [N, N, 3, 3]
    f = torch.stack([vals, z, z, z, vals, z, z, z, vals], dim=-1)
    f = f.reshape(N, N, 3, 3)

    return f


def get_rand_transform_matrix(image_size, d, batch_size):
    """
    Generate random perspective transformation matrices for data augmentation.

    Args:
        image_size: size of image (assumed square)
        d: maximum displacement of corners
        batch_size: number of transformations to generate

    Returns:
        Ms: [batch_size, 2, 8] - transformation matrices [M_inv, M]
    """
    Ms = np.zeros((batch_size, 2, 8))

    for i in range(batch_size):
        tl_x = random.uniform(-d, d)     # Top left corner, top
        tl_y = random.uniform(-d, d)     # Top left corner, left
        bl_x = random.uniform(-d, d)     # Bot left corner, bot
        bl_y = random.uniform(-d, d)     # Bot left corner, left
        tr_x = random.uniform(-d, d)     # Top right corner, top
        tr_y = random.uniform(-d, d)     # Top right corner, right
        br_x = random.uniform(-d, d)     # Bot right corner, bot
        br_y = random.uniform(-d, d)     # Bot right corner, right

        rect = np.array([
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size]], dtype="float32")

        dst = np.array([
            [0, 0],
            [image_size, 0],
            [image_size, image_size],
            [0, image_size]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        M_inv = np.linalg.inv(M)
        Ms[i, 0, :] = M_inv.flatten()[:8]
        Ms[i, 1, :] = M.flatten()[:8]

    return Ms


def get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size, device='cpu'):
    """
    Generate random brightness and hue adjustments.

    Args:
        rnd_bri: brightness adjustment range
        rnd_hue: hue adjustment range
        batch_size: number of samples
        device: torch device

    Returns:
        adjustment: [batch_size, 1, 1, 3] - brightness+hue adjustment
    """
    rnd_hue = torch.rand(batch_size, 1, 1, 3, device=device) * 2 * rnd_hue - rnd_hue
    rnd_brightness = torch.rand(batch_size, 1, 1, 1, device=device) * 2 * rnd_bri - rnd_bri
    return rnd_hue + rnd_brightness


# ============================================================================
# Differentiable JPEG Implementation
# Source: https://github.com/rshin/differentiable-jpeg
# ============================================================================

# JPEG quantization tables
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array(
    [[17, 18, 24, 47],
     [18, 21, 26, 66],
     [24, 26, 56, 99],
     [47, 66, 99, 99]]).T


def rgb_to_ycbcr_jpeg(image):
    """
    Convert RGB to YCbCr color space (JPEG standard).

    Args:
        image: [B, C, H, W] - RGB image in [0, 255]

    Returns:
        [B, C, H, W] - YCbCr image
    """
    matrix = torch.tensor(
        [[0.299, 0.587, 0.114],
         [-0.168736, -0.331264, 0.5],
         [0.5, -0.418688, -0.081312]],
        dtype=image.dtype, device=image.device).T

    shift = torch.tensor([0., 128., 128.], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)

    # Rearrange to [B, H, W, C] for matrix multiplication
    image_hwc = image.permute(0, 2, 3, 1)
    result = torch.matmul(image_hwc, matrix) + shift.squeeze(0).squeeze(-1).squeeze(-1)

    # Back to [B, C, H, W]
    return result.permute(0, 3, 1, 2)


def ycbcr_to_rgb_jpeg(image):
    """
    Convert YCbCr to RGB color space (JPEG standard).

    Args:
        image: [B, C, H, W] - YCbCr image

    Returns:
        [B, C, H, W] - RGB image in [0, 255]
    """
    matrix = torch.tensor(
        [[1., 0., 1.402],
         [1, -0.344136, -0.714136],
         [1, 1.772, 0]],
        dtype=image.dtype, device=image.device).T

    shift = torch.tensor([0., -128., -128.], dtype=image.dtype, device=image.device).view(1, 3, 1, 1)

    # Rearrange to [B, H, W, C]
    image_hwc = image.permute(0, 2, 3, 1)
    result = torch.matmul(image_hwc + shift.squeeze(0).squeeze(-1).squeeze(-1), matrix)

    # Back to [B, C, H, W]
    return result.permute(0, 3, 1, 2)


def downsampling_420(image):
    """
    Chroma subsampling (4:2:0).

    Args:
        image: [B, 3, H, W] - YCbCr image

    Returns:
        (y, cb, cr) where:
            y: [B, H, W]
            cb: [B, H/2, W/2]
            cr: [B, H/2, W/2]
    """
    y, cb, cr = torch.chunk(image, 3, dim=1)  # Each is [B, 1, H, W]

    # Average pooling for chroma channels
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2)

    return y.squeeze(1), cb.squeeze(1), cr.squeeze(1)


def upsampling_420(y, cb, cr):
    """
    Chroma upsampling (4:2:0 to 4:4:4).

    Args:
        y: [B, H, W]
        cb: [B, H/2, W/2]
        cr: [B, H/2, W/2]

    Returns:
        image: [B, 3, H, W] - YCbCr image
    """
    def repeat(x, k=2):
        # Simple nearest neighbor upsampling
        # [B, H, W] -> [B, 1, H, W] -> [B, 1, H*k, W*k]
        x = x.unsqueeze(1)
        x = F.interpolate(x, scale_factor=k, mode='nearest')
        return x.squeeze(1)

    cb = repeat(cb)
    cr = repeat(cr)

    return torch.stack([y, cb, cr], dim=1)


def image_to_patches(image):
    """
    Split image into 8x8 patches.

    Args:
        image: [B, H, W]

    Returns:
        patches: [B, num_patches, 8, 8]
    """
    k = 8
    batch_size, height, width = image.shape

    # Reshape: [B, H, W] -> [B, H//k, k, W//k, k]
    image_reshaped = image.reshape(batch_size, height // k, k, width // k, k)
    # Transpose: [B, H//k, k, W//k, k] -> [B, H//k, W//k, k, k]
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    # Reshape: [B, H//k, W//k, k, k] -> [B, (H//k)*(W//k), k, k]
    return image_transposed.reshape(batch_size, -1, k, k)


def patches_to_image(patches, height, width):
    """
    Reconstruct image from 8x8 patches.

    Args:
        patches: [B, num_patches, 8, 8]
        height: target height
        width: target width

    Returns:
        image: [B, height, width]
    """
    k = 8
    batch_size = patches.shape[0]

    # Reshape: [B, num_patches, k, k] -> [B, H//k, W//k, k, k]
    image_reshaped = patches.reshape(batch_size, height // k, width // k, k, k)
    # Transpose: [B, H//k, W//k, k, k] -> [B, H//k, k, W//k, k]
    image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
    # Reshape: [B, H//k, k, W//k, k] -> [B, H, W]
    return image_transposed.reshape(batch_size, height, width)


def dct_8x8(image):
    """
    2D Discrete Cosine Transform on 8x8 patches.

    Args:
        image: [B, num_patches, 8, 8]

    Returns:
        dct: [B, num_patches, 8, 8]
    """
    # Pre-compute DCT tensor
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
            (2 * y + 1) * v * np.pi / 16)

    tensor = torch.tensor(tensor, dtype=image.dtype, device=image.device)

    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    scale = np.outer(alpha, alpha) * 0.25
    scale = torch.tensor(scale, dtype=image.dtype, device=image.device)

    # Center around 0
    image = image - 128

    # Apply DCT
    result = scale * torch.tensordot(image, tensor, dims=[[2, 3], [0, 1]])

    return result


def idct_8x8(image):
    """
    2D Inverse Discrete Cosine Transform on 8x8 patches.

    Args:
        image: [B, num_patches, 8, 8]

    Returns:
        reconstructed: [B, num_patches, 8, 8]
    """
    alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
    alpha = np.outer(alpha, alpha)
    alpha = torch.tensor(alpha, dtype=image.dtype, device=image.device)

    image = image * alpha

    # Pre-compute IDCT tensor
    tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
    for x, y, u, v in itertools.product(range(8), repeat=4):
        tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
            (2 * v + 1) * y * np.pi / 16)

    tensor = torch.tensor(tensor, dtype=image.dtype, device=image.device)

    result = 0.25 * torch.tensordot(image, tensor, dims=[[2, 3], [0, 1]]) + 128

    return result


def diff_round(x):
    """Differentiable rounding function."""
    return torch.round(x) + (x - torch.round(x)) ** 3


def round_only_at_0(x):
    """Differentiable rounding that only rounds near 0."""
    cond = (torch.abs(x) < 0.5).float()
    return cond * (x ** 3) + (1 - cond) * x


def y_quantize(image, rounding, factor=1):
    """
    Quantize Y channel DCT coefficients.

    Args:
        image: [B, num_patches, 8, 8] - DCT coefficients
        rounding: rounding function (diff_round or round_only_at_0)
        factor: JPEG quality factor

    Returns:
        quantized: [B, num_patches, 8, 8]
    """
    table = torch.tensor(y_table, dtype=image.dtype, device=image.device)
    image = image / (table * factor)
    image = rounding(image)
    return image


def c_quantize(image, rounding, factor=1):
    """
    Quantize Cb/Cr channel DCT coefficients.

    Args:
        image: [B, num_patches, 8, 8] - DCT coefficients
        rounding: rounding function
        factor: JPEG quality factor

    Returns:
        quantized: [B, num_patches, 8, 8]
    """
    table = torch.tensor(c_table, dtype=image.dtype, device=image.device)
    image = image / (table * factor)
    image = rounding(image)
    return image


def y_dequantize(image, factor=1):
    """Dequantize Y channel."""
    table = torch.tensor(y_table, dtype=image.dtype, device=image.device)
    return image * (table * factor)


def c_dequantize(image, factor=1):
    """Dequantize Cb/Cr channel."""
    table = torch.tensor(c_table, dtype=image.dtype, device=image.device)
    return image * (table * factor)


def jpeg_compress_decompress(image, downsample_c=True, rounding=diff_round, factor=1):
    """
    Differentiable JPEG compression and decompression.

    Args:
        image: [B, C, H, W] - RGB image in [0, 1]
        downsample_c: whether to use 4:2:0 chroma subsampling
        rounding: rounding function for quantization
        factor: JPEG quality factor

    Returns:
        compressed: [B, C, H, W] - JPEG compressed image in [0, 1]
    """
    # Convert to [0, 255]
    image = image * 255

    batch_size, channels, orig_height, orig_width = image.shape
    height, width = orig_height, orig_width

    # Pad to multiple of 16
    if height % 16 != 0 or width % 16 != 0:
        height = ((height - 1) // 16 + 1) * 16
        width = ((width - 1) // 16 + 1) * 16

        vpad = height - orig_height
        wpad = width - orig_width

        # Symmetric padding: [B, C, H, W] format uses (left, right, top, bottom)
        image = F.pad(image, (0, wpad, 0, vpad), mode='reflect')

    # RGB -> YCbCr
    image = rgb_to_ycbcr_jpeg(image)

    # Chroma subsampling
    if downsample_c:
        y, cb, cr = downsampling_420(image)
    else:
        y, cb, cr = torch.chunk(image, 3, dim=1)
        y, cb, cr = y.squeeze(1), cb.squeeze(1), cr.squeeze(1)

    components = {'y': y, 'cb': cb, 'cr': cr}

    # Compression: DCT + Quantization
    for k in components.keys():
        comp = components[k]
        comp = image_to_patches(comp)
        comp = dct_8x8(comp)

        if k in ('cb', 'cr'):
            comp = c_quantize(comp, rounding, factor)
        else:
            comp = y_quantize(comp, rounding, factor)

        components[k] = comp

    # Decompression: Dequantization + IDCT
    for k in components.keys():
        comp = components[k]

        if k in ('cb', 'cr'):
            comp = c_dequantize(comp, factor)
        else:
            comp = y_dequantize(comp, factor)

        comp = idct_8x8(comp)

        # Reconstruct image from patches
        if k in ('cb', 'cr'):
            if downsample_c:
                comp = patches_to_image(comp, height // 2, width // 2)
            else:
                comp = patches_to_image(comp, height, width)
        else:
            comp = patches_to_image(comp, height, width)

        components[k] = comp

    # Chroma upsampling
    y, cb, cr = components['y'], components['cb'], components['cr']
    if downsample_c:
        image = upsampling_420(y, cb, cr)
    else:
        image = torch.stack([y, cb, cr], dim=1)

    # YCbCr -> RGB
    image = ycbcr_to_rgb_jpeg(image)

    # Crop to original size
    if orig_height != height or orig_width != width:
        image = image[:, :, :orig_height, :orig_width]

    # Clamp to valid range
    image = torch.clamp(image, 0, 255)

    # Convert back to [0, 1]
    image = image / 255

    return image


def quality_to_factor(quality):
    """
    Convert JPEG quality (0-100) to quantization factor.

    Args:
        quality: JPEG quality in [0, 100]

    Returns:
        factor: quantization factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality * 2
    return quality / 100.


if __name__ == '__main__':
    print("Testing PyTorch utilities...")

    # Test blur kernel
    print("\n1. Testing random blur kernel...")
    kernel = random_blur_kernel(probs=[0.25, 0.25], N_blur=7,
                                sigrange_gauss=[1., 3.],
                                sigrange_line=[0.25, 1.], wmin_line=3)
    print(f"   Kernel shape: {kernel.shape}")
    print(f"   Kernel sum: {kernel.sum().item():.4f}")

    # Test perspective transform
    print("\n2. Testing perspective transform...")
    Ms = get_rand_transform_matrix(400, 50, 4)
    print(f"   Transform matrices shape: {Ms.shape}")

    # Test brightness adjustment
    print("\n3. Testing brightness adjustment...")
    brightness = get_rnd_brightness_torch(0.1, 0.05, 4)
    print(f"   Brightness adjustment shape: {brightness.shape}")

    # Test JPEG compression
    print("\n4. Testing differentiable JPEG...")
    image = torch.rand(2, 3, 400, 400)
    compressed = jpeg_compress_decompress(image, downsample_c=True,
                                         rounding=round_only_at_0, factor=1.0)
    print(f"   Input shape: {image.shape}")
    print(f"   Output shape: {compressed.shape}")
    print(f"   MSE: {F.mse_loss(image, compressed).item():.6f}")

    # Test gradient flow through JPEG
    print("\n5. Testing JPEG gradient flow...")
    image = torch.rand(1, 3, 400, 400, requires_grad=True)
    compressed = jpeg_compress_decompress(image, downsample_c=True,
                                         rounding=diff_round, factor=1.0)
    loss = compressed.mean()
    loss.backward()
    print(f"   Input has gradient: {image.grad is not None}")
    print(f"   Gradient mean: {image.grad.mean().item():.6f}")

    print("\n✓ All utility tests passed!")

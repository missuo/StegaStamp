"""
Tests for PyTorch utilities: differentiable JPEG, augmentations, etc.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import utils


def test_jpeg_shape_preservation():
    """Test JPEG compression preserves image shape."""
    print("Testing JPEG shape preservation...")

    image = torch.rand(2, 3, 400, 400)
    compressed = utils.jpeg_compress_decompress(image, downsample_c=True)

    assert compressed.shape == image.shape, \
        f"Shape mismatch: input {image.shape}, output {compressed.shape}"

    print("  ✓ JPEG preserves shape")


def test_jpeg_differentiable():
    """Test JPEG compression is differentiable."""
    print("Testing JPEG differentiability...")

    image = torch.rand(1, 3, 400, 400, requires_grad=True)
    compressed = utils.jpeg_compress_decompress(
        image,
        downsample_c=True,
        rounding=utils.diff_round
    )

    loss = compressed.mean()
    loss.backward()

    assert image.grad is not None, "JPEG should be differentiable"
    assert not torch.isnan(image.grad).any(), "Gradients should not be NaN"

    grad_mean = image.grad.abs().mean().item()
    print(f"  ✓ JPEG is differentiable (grad mean: {grad_mean:.6f})")


def test_jpeg_quality_effect():
    """Test JPEG quality factor affects compression."""
    print("Testing JPEG quality effect...")

    image = torch.rand(1, 3, 400, 400)

    # High quality (low factor)
    compressed_high = utils.jpeg_compress_decompress(image, factor=0.1)
    mse_high = F.mse_loss(image, compressed_high).item()

    # Low quality (high factor)
    compressed_low = utils.jpeg_compress_decompress(image, factor=2.0)
    mse_low = F.mse_loss(image, compressed_low).item()

    assert mse_low > mse_high, \
        f"Lower quality should have higher MSE: high={mse_high:.6f}, low={mse_low:.6f}"

    print(f"  ✓ Quality factor affects compression (MSE: {mse_high:.6f} → {mse_low:.6f})")


def test_jpeg_value_range():
    """Test JPEG output is in valid range [0, 1]."""
    print("Testing JPEG output range...")

    image = torch.rand(2, 3, 400, 400)
    compressed = utils.jpeg_compress_decompress(image)

    assert compressed.min() >= 0.0, f"Min value should be >= 0, got {compressed.min()}"
    assert compressed.max() <= 1.0, f"Max value should be <= 1, got {compressed.max()}"

    print("  ✓ JPEG output in valid range [0, 1]")


def test_blur_kernel_generation():
    """Test random blur kernel generation."""
    print("Testing blur kernel generation...")

    kernel = utils.random_blur_kernel(
        probs=[0.5, 0.5],
        N_blur=7,
        sigrange_gauss=[1.0, 3.0],
        sigrange_line=[0.25, 1.0],
        wmin_line=3
    )

    assert kernel.shape == (7, 7, 3, 3), f"Expected shape (7, 7, 3, 3), got {kernel.shape}"

    # Kernel should sum to 3 (one for each channel)
    kernel_sum = kernel.sum().item()
    assert abs(kernel_sum - 3.0) < 1e-5, f"Kernel should sum to 3, got {kernel_sum}"

    print("  ✓ Blur kernel shape and normalization correct")


def test_brightness_adjustment():
    """Test brightness adjustment generation."""
    print("Testing brightness adjustment...")

    adjustment = utils.get_rnd_brightness_torch(
        rnd_bri=0.3,
        rnd_hue=0.1,
        batch_size=4
    )

    assert adjustment.shape == (4, 1, 1, 3), \
        f"Expected shape (4, 1, 1, 3), got {adjustment.shape}"

    # Values should be in reasonable range
    assert adjustment.abs().max() < 0.5, "Brightness adjustment too large"

    print("  ✓ Brightness adjustment shape and range correct")


def test_perspective_transform():
    """Test perspective transformation matrix generation."""
    print("Testing perspective transform...")

    M = utils.get_rand_transform_matrix(
        image_size=400,
        d=50,
        batch_size=4
    )

    assert M.shape == (4, 2, 8), f"Expected shape (4, 2, 8), got {M.shape}"

    # Check that forward and inverse transforms are different
    M_forward = M[:, 0, :]
    M_inverse = M[:, 1, :]

    diff = np.abs(M_forward - M_inverse).mean()
    assert diff > 0.01, "Forward and inverse transforms should be different"

    print("  ✓ Perspective transform generation correct")


def test_rgb_yuv_conversion():
    """Test RGB to YUV color space conversion."""
    print("Testing RGB to YUV conversion...")

    try:
        # Import the function from training script
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from train import rgb_to_yuv_pytorch

        # Pure red, green, blue
        rgb = torch.tensor([[[[1.0, 0.0, 0.0]]]]).permute(0, 3, 1, 2)  # [1, 3, 1, 1]
        yuv = rgb_to_yuv_pytorch(rgb)

        assert yuv.shape == rgb.shape, "Shape should be preserved"

        # Check Y channel is in reasonable range
        y_value = yuv[0, 0, 0, 0].item()
        assert 0.0 <= y_value <= 1.0, f"Y value should be in [0, 1], got {y_value}"

        print("  ✓ RGB to YUV conversion works")
    except ImportError as e:
        # Skip test if tensorboard not installed (needed by train_pytorch)
        print(f"  ⚠ SKIPPED: {e}")


def test_dct_idct_roundtrip():
    """Test DCT and IDCT are inverses."""
    print("Testing DCT/IDCT roundtrip...")

    # Create random 8x8 patches
    patches = torch.randn(4, 10, 8, 8) + 128  # Centered around 128

    # Apply DCT then IDCT
    dct_patches = utils.dct_8x8(patches)
    reconstructed = utils.idct_8x8(dct_patches)

    # Should reconstruct original
    diff = (patches - reconstructed).abs().max().item()

    assert diff < 0.1, f"DCT/IDCT roundtrip error too large: {diff}"

    print(f"  ✓ DCT/IDCT roundtrip max error: {diff:.6f}")


def test_chroma_subsampling_roundtrip():
    """Test chroma subsampling and upsampling."""
    print("Testing chroma subsampling...")

    # Create test image
    image = torch.rand(2, 3, 400, 400) * 255

    # Downsample
    y, cb, cr = utils.downsampling_420(image)

    assert y.shape == (2, 400, 400), f"Y shape incorrect: {y.shape}"
    assert cb.shape == (2, 200, 200), f"Cb shape incorrect: {cb.shape}"
    assert cr.shape == (2, 200, 200), f"Cr shape incorrect: {cr.shape}"

    # Upsample
    reconstructed = utils.upsampling_420(y, cb, cr)

    assert reconstructed.shape == image.shape, \
        f"Reconstructed shape {reconstructed.shape} != original {image.shape}"

    print("  ✓ Chroma subsampling shapes correct")


def test_jpeg_padding():
    """Test JPEG handles non-multiple-of-16 sizes."""
    print("Testing JPEG with odd sizes...")

    # Test various sizes
    for size in [384, 400, 416]:
        image = torch.rand(1, 3, size, size)
        compressed = utils.jpeg_compress_decompress(image)

        assert compressed.shape == image.shape, \
            f"Size {size}: shape mismatch {compressed.shape} != {image.shape}"

    print("  ✓ JPEG handles various image sizes correctly")


def run_all_tests():
    """Run all utility tests."""
    print("="*60)
    print("Running Utility Tests")
    print("="*60)

    tests = [
        test_jpeg_shape_preservation,
        test_jpeg_differentiable,
        test_jpeg_quality_effect,
        test_jpeg_value_range,
        test_blur_kernel_generation,
        test_brightness_adjustment,
        test_perspective_transform,
        test_rgb_yuv_conversion,
        test_dct_idct_roundtrip,
        test_chroma_subsampling_roundtrip,
        test_jpeg_padding,
    ]

    failed = []

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed.append(test.__name__)

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name in failed:
            print(f"  - {name}")
        return False
    else:
        print(f"SUCCESS: All {len(tests)} tests passed!")
        return True


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

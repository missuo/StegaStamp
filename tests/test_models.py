"""
Tests for PyTorch models: StegaStampEncoder, StegaStampDecoder, Discriminator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import models


def test_encoder_shapes():
    """Test encoder produces correct output shapes."""
    print("Testing encoder output shapes...")

    batch_size = 4
    encoder = models.StegaStampEncoder()

    secret = torch.rand(batch_size, 100)
    image = torch.rand(batch_size, 3, 400, 400)

    residual = encoder(secret, image)

    assert residual.shape == (batch_size, 3, 400, 400), \
        f"Expected shape (4, 3, 400, 400), got {residual.shape}"

    print("  ✓ Encoder output shape correct")


def test_decoder_shapes():
    """Test decoder produces correct output shapes."""
    print("Testing decoder output shapes...")

    batch_size = 4
    decoder = models.StegaStampDecoder(secret_size=100)

    image = torch.rand(batch_size, 3, 400, 400)
    secret_logits = decoder(image)

    assert secret_logits.shape == (batch_size, 100), \
        f"Expected shape (4, 100), got {secret_logits.shape}"

    print("  ✓ Decoder output shape correct")


def test_discriminator_shapes():
    """Test discriminator produces correct output shapes."""
    print("Testing discriminator output shapes...")

    batch_size = 4
    discriminator = models.Discriminator()

    image = torch.rand(batch_size, 3, 400, 400)
    output, heatmap = discriminator(image)

    assert isinstance(output, torch.Tensor), "Output should be a tensor"
    assert output.dim() == 0, "Output should be a scalar"
    assert heatmap.shape == (batch_size, 1, 25, 25), \
        f"Expected heatmap shape (4, 1, 25, 25), got {heatmap.shape}"

    print("  ✓ Discriminator output shapes correct")


def test_gradient_flow():
    """Test gradients flow through the entire pipeline."""
    print("Testing gradient flow...")

    encoder = models.StegaStampEncoder()
    decoder = models.StegaStampDecoder()

    secret = torch.rand(2, 100)
    image = torch.rand(2, 3, 400, 400)

    # Forward pass
    residual = encoder(secret, image)
    encoded_image = image + residual
    decoded_secret = decoder(encoded_image)

    # Compute loss
    loss = F.binary_cross_entropy_with_logits(decoded_secret, secret)

    # Backward pass
    loss.backward()

    # Check gradients exist
    encoder_has_grad = any(p.grad is not None for p in encoder.parameters())
    decoder_has_grad = any(p.grad is not None for p in decoder.parameters())

    assert encoder_has_grad, "Encoder should have gradients"
    assert decoder_has_grad, "Decoder should have gradients"

    print("  ✓ Gradients flow through encoder and decoder")


def test_encoder_residual_range():
    """Test encoder residual is in reasonable range."""
    print("Testing encoder residual range...")

    encoder = models.StegaStampEncoder()
    encoder.eval()

    with torch.no_grad():
        secret = torch.rand(2, 100)
        image = torch.rand(2, 3, 400, 400) * 0.5  # Mid-range image

        residual = encoder(secret, image)

        # Residual should be small relative to image intensity
        # Note: untrained model may have larger residuals
        residual_magnitude = residual.abs().mean().item()

        assert residual_magnitude < 1.0, \
            f"Residual magnitude too large: {residual_magnitude:.4f}"

        print(f"  ✓ Residual magnitude: {residual_magnitude:.4f} (reasonable)")


def test_decoder_stn():
    """Test decoder's spatial transformer network."""
    print("Testing decoder STN...")

    decoder = models.StegaStampDecoder()

    # Create shifted image
    image = torch.zeros(1, 3, 400, 400)
    image[:, :, 100:300, 100:300] = 1.0  # White square

    with torch.no_grad():
        secret1 = decoder(image)

        # Shift image
        image_shifted = torch.roll(image, shifts=50, dims=2)
        secret2 = decoder(image_shifted)

        # STN should help maintain similarity despite shift
        diff = (secret1 - secret2).abs().mean().item()

        print(f"  ✓ STN output difference after shift: {diff:.4f}")


def test_bit_accuracy_calculation():
    """Test bit accuracy calculation."""
    print("Testing bit accuracy calculation...")

    # Perfect prediction
    secret_true = torch.ones(4, 100)
    secret_pred = torch.ones(4, 100) * 10.0  # Large positive logits

    bit_acc, str_acc = models.get_secret_acc(secret_true, secret_pred)

    assert bit_acc.item() == 1.0, f"Expected bit_acc=1.0, got {bit_acc.item()}"
    assert str_acc.item() == 1.0, f"Expected str_acc=1.0, got {str_acc.item()}"

    # Random prediction (should be ~50% bit accuracy)
    secret_true = torch.randint(0, 2, (100, 100)).float()
    secret_pred = torch.randn(100, 100)

    bit_acc, str_acc = models.get_secret_acc(secret_true, secret_pred)

    assert 0.4 < bit_acc.item() < 0.6, \
        f"Random predictions should give ~50% bit accuracy, got {bit_acc.item():.3f}"

    print(f"  ✓ Bit accuracy: {bit_acc.item():.3f}, String accuracy: {str_acc.item():.3f}")


def test_model_device_transfer():
    """Test models can be moved to different devices."""
    print("Testing device transfer...")

    encoder = models.StegaStampEncoder()

    # Test CPU
    encoder_cpu = encoder.cpu()
    secret = torch.rand(1, 100)
    image = torch.rand(1, 3, 400, 400)
    residual = encoder_cpu(secret, image)

    assert residual.device.type == 'cpu', "Output should be on CPU"

    # Test CUDA if available
    if torch.cuda.is_available():
        encoder_cuda = encoder.cuda()
        secret_cuda = secret.cuda()
        image_cuda = image.cuda()
        residual_cuda = encoder_cuda(secret_cuda, image_cuda)

        assert residual_cuda.device.type == 'cuda', "Output should be on CUDA"
        print("  ✓ Models work on CPU and CUDA")
    else:
        print("  ✓ Models work on CPU (CUDA not available)")


def test_encoder_deterministic():
    """Test encoder produces same output for same input."""
    print("Testing encoder determinism...")

    encoder = models.StegaStampEncoder()
    encoder.eval()

    secret = torch.rand(1, 100)
    image = torch.rand(1, 3, 400, 400)

    with torch.no_grad():
        residual1 = encoder(secret, image)
        residual2 = encoder(secret, image)

    diff = (residual1 - residual2).abs().max().item()

    assert diff < 1e-6, f"Encoder should be deterministic, max diff: {diff}"

    print("  ✓ Encoder is deterministic")


def run_all_tests():
    """Run all model tests."""
    print("="*60)
    print("Running Model Tests")
    print("="*60)

    tests = [
        test_encoder_shapes,
        test_decoder_shapes,
        test_discriminator_shapes,
        test_gradient_flow,
        test_encoder_residual_range,
        test_decoder_stn,
        test_bit_accuracy_calculation,
        test_model_device_transfer,
        test_encoder_deterministic,
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

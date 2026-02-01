import argparse
import os

import torch
import onnx
import onnxruntime as ort
import numpy as np

import models


def export_encoder(encoder, output_path, opset_version=14):
    """
    Export encoder model to ONNX format.

    Args:
        encoder: PyTorch encoder model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    print("\n=== Exporting Encoder to ONNX ===")

    encoder.eval()

    # Create dummy inputs
    dummy_secret = torch.randn(1, 100)
    dummy_image = torch.randn(1, 3, 400, 400)

    # Export to ONNX
    torch.onnx.export(
        encoder,
        (dummy_secret, dummy_image),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['secret', 'image'],
        output_names=['residual'],
        dynamic_axes={
            'secret': {0: 'batch_size'},
            'image': {0: 'batch_size'},
            'residual': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ Encoder exported to {output_path}")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

    return output_path


def export_decoder(decoder, output_path, opset_version=14):
    """
    Export decoder model to ONNX format.

    Args:
        decoder: PyTorch decoder model
        output_path: Path to save ONNX model
        opset_version: ONNX opset version
    """
    print("\n=== Exporting Decoder to ONNX ===")

    decoder.eval()

    # Create dummy input
    dummy_image = torch.randn(1, 3, 400, 400)

    # Export to ONNX
    torch.onnx.export(
        decoder,
        dummy_image,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['secret_logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'secret_logits': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ Decoder exported to {output_path}")

    # Verify the model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model verification passed")

    return output_path


def test_onnx_models(encoder_path, decoder_path, pytorch_encoder, pytorch_decoder):
    """
    Test ONNX models against PyTorch models to ensure numerical equivalence.

    Args:
        encoder_path: Path to ONNX encoder
        decoder_path: Path to ONNX decoder
        pytorch_encoder: PyTorch encoder for comparison
        pytorch_decoder: PyTorch decoder for comparison
    """
    print("\n=== Testing ONNX Models ===")

    # Create ONNX Runtime sessions
    encoder_session = ort.InferenceSession(encoder_path)
    decoder_session = ort.InferenceSession(decoder_path)

    # Test encoder
    print("\nTesting encoder...")
    test_secret = np.random.randn(2, 100).astype(np.float32)
    test_image = np.random.randn(2, 3, 400, 400).astype(np.float32)

    # PyTorch inference
    pytorch_encoder.eval()
    with torch.no_grad():
        pytorch_residual = pytorch_encoder(
            torch.from_numpy(test_secret),
            torch.from_numpy(test_image)
        ).numpy()

    # ONNX inference
    onnx_residual = encoder_session.run(
        ['residual'],
        {'secret': test_secret, 'image': test_image}
    )[0]

    # Compare outputs
    diff = np.abs(pytorch_residual - onnx_residual)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("  ✓ Encoder outputs match (rtol=1e-3)")
    else:
        print(f"  ⚠ Warning: Large difference between PyTorch and ONNX encoder outputs")

    # Test decoder
    print("\nTesting decoder...")
    test_image = np.random.randn(2, 3, 400, 400).astype(np.float32)

    # PyTorch inference
    pytorch_decoder.eval()
    with torch.no_grad():
        pytorch_secret = pytorch_decoder(torch.from_numpy(test_image)).numpy()

    # ONNX inference
    onnx_secret = decoder_session.run(
        ['secret_logits'],
        {'image': test_image}
    )[0]

    # Compare outputs
    diff = np.abs(pytorch_secret - onnx_secret)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("  ✓ Decoder outputs match (rtol=1e-3)")
    else:
        print(f"  ⚠ Warning: Large difference between PyTorch and ONNX decoder outputs")

    print("\n✓ ONNX model testing complete")


def main():
    parser = argparse.ArgumentParser(description='Export StegaStamp models to ONNX')
    parser.add_argument('checkpoint', type=str, help='Path to PyTorch checkpoint (.pth file)')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                       help='Directory to save ONNX models')
    parser.add_argument('--secret_size', type=int, default=100, help='Size of secret in bits')
    parser.add_argument('--opset_version', type=int, default=14,
                       help='ONNX opset version (default: 14)')
    parser.add_argument('--test', action='store_true',
                       help='Test ONNX models against PyTorch for numerical equivalence')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create models
    encoder = models.StegaStampEncoder(height=400, width=400).to(device)
    decoder = models.StegaStampDecoder(secret_size=args.secret_size, height=400, width=400).to(device)

    # Load weights
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder.eval()
    decoder.eval()

    print("✓ Models loaded successfully")

    # Export paths
    encoder_path = os.path.join(args.output_dir, 'encoder.onnx')
    decoder_path = os.path.join(args.output_dir, 'decoder.onnx')

    # Export encoder
    export_encoder(encoder, encoder_path, opset_version=args.opset_version)

    # Export decoder
    export_decoder(decoder, decoder_path, opset_version=args.opset_version)

    # Test if requested
    if args.test:
        # Move models to CPU for fair comparison
        encoder_cpu = encoder.cpu()
        decoder_cpu = decoder.cpu()
        test_onnx_models(encoder_path, decoder_path, encoder_cpu, decoder_cpu)

    print("\n" + "="*50)
    print("ONNX Export Complete!")
    print("="*50)
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")
    print(f"\nUsage example:")
    print(f"  python onnx_inference.py {encoder_path} {decoder_path} --image test.jpg")


if __name__ == '__main__':
    main()

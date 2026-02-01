import argparse
import bchlib
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def encode_with_onnx(encoder_session, image_path, secret_text, save_path=None):
    """
    Encode a secret into an image using ONNX encoder.

    Args:
        encoder_session: ONNX Runtime session for encoder
        image_path: Path to input image
        secret_text: Secret text to encode (max 7 characters)
        save_path: Path to save encoded image (optional)

    Returns:
        encoded_image: Numpy array of encoded image [H, W, 3]
    """
    # Setup BCH error correction
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(secret_text) > 7:
        raise ValueError('Can only encode 56 bits (7 characters) with ECC')

    # Encode secret with BCH
    data = bytearray(secret_text + ' ' * (7 - len(secret_text)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    # Convert to binary
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])  # Pad to 100 bits

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (400, 400))
    image_np = np.array(image, dtype=np.float32) / 255.0

    # Convert to CHW format for ONNX
    image_chw = np.transpose(image_np, (2, 0, 1))  # [3, 400, 400]
    image_batch = np.expand_dims(image_chw, axis=0)  # [1, 3, 400, 400]

    # Prepare secret
    secret_np = np.array([secret], dtype=np.float32)  # [1, 100]

    # Run ONNX inference
    residual = encoder_session.run(
        ['residual'],
        {'secret': secret_np, 'image': image_batch}
    )[0]

    # Add residual to image
    encoded_image = image_batch + residual
    encoded_image = np.clip(encoded_image, 0, 1)

    # Convert back to HWC format
    encoded_image_hwc = np.transpose(encoded_image[0], (1, 2, 0))  # [400, 400, 3]

    # Save if path provided
    if save_path is not None:
        encoded_uint8 = (encoded_image_hwc * 255).astype(np.uint8)
        Image.fromarray(encoded_uint8).save(save_path)
        print(f"✓ Encoded image saved to {save_path}")

    return encoded_image_hwc


def decode_with_onnx(decoder_session, image_path):
    """
    Decode a secret from an image using ONNX decoder.

    Args:
        decoder_session: ONNX Runtime session for decoder
        image_path: Path to encoded image

    Returns:
        decoded_text: Decoded secret text (or None if decoding failed)
    """
    # Setup BCH error correction
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, (400, 400))
    image_np = np.array(image, dtype=np.float32) / 255.0

    # Convert to CHW format for ONNX
    image_chw = np.transpose(image_np, (2, 0, 1))  # [3, 400, 400]
    image_batch = np.expand_dims(image_chw, axis=0)  # [1, 3, 400, 400]

    # Run ONNX inference
    secret_logits = decoder_session.run(
        ['secret_logits'],
        {'image': image_batch}
    )[0]

    # Apply sigmoid and round
    secret = np.round(1 / (1 + np.exp(-secret_logits))).astype(np.int32)  # Sigmoid + round
    secret = secret[0]  # Remove batch dimension

    # Extract first 96 bits for BCH decoding
    packet_binary = "".join([str(int(bit)) for bit in secret[:96]])

    # Convert to bytes
    packet = bytes(int(packet_binary[i:i + 8], 2) for i in range(0, len(packet_binary), 8))
    packet = bytearray(packet)

    # Split into data and ECC
    data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

    # Attempt BCH error correction
    bitflips = bch.decode_inplace(data, ecc)

    if bitflips != -1:
        try:
            decoded_text = data.decode("utf-8").rstrip()
            print(f"✓ Decoded: '{decoded_text}' (corrected {bitflips} bit errors)")
            return decoded_text
        except UnicodeDecodeError:
            print("✗ Decoded but not valid UTF-8")
            return None
    else:
        print("✗ Failed to decode (too many errors)")
        return None


def main():
    parser = argparse.ArgumentParser(description='StegaStamp ONNX Inference')
    parser.add_argument('encoder', type=str, help='Path to ONNX encoder model')
    parser.add_argument('decoder', type=str, help='Path to ONNX decoder model')

    # Operation mode
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--encode', action='store_true', help='Encode mode')
    group.add_argument('--decode', action='store_true', help='Decode mode')
    group.add_argument('--test', action='store_true', help='Test encode/decode roundtrip')

    # Common arguments
    parser.add_argument('--image', type=str, help='Input image path')
    parser.add_argument('--secret', type=str, default='Stega!!', help='Secret text (max 7 chars)')
    parser.add_argument('--output', type=str, help='Output image path (for encode mode)')

    args = parser.parse_args()

    # Create ONNX Runtime sessions
    print("Loading ONNX models...")
    encoder_session = ort.InferenceSession(args.encoder)
    decoder_session = ort.InferenceSession(args.decoder)
    print("✓ Models loaded\n")

    if args.encode:
        # Encode mode
        if args.image is None:
            print("Error: --image required for encode mode")
            return

        print(f"Encoding secret '{args.secret}' into {args.image}")
        encode_with_onnx(encoder_session, args.image, args.secret, args.output)

    elif args.decode:
        # Decode mode
        if args.image is None:
            print("Error: --image required for decode mode")
            return

        print(f"Decoding secret from {args.image}")
        decode_with_onnx(decoder_session, args.image)

    elif args.test:
        # Test roundtrip encode/decode
        if args.image is None:
            print("Error: --image required for test mode")
            return

        print(f"Testing roundtrip encode/decode with secret '{args.secret}'")
        print("="*50)

        # Encode
        print("\n1. Encoding...")
        temp_output = args.output or 'temp_encoded.png'
        encode_with_onnx(encoder_session, args.image, args.secret, temp_output)

        # Decode
        print("\n2. Decoding...")
        decoded_text = decode_with_onnx(decoder_session, temp_output)

        # Compare
        print("\n" + "="*50)
        print("ROUNDTRIP TEST RESULTS")
        print("="*50)
        print(f"Original secret: '{args.secret}'")
        print(f"Decoded secret:  '{decoded_text}'")

        if decoded_text == args.secret:
            print("\n✓ SUCCESS: Roundtrip encode/decode successful!")
        else:
            print("\n✗ FAILURE: Decoded text does not match original")

        # Clean up temp file if we created it
        if args.output is None and os.path.exists(temp_output):
            os.remove(temp_output)
            print(f"\n(Temporary file {temp_output} removed)")


if __name__ == '__main__':
    main()

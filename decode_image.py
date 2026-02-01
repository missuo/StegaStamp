import argparse
import bchlib
import glob
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import models


BCH_POLYNOMIAL = 137
BCH_BITS = 5


def main():
    parser = argparse.ArgumentParser(description='Decode secret from image using StegaStamp PyTorch')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--image', type=str, default=None, help='Path to single image')
    parser.add_argument('--images_dir', type=str, default=None, help='Path to directory of images')
    parser.add_argument('--secret_size', type=int, default=100, help='Size of secret in bits')
    args = parser.parse_args()

    # Get list of images to process
    if args.image is not None:
        files_list = [args.image]
    elif args.images_dir is not None:
        files_list = glob.glob(os.path.join(args.images_dir, '*'))
        # Filter for image files
        files_list = [f for f in files_list if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        print('Error: Must specify either --image or --images_dir')
        return

    if len(files_list) == 0:
        print('Error: No images found')
        return

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create decoder
    decoder = models.StegaStampDecoder(secret_size=args.secret_size, height=400, width=400).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()

    print("Decoder loaded successfully\n")

    # Setup BCH error correction
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    # Process each image
    with torch.no_grad():
        for filename in files_list:
            # Load and preprocess image
            try:
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, (400, 400))
                image_np = np.array(image, dtype=np.float32) / 255.0
            except Exception as e:
                print(f"{filename}: Error loading image - {e}")
                continue

            # Convert to PyTorch tensor: HWC -> CHW
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Decode secret
            secret_logits = decoder(image_tensor)  # [1, secret_size]

            # Apply sigmoid and round to get binary secret
            secret = torch.round(torch.sigmoid(secret_logits)).squeeze(0).cpu().numpy()  # [secret_size]

            # Extract first 96 bits (12 bytes) for BCH decoding
            packet_binary = "".join([str(int(bit)) for bit in secret[:96]])

            # Convert binary string to bytes
            packet = bytes(int(packet_binary[i:i + 8], 2) for i in range(0, len(packet_binary), 8))
            packet = bytearray(packet)

            # Split into data and ECC
            data, ecc = packet[:-bch.ecc_bytes], packet[-bch.ecc_bytes:]

            # Attempt BCH error correction
            bitflips = bch.decode_inplace(data, ecc)

            if bitflips != -1:
                try:
                    # Decode as UTF-8 string
                    code = data.decode("utf-8")
                    print(f"{filename}: '{code}' (corrected {bitflips} bit errors)")
                except UnicodeDecodeError:
                    print(f"{filename}: Decoded but not valid UTF-8")
            else:
                print(f"{filename}: Failed to decode (too many errors)")

    print(f"\n✓ Processed {len(files_list)} images")


if __name__ == "__main__":
    main()

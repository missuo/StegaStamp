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
    parser = argparse.ArgumentParser(description='Encode secret into image using StegaStamp PyTorch')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--image', type=str, default=None, help='Path to single image')
    parser.add_argument('--images_dir', type=str, default=None, help='Path to directory of images')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save encoded images')
    parser.add_argument('--secret', type=str, default='Stega!!', help='Secret message to encode (max 7 chars)')
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

    # Create encoder
    encoder = models.StegaStampEncoder(height=400, width=400).to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

    print("Encoder loaded successfully")

    # Setup BCH error correction
    bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

    if len(args.secret) > 7:
        print('Error: Can only encode 56 bits (7 characters) with ECC')
        return

    # Encode secret with BCH error correction
    data = bytearray(args.secret + ' ' * (7 - len(args.secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    # Convert to binary
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    # Pad to 100 bits
    secret.extend([0, 0, 0, 0])

    print(f"Secret message: '{args.secret}'")
    print(f"Encoded as {len(secret)} bits with BCH error correction")

    # Convert secret to tensor
    secret_tensor = torch.FloatTensor([secret]).to(device)

    # Create save directory if needed
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

    width, height = 400, 400

    # Process each image
    with torch.no_grad():
        for idx, filename in enumerate(files_list):
            print(f"\nProcessing {idx+1}/{len(files_list)}: {filename}")

            # Load and preprocess image
            try:
                image = Image.open(filename).convert("RGB")
                image = ImageOps.fit(image, (width, height))
                image_np = np.array(image, dtype=np.float32) / 255.0
            except Exception as e:
                print(f"Error loading image: {e}")
                continue

            # Convert to PyTorch tensor: HWC -> CHW
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

            # Encode secret
            residual = encoder(secret_tensor, image_tensor)  # [1, 3, H, W]
            hidden_img = image_tensor + residual
            hidden_img = torch.clamp(hidden_img, 0, 1)

            # Convert back to numpy: CHW -> HWC
            hidden_img_np = hidden_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            residual_np = residual.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Scale to [0, 255]
            hidden_img_uint8 = (hidden_img_np * 255).astype(np.uint8)
            residual_uint8 = ((residual_np + 0.5) * 255).astype(np.uint8)

            # Save images if save_dir is specified
            if args.save_dir is not None:
                save_name = os.path.splitext(os.path.basename(filename))[0]

                # Save hidden image
                hidden_pil = Image.fromarray(hidden_img_uint8)
                hidden_path = os.path.join(args.save_dir, f'{save_name}_hidden.png')
                hidden_pil.save(hidden_path)

                # Save residual
                residual_pil = Image.fromarray(residual_uint8)
                residual_path = os.path.join(args.save_dir, f'{save_name}_residual.png')
                residual_pil.save(residual_path)

                print(f"  Saved: {hidden_path}")
                print(f"  Saved: {residual_path}")
            else:
                print("  Encoding successful (no save_dir specified)")

    print(f"\n✓ Processed {len(files_list)} images")


if __name__ == "__main__":
    main()

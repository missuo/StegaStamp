import argparse
import glob
import os
from os.path import join

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import models
import utils
from dataset import StegaStampDataset

# Try to import lpips, provide helpful error if not available
try:
    import lpips
except ImportError:
    print("ERROR: lpips package not found. Install with: pip install lpips")
    print("This package is required for perceptual loss calculation.")
    raise


TRAIN_PATH = './data/mirflickr/images1/images/'
LOGS_PATH = "./logs/"
CHECKPOINTS_PATH = './checkpoints/'


def get_ramp_value(global_step, ramp_steps, max_value):
    """Calculate ramped value that gradually increases from 0 to max_value."""
    return min(max_value * global_step / ramp_steps, max_value)


def apply_perspective_transform(image, M_flat, device='cpu'):
    """
    Apply perspective transformation using transformation matrix.

    Args:
        image: [B, C, H, W] - input image
        M_flat: [B, 8] - flattened 3x3 transformation matrix (first 8 elements)
        device: torch device

    Returns:
        transformed: [B, C, H, W] - transformed image
    """
    batch_size, channels, height, width = image.shape

    # Reconstruct 3x3 matrix from flat representation
    M = torch.zeros(batch_size, 3, 3, device=device)
    M[:, 0, 0] = M_flat[:, 0]
    M[:, 0, 1] = M_flat[:, 1]
    M[:, 0, 2] = M_flat[:, 2]
    M[:, 1, 0] = M_flat[:, 3]
    M[:, 1, 1] = M_flat[:, 4]
    M[:, 1, 2] = M_flat[:, 5]
    M[:, 2, 0] = M_flat[:, 6]
    M[:, 2, 1] = M_flat[:, 7]
    M[:, 2, 2] = 1.0

    # Convert perspective matrix to grid
    # Create normalized coordinate grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device),
        indexing='ij'
    )

    # Convert from [-1, 1] to [0, H-1], [0, W-1]
    grid_x = (grid_x + 1) * (width - 1) / 2
    grid_y = (grid_y + 1) * (height - 1) / 2

    # Create homogeneous coordinates
    ones = torch.ones_like(grid_x)
    coords = torch.stack([grid_x, grid_y, ones], dim=-1)  # [H, W, 3]
    coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, W, 3]

    # Apply transformation: new_coords = M @ coords^T
    coords_transformed = torch.matmul(coords, M.transpose(1, 2))  # [B, H, W, 3]

    # Convert from homogeneous to Cartesian
    grid_x_new = coords_transformed[..., 0] / (coords_transformed[..., 2] + 1e-8)
    grid_y_new = coords_transformed[..., 1] / (coords_transformed[..., 2] + 1e-8)

    # Normalize back to [-1, 1] for grid_sample
    grid_x_new = 2 * grid_x_new / (width - 1) - 1
    grid_y_new = 2 * grid_y_new / (height - 1) - 1

    grid = torch.stack([grid_x_new, grid_y_new], dim=-1)  # [B, H, W, 2]

    # Sample from image
    transformed = F.grid_sample(image, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

    return transformed


def transform_net(encoded_image, args, global_step, device='cpu'):
    """
    Apply data augmentation transformations to encoded image.

    Includes: blur, noise, contrast, brightness, saturation, JPEG compression.
    All augmentations are gradually ramped up during training.
    """
    batch_size = encoded_image.shape[0]

    # Calculate ramp factors for each augmentation
    ramp_fn = lambda ramp: min(float(global_step) / ramp, 1.0)

    # Brightness and hue
    rnd_bri = ramp_fn(args.rnd_bri_ramp) * args.rnd_bri
    rnd_hue = ramp_fn(args.rnd_hue_ramp) * args.rnd_hue
    rnd_brightness = utils.get_rnd_brightness_torch(rnd_bri, rnd_hue, batch_size, device=device)

    # JPEG quality
    jpeg_quality = 100.0 - torch.rand(1, device=device).item() * ramp_fn(args.jpeg_quality_ramp) * (100.0 - args.jpeg_quality)
    if jpeg_quality < 50:
        jpeg_factor = 5000.0 / jpeg_quality
    else:
        jpeg_factor = 200.0 - jpeg_quality * 2
    jpeg_factor = jpeg_factor / 100.0 + 0.0001

    # Noise
    rnd_noise = torch.rand(1, device=device).item() * ramp_fn(args.rnd_noise_ramp) * args.rnd_noise

    # Contrast
    contrast_low = 1.0 - (1.0 - args.contrast_low) * ramp_fn(args.contrast_ramp)
    contrast_high = 1.0 + (args.contrast_high - 1.0) * ramp_fn(args.contrast_ramp)

    # Saturation
    rnd_sat = torch.rand(1, device=device).item() * ramp_fn(args.rnd_sat_ramp) * args.rnd_sat

    # Apply blur
    blur_kernel = utils.random_blur_kernel(
        probs=[0.25, 0.25], N_blur=7,
        sigrange_gauss=[1.0, 3.0],
        sigrange_line=[0.25, 1.0],
        wmin_line=3,
        device=device
    )  # [7, 7, 3, 3]

    # Prepare kernel for conv2d: [out_channels, in_channels, H, W]
    # We want a depthwise convolution, so reshape kernel
    blur_kernel = blur_kernel.permute(2, 3, 0, 1)  # [3, 3, 7, 7]

    # Apply convolution per channel (groups=3 for depthwise)
    encoded_image = F.conv2d(encoded_image, blur_kernel, padding=3, groups=3)

    # Add noise
    noise = torch.randn_like(encoded_image) * rnd_noise
    encoded_image = encoded_image + noise
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # Apply contrast
    contrast_scale = torch.rand(batch_size, device=device) * (contrast_high - contrast_low) + contrast_low
    contrast_scale = contrast_scale.view(batch_size, 1, 1, 1)
    encoded_image = encoded_image * contrast_scale

    # Apply brightness (convert from NHWC to NCHW format first)
    rnd_brightness = rnd_brightness.permute(0, 3, 1, 2)  # [B, 3, 1, 1]
    encoded_image = encoded_image + rnd_brightness
    encoded_image = torch.clamp(encoded_image, 0, 1)

    # Apply saturation (desaturate by converting to luminance)
    # Luminance weights: [0.3, 0.6, 0.1] for RGB
    weights = torch.tensor([0.3, 0.6, 0.1], device=device).view(1, 3, 1, 1)
    encoded_image_lum = torch.sum(encoded_image * weights, dim=1, keepdim=True)
    encoded_image = (1 - rnd_sat) * encoded_image + rnd_sat * encoded_image_lum

    # Apply JPEG compression
    if not args.no_jpeg:
        encoded_image = utils.jpeg_compress_decompress(
            encoded_image,
            rounding=utils.round_only_at_0,
            factor=jpeg_factor,
            downsample_c=True
        )

    summaries = {
        'rnd_bri': rnd_bri,
        'rnd_sat': rnd_sat,
        'rnd_hue': rnd_hue,
        'rnd_noise': rnd_noise,
        'contrast_low': contrast_low,
        'contrast_high': contrast_high,
        'jpeg_quality': jpeg_quality,
    }

    return encoded_image, summaries


def train(args):
    """Main training function."""

    # Create directories
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)
    os.makedirs(join(CHECKPOINTS_PATH, args.exp_name), exist_ok=True)
    os.makedirs(LOGS_PATH, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = StegaStampDataset(
        data_dir=TRAIN_PATH,
        secret_size=args.secret_size,
        image_size=(400, 400)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create models
    encoder = models.StegaStampEncoder(height=400, width=400).to(device)
    decoder = models.StegaStampDecoder(secret_size=args.secret_size, height=400, width=400).to(device)
    discriminator = models.Discriminator().to(device)

    # Create LPIPS loss model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()  # LPIPS should not be trained

    # Create optimizers
    g_vars = list(encoder.parameters()) + list(decoder.parameters())
    optimizer_G = torch.optim.Adam(g_vars, lr=args.lr)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=0.00001)

    # TensorBoard writer
    writer = SummaryWriter(join(LOGS_PATH, args.exp_name))

    # Load pretrained checkpoint if provided
    global_step = 0
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        global_step = checkpoint.get('global_step', 0)
        print(f"Loaded pretrained checkpoint from {args.pretrained}, global_step={global_step}")

    # Create edge falloff mask for L2 loss
    size = (400, 400)
    falloff_speed = 4
    falloff_im = np.ones(size, dtype=np.float32)
    for i in range(int(falloff_im.shape[0] / falloff_speed)):
        falloff_im[-i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2
        falloff_im[i, :] *= (np.cos(4 * np.pi * i / size[0] + np.pi) + 1) / 2
    for j in range(int(falloff_im.shape[1] / falloff_speed)):
        falloff_im[:, -j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
        falloff_im[:, j] *= (np.cos(4 * np.pi * j / size[0] + np.pi) + 1) / 2
    falloff_im = 1 - falloff_im
    falloff_im = torch.from_numpy(falloff_im).to(device)

    # YUV loss scales
    yuv_scales = torch.tensor([args.y_scale, args.u_scale, args.v_scale], device=device)

    # Training loop
    dataloader_iter = iter(dataloader)

    print(f"Starting training for {args.num_steps} steps...")

    while global_step < args.num_steps:
        # Get batch
        try:
            images, secrets = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            images, secrets = next(dataloader_iter)

        images = images.to(device)
        secrets = secrets.to(device)

        # Calculate loss scales with ramp-up
        no_im_loss = global_step < args.no_im_loss_steps
        l2_loss_scale = get_ramp_value(global_step, args.l2_loss_ramp, args.l2_loss_scale)
        lpips_loss_scale = get_ramp_value(global_step, args.lpips_loss_ramp, args.lpips_loss_scale)
        secret_loss_scale = get_ramp_value(global_step, args.secret_loss_ramp, args.secret_loss_scale)
        G_loss_scale = get_ramp_value(global_step, args.G_loss_ramp, args.G_loss_scale)

        # Edge gain with delay
        l2_edge_gain = 0.0
        if global_step > args.l2_edge_delay:
            l2_edge_gain = get_ramp_value(
                global_step - args.l2_edge_delay,
                args.l2_edge_ramp,
                args.l2_edge_gain
            )

        # Random perspective transformation
        rnd_tran = get_ramp_value(global_step, args.rnd_trans_ramp, args.rnd_trans)
        rnd_tran = np.random.uniform() * rnd_tran
        M = utils.get_rand_transform_matrix(400, int(np.floor(400 * rnd_tran)), args.batch_size)
        M = torch.from_numpy(M).float().to(device)  # [B, 2, 8]

        # ===== ENCODING =====
        # Apply warp to input image
        input_warped = apply_perspective_transform(images, M[:, 1, :], device=device)

        # Create mask for warped region
        mask_warped = apply_perspective_transform(
            torch.ones_like(images),
            M[:, 1, :],
            device=device
        )

        # Fill borders with original image where mask is 0
        input_warped = input_warped + (1 - mask_warped) * images

        # Encode secret into warped image
        residual_warped = encoder(secrets, input_warped)
        encoded_warped = residual_warped + input_warped

        # Unwarp the residual
        residual = apply_perspective_transform(residual_warped, M[:, 0, :], device=device)

        # Create encoded image based on border mode
        if args.borders == 'no_edge':
            encoded_image = images + residual
            D_input_real = images
            D_input_fake = encoded_image
        elif args.borders == 'black':
            encoded_image = apply_perspective_transform(encoded_warped, M[:, 0, :], device=device)
            D_input_real = input_warped
            D_input_fake = encoded_warped
        # Add other border modes as needed

        # ===== AUGMENTATION =====
        transformed_image, transform_summaries = transform_net(
            encoded_image, args, global_step, device=device
        )

        # ===== DECODING =====
        decoded_secret = decoder(transformed_image)

        # ===== DISCRIMINATOR =====
        D_output_real, _ = discriminator(D_input_real)
        D_output_fake, D_heatmap = discriminator(D_input_fake)

        # ===== LOSSES =====
        # 1. Secret loss (BCE)
        secret_loss = F.binary_cross_entropy_with_logits(decoded_secret, secrets)

        # 2. LPIPS perceptual loss
        # LPIPS expects images in [-1, 1] range
        lpips_loss = lpips_model(images * 2 - 1, encoded_image * 2 - 1).mean()

        # 3. L2 loss in YUV color space with edge emphasis
        encoded_image_yuv = rgb_to_yuv_pytorch(encoded_image)
        image_input_yuv = rgb_to_yuv_pytorch(images)
        im_diff = encoded_image_yuv - image_input_yuv

        # Add edge emphasis
        im_diff = im_diff + im_diff * falloff_im.unsqueeze(0).unsqueeze(0) * l2_edge_gain

        # Calculate per-channel YUV loss
        yuv_loss = torch.mean(im_diff ** 2, dim=[0, 2, 3])  # [3]
        image_loss = torch.dot(yuv_loss, yuv_scales)

        # 4. GAN loss
        D_loss = D_output_real - D_output_fake
        G_loss = D_output_fake

        # Total generator loss
        if no_im_loss:
            loss_G = secret_loss_scale * secret_loss
        else:
            loss_G = (l2_loss_scale * image_loss +
                     lpips_loss_scale * lpips_loss +
                     secret_loss_scale * secret_loss)
            if not args.no_gan:
                loss_G = loss_G + G_loss_scale * G_loss

        # ===== OPTIMIZATION =====
        if no_im_loss:
            # Train only on secret loss for first few steps
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()
        else:
            # Train generator
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Train discriminator (WGAN)
            if not args.no_gan:
                optimizer_D.zero_grad()
                D_loss_backward = -D_loss  # Minimize -D_loss = maximize D_loss
                D_loss_backward.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_value_(discriminator.parameters(), 0.25)

                optimizer_D.step()

                # Weight clipping for WGAN
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

        global_step += 1

        # ===== LOGGING =====
        if global_step % 100 == 0:
            # Calculate accuracy
            with torch.no_grad():
                bit_acc, str_acc = models.get_secret_acc(secrets, decoded_secret)

            # Log scalars
            writer.add_scalar('train/loss', loss_G.item(), global_step)
            writer.add_scalar('train/image_loss', image_loss.item(), global_step)
            writer.add_scalar('train/lpips_loss', lpips_loss.item(), global_step)
            writer.add_scalar('train/secret_loss', secret_loss.item(), global_step)
            writer.add_scalar('train/G_loss', G_loss.item(), global_step)
            writer.add_scalar('train/D_loss', D_loss.item(), global_step)
            writer.add_scalar('train/bit_acc', bit_acc.item(), global_step)
            writer.add_scalar('train/str_acc', str_acc.item(), global_step)

            # Log color losses
            writer.add_scalar('color_loss/Y_loss', yuv_loss[0].item(), global_step)
            writer.add_scalar('color_loss/U_loss', yuv_loss[1].item(), global_step)
            writer.add_scalar('color_loss/V_loss', yuv_loss[2].item(), global_step)

            # Log loss scales
            writer.add_scalar('loss_scales/l2_loss_scale', l2_loss_scale, global_step)
            writer.add_scalar('loss_scales/lpips_loss_scale', lpips_loss_scale, global_step)
            writer.add_scalar('loss_scales/secret_loss_scale', secret_loss_scale, global_step)
            writer.add_scalar('loss_scales/G_loss_scale', G_loss_scale, global_step)
            writer.add_scalar('loss_scales/l2_edge_gain', l2_edge_gain, global_step)

            # Log transform params
            for key, value in transform_summaries.items():
                writer.add_scalar(f'transformer/{key}', value, global_step)
            writer.add_scalar('transformer/rnd_tran', rnd_tran, global_step)

            print(f"Step {global_step}/{args.num_steps}: "
                  f"Loss={loss_G.item():.4f}, "
                  f"BitAcc={bit_acc.item():.3f}, "
                  f"StrAcc={str_acc.item():.3f}")

        # Log images
        if global_step % 1000 == 0:
            with torch.no_grad():
                writer.add_images('input/image', images[:4], global_step)
                writer.add_images('input/warped', input_warped[:4], global_step)
                writer.add_images('encoded/encoded_warped', encoded_warped[:4].clamp(0, 1), global_step)
                writer.add_images('encoded/residual', (residual_warped[:4] + 0.5).clamp(0, 1), global_step)
                writer.add_images('encoded/encoded_image', encoded_image[:4].clamp(0, 1), global_step)
                writer.add_images('transformed/transformed_image', transformed_image[:4].clamp(0, 1), global_step)

        # Save checkpoint
        if global_step % 10000 == 0:
            checkpoint_path = join(
                CHECKPOINTS_PATH,
                args.exp_name,
                f"{args.exp_name}_{global_step}.pth"
            )
            torch.save({
                'global_step': global_step,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'args': vars(args)
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final checkpoint
    final_path = join(CHECKPOINTS_PATH, args.exp_name, f"{args.exp_name}_final.pth")
    torch.save({
        'global_step': global_step,
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'discriminator': discriminator.state_dict(),
        'args': vars(args)
    }, final_path)
    print(f"Training complete! Final checkpoint saved to {final_path}")

    writer.close()


def rgb_to_yuv_pytorch(image):
    """
    Convert RGB to YUV color space.

    Args:
        image: [B, 3, H, W] - RGB image in [0, 1]

    Returns:
        yuv: [B, 3, H, W] - YUV image
    """
    # Conversion matrix (same as TensorFlow's rgb_to_yuv)
    matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.14714119, -0.28886916, 0.43601035],
        [0.61497538, -0.51496512, -0.10001026]
    ], device=image.device, dtype=image.dtype).T

    # Rearrange to [B, H, W, C] for matrix multiplication
    image_hwc = image.permute(0, 2, 3, 1)
    yuv = torch.matmul(image_hwc, matrix)

    # Back to [B, C, H, W]
    return yuv.permute(0, 3, 1, 2)


def main():
    parser = argparse.ArgumentParser(description='Train StegaStamp in PyTorch')
    parser.add_argument('exp_name', type=str, help='Experiment name')
    parser.add_argument('--secret_size', type=int, default=100, help='Number of bits in secret')
    parser.add_argument('--num_steps', type=int, default=140000, help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')

    # Loss scales and ramps
    parser.add_argument('--l2_loss_scale', type=float, default=1.5)
    parser.add_argument('--l2_loss_ramp', type=int, default=20000)
    parser.add_argument('--l2_edge_gain', type=float, default=10.0)
    parser.add_argument('--l2_edge_ramp', type=int, default=20000)
    parser.add_argument('--l2_edge_delay', type=int, default=60000)
    parser.add_argument('--lpips_loss_scale', type=float, default=1.0)
    parser.add_argument('--lpips_loss_ramp', type=int, default=20000)
    parser.add_argument('--secret_loss_scale', type=float, default=1.0)
    parser.add_argument('--secret_loss_ramp', type=int, default=1)
    parser.add_argument('--G_loss_scale', type=float, default=1.0)
    parser.add_argument('--G_loss_ramp', type=int, default=20000)

    # YUV scales
    parser.add_argument('--y_scale', type=float, default=1.0)
    parser.add_argument('--u_scale', type=float, default=1.0)
    parser.add_argument('--v_scale', type=float, default=1.0)

    # Augmentation parameters
    parser.add_argument('--borders', type=str, choices=['no_edge', 'black', 'random', 'randomrgb', 'image', 'white'],
                       default='black', help='Border handling mode')
    parser.add_argument('--rnd_trans', type=float, default=0.1, help='Random transformation magnitude')
    parser.add_argument('--rnd_bri', type=float, default=0.3, help='Random brightness')
    parser.add_argument('--rnd_noise', type=float, default=0.02, help='Random noise')
    parser.add_argument('--rnd_sat', type=float, default=1.0, help='Random saturation')
    parser.add_argument('--rnd_hue', type=float, default=0.1, help='Random hue')
    parser.add_argument('--contrast_low', type=float, default=0.5, help='Contrast lower bound')
    parser.add_argument('--contrast_high', type=float, default=1.5, help='Contrast upper bound')
    parser.add_argument('--jpeg_quality', type=float, default=25, help='Minimum JPEG quality')
    parser.add_argument('--no_jpeg', action='store_true', help='Disable JPEG augmentation')
    parser.add_argument('--no_gan', action='store_true', help='Disable GAN loss')

    # Augmentation ramps
    parser.add_argument('--rnd_trans_ramp', type=int, default=10000)
    parser.add_argument('--rnd_bri_ramp', type=int, default=1000)
    parser.add_argument('--rnd_sat_ramp', type=int, default=1000)
    parser.add_argument('--rnd_hue_ramp', type=int, default=1000)
    parser.add_argument('--rnd_noise_ramp', type=int, default=1000)
    parser.add_argument('--contrast_ramp', type=int, default=1000)
    parser.add_argument('--jpeg_quality_ramp', type=int, default=1000)

    # Other
    parser.add_argument('--no_im_loss_steps', type=int, default=500,
                       help='Train without image loss for first x steps')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained checkpoint')

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()

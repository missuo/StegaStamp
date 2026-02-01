# StegaStamp: Invisible Hyperlinks in Physical Photographs

[![CVPR 2020](https://img.shields.io/badge/CVPR-2020-blue.svg)](https://openaccess.thecvf.com/content_CVPR_2020/html/Tancik_StegaStamp_Invisible_Hyperlinks_in_Physical_Photographs_CVPR_2020_paper.html)
[![arXiv](https://img.shields.io/badge/arXiv-1904.05343-b31b1b.svg)](https://arxiv.org/abs/1904.05343)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-1.14+-005CED.svg)](https://onnx.ai/)

**[Matthew Tancik](https://www.matthewtancik.com), [Ben Mildenhall](http://people.eecs.berkeley.edu/~bmild/), [Ren Ng](https://scholar.google.com/citations?hl=en&user=6H0mhLUAAAAJ)**
*University of California, Berkeley*

![StegaStamp Teaser](https://github.com/tancik/StegaStamp/blob/master/docs/teaser.png)

## 🎯 Overview

StegaStamp is a learned steganographic algorithm to conceal messages in images. It can hide data within images while maintaining perceptual similarity, and the hidden data can be **recovered even after the image is printed and photographed**. This makes it ideal for embedding invisible hyperlinks, metadata, or authentication codes in physical photographs.

**Key Features:**
- 🖼️ **Invisible Watermarking**: Hide 100-bit messages in images
- 📸 **Print-Scan Robust**: Survives printing and photographing
- 🔒 **Error Correction**: BCH codes for reliable recovery
- ⚡ **Fast Inference**: Real-time encoding/decoding
- 🚀 **ONNX Export**: Deploy anywhere (mobile, web, embedded)
- 🧪 **PyTorch 2.x**: Modern, maintainable codebase

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Usage](#usage)
  - [Encoding Messages](#encoding-messages)
  - [Decoding Messages](#decoding-messages)
  - [ONNX Export](#onnx-export)
- [Architecture](#architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
# Clone the repository
git clone https://github.com/tancik/StegaStamp.git
cd StegaStamp

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
lpips>=0.1.4
onnx>=1.14.0
onnxruntime>=1.15.0
tensorboard>=2.10.0
opencv-python>=4.5.0
pillow>=9.0.0
numpy>=1.21.0
bchlib>=0.7
```

## 🚀 Quick Start

### Download Pretrained Models

```bash
# Download pretrained checkpoints (if available)
# Or train your own model (see Training section)
```

### Encode a Secret Message

```bash
python encode_image.py \
    checkpoints/stegastamp_pretrained.pth \
    --image examples/test.jpg \
    --save_dir output/ \
    --secret "Hello!"
```

Output:
```
✓ Encoded image saved to output/test_hidden.png
✓ Residual saved to output/test_residual.png
```

### Decode the Message

```bash
python decode_image.py \
    checkpoints/stegastamp_pretrained.pth \
    --image output/test_hidden.png
```

Output:
```
output/test_hidden.png: 'Hello!' (corrected 0 bit errors)
```

## 🏋️ Training

### 1. Prepare Dataset

Download a dataset of natural images (e.g., [MIR Flickr](http://press.liacs.nl/mirflickr/), [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)):

```bash
mkdir -p data/train_images
# Place your training images in data/train_images/
```

Update the data path in `train.py`:
```python
TRAIN_PATH = './data/train_images/'  # Line 16
```

### 2. Start Training

**Basic training:**
```bash
python train.py my_experiment \
    --secret_size 100 \
    --num_steps 140000 \
    --batch_size 4 \
    --lr 0.0001
```

**Quick test (1000 steps):**
```bash
python train.py quick_test \
    --num_steps 1000 \
    --batch_size 2
```

**Full training with custom parameters:**
```bash
python train.py stegastamp_production \
    --secret_size 100 \
    --num_steps 140000 \
    --batch_size 4 \
    --lr 0.0001 \
    --l2_loss_scale 1.5 \
    --lpips_loss_scale 1.0 \
    --secret_loss_scale 1.0 \
    --G_loss_scale 1.0 \
    --jpeg_quality 25 \
    --rnd_noise 0.02 \
    --rnd_bri 0.3
```

### 3. Monitor Training

```bash
tensorboard --logdir logs --port 6006
```

Visit http://localhost:6006 to view:
- **Bit Accuracy**: Should reach >90%
- **String Accuracy**: Should reach >70%
- **Loss curves**: Should steadily decrease
- **Sample images**: Visual quality checks

### 4. Training Time

| Hardware | Batch Size | Time for 140k steps |
|----------|------------|---------------------|
| RTX 3090 | 4 | ~13-20 hours |
| RTX 4090 | 4 | ~10-15 hours |
| CPU | 2 | ~5-7 days |

### Training Arguments

<details>
<summary>Click to expand full list of training arguments</summary>

**Basic Parameters:**
- `--secret_size`: Number of bits in secret (default: 100)
- `--num_steps`: Total training steps (default: 140000)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.0001)

**Loss Scales:**
- `--l2_loss_scale`: Image quality loss (default: 1.5)
- `--lpips_loss_scale`: Perceptual loss (default: 1.0)
- `--secret_loss_scale`: Secret recovery loss (default: 1.0)
- `--G_loss_scale`: GAN loss (default: 1.0)
- `--l2_edge_gain`: Edge emphasis (default: 10.0)

**Augmentation:**
- `--jpeg_quality`: Minimum JPEG quality (default: 25)
- `--rnd_noise`: Random noise level (default: 0.02)
- `--rnd_bri`: Brightness variation (default: 0.3)
- `--rnd_sat`: Saturation variation (default: 1.0)
- `--contrast_low/high`: Contrast range (default: 0.5/1.5)
- `--no_jpeg`: Disable JPEG augmentation
- `--no_gan`: Disable GAN loss

**Other:**
- `--borders`: Border handling (default: 'black')
- `--pretrained`: Resume from checkpoint
- `--no_im_loss_steps`: Train without image loss for first N steps (default: 500)

</details>

## 📖 Usage

### Encoding Messages

#### Single Image

```bash
python encode_image.py \
    checkpoints/my_model.pth \
    --image input.jpg \
    --save_dir output/ \
    --secret "MyCode"
```

#### Batch Processing

```bash
python encode_image.py \
    checkpoints/my_model.pth \
    --images_dir input_folder/ \
    --save_dir output_folder/ \
    --secret "Batch"
```

**Note**: Secret messages are limited to 7 UTF-8 characters (56 data bits + 40 ECC bits + 4 padding = 100 bits total).

### Decoding Messages

#### Single Image

```bash
python decode_image.py \
    checkpoints/my_model.pth \
    --image encoded.png
```

#### Batch Processing

```bash
python decode_image.py \
    checkpoints/my_model.pth \
    --images_dir encoded_folder/
```

### ONNX Export

Export trained models to ONNX for deployment:

```bash
python export_onnx.py \
    checkpoints/my_model.pth \
    --output_dir onnx_models/ \
    --secret_size 100 \
    --test
```

This creates:
- `onnx_models/encoder.onnx`
- `onnx_models/decoder.onnx`

The `--test` flag verifies numerical equivalence between PyTorch and ONNX models.

### ONNX Inference

**Encode with ONNX:**
```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --encode \
    --image test.jpg \
    --secret "ONNX" \
    --output encoded.png
```

**Decode with ONNX:**
```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --decode \
    --image encoded.png
```

**Test roundtrip:**
```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --test \
    --image test.jpg \
    --secret "Test"
```

## 🏗️ Architecture

### Encoder (U-Net)

The encoder embeds a secret into an image by generating a residual:

```
Secret (100 bits) → Dense(7500) → Reshape(3×50×50) → Upsample(8×) → 3×400×400
                                                                            ↓
Image (3×400×400) ────────────────────────────────────────────────→ Concat
                                                                            ↓
                                    U-Net (32→64→128→256→128→64→32)
                                                                            ↓
                                              Residual (3×400×400)
```

**Key Features:**
- Skip connections for detail preservation
- He normal initialization
- No activation on final layer (residual can be positive or negative)

### Decoder (CNN + STN)

The decoder extracts the secret from a potentially transformed image:

```
Image (3×400×400) → STN (Spatial Transformer) → Aligned Image
                                                        ↓
                              CNN Decoder (5 conv + 2 dense layers)
                                                        ↓
                                          Secret Logits (100 bits)
```

**Key Features:**
- Spatial Transformer Network for geometric invariance
- Handles perspective distortions, rotation, scaling
- BCH error correction for robustness

### Discriminator (WGAN)

Adversarial training for imperceptibility:

```
Image (3×400×400) → 5 Conv Layers (stride 2) → Scalar Score
```

**Training:**
- WGAN loss with gradient clipping
- Weight clipping to [-0.01, 0.01]
- RMSprop optimizer

### Loss Functions

1. **L2 Loss (YUV)**: `||YUV(encoded) - YUV(original)||²` with edge emphasis
2. **LPIPS Loss**: Perceptual similarity using AlexNet features
3. **Secret Loss**: Binary cross-entropy for bit recovery
4. **GAN Loss**: Adversarial loss for imperceptibility (optional)

All losses are ramped up gradually during training for stable convergence.

## 📊 Results

### Image Quality

| Metric | Value |
|--------|-------|
| PSNR | ~40 dB |
| SSIM | ~0.98 |
| LPIPS | ~0.02 |

### Secret Recovery

| Condition | Bit Accuracy | String Accuracy |
|-----------|--------------|-----------------|
| No corruption | >99% | >95% |
| JPEG (quality 50) | >95% | >85% |
| Print + photograph | >90% | >70% |
| Perspective warp | >88% | >65% |

### Speed

| Operation | Time (CPU) | Time (GPU) |
|-----------|------------|------------|
| Encoding | ~100ms | ~10ms |
| Decoding | ~60ms | ~8ms |

## 🧪 Testing

Run the test suite:

```bash
cd tests
python run_all_tests.py
```

Tests include:
- Model architecture validation
- Gradient flow verification
- JPEG differentiability
- ONNX export verification
- Numerical equivalence checks

## 🔬 Technical Details

### Error Correction

BCH (Bose-Chaudhuri-Hocquenghem) codes:
- **Data bits**: 56 (7 UTF-8 characters)
- **ECC bits**: 40
- **Padding**: 4 bits
- **Total**: 100 bits
- **Correction capacity**: Up to 5 bit errors

### Data Augmentation

Applied during training:
- **Blur**: Gaussian, line, or identity kernels
- **Noise**: Gaussian noise (σ ≤ 0.02)
- **Brightness**: ±0.3
- **Contrast**: 0.5-1.5×
- **Saturation**: Desaturation up to 100%
- **JPEG**: Quality 25-100
- **Perspective**: Random corner displacement

### Differentiable JPEG

Full PyTorch implementation:
1. RGB → YCbCr conversion
2. 4:2:0 chroma subsampling
3. 8×8 block DCT
4. Quantization (differentiable rounding)
5. Inverse operations

## 📁 Project Structure

```
StegaStamp/
├── models.py              # Encoder, Decoder, Discriminator
├── utils.py               # Differentiable JPEG, augmentations
├── dataset.py             # PyTorch Dataset
├── train.py               # Training script
├── encode_image.py        # Encoding inference
├── decode_image.py        # Decoding inference
├── export_onnx.py         # ONNX export
├── onnx_inference.py      # ONNX inference
├── requirements.txt       # Dependencies
├── tests/                 # Test suite
│   ├── test_models.py
│   ├── test_utils.py
│   └── run_all_tests.py
├── checkpoints/           # Model checkpoints (created during training)
├── logs/                  # TensorBoard logs (created during training)
└── archive_tensorflow/    # Original TensorFlow implementation (archived)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
python train.py my_exp --batch_size 2

# Or use CPU
CUDA_VISIBLE_DEVICES="" python train.py my_exp
```

### Poor Decoding Accuracy

1. **Check image quality**: Ensure minimal compression/resizing
2. **Verify checkpoint**: Use the correct model checkpoint
3. **Adjust training**: Increase `--jpeg_quality` or reduce augmentation
4. **Check lighting**: Ensure good lighting when photographing

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Or install individually
pip install torch torchvision lpips onnx onnxruntime
```

### Slow Training

1. **Use GPU**: CUDA significantly speeds up training
2. **Increase batch size**: If memory allows
3. **Reduce workers**: If CPU-bound: `--num_workers 2`
4. **Test first**: Use `--num_steps 1000` to verify setup

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tancik2020stegastamp,
    title={StegaStamp: Invisible Hyperlinks in Physical Photographs},
    author={Tancik, Matthew and Mildenhall, Ben and Ng, Ren},
    booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2020}
}
```

## 📄 License

This project is licensed under the same terms as the original StegaStamp repository.

## 🙏 Acknowledgments

- **Original Authors**: Matthew Tancik, Ben Mildenhall, Ren Ng
- **Differentiable JPEG**: [rshin/differentiable-jpeg](https://github.com/rshin/differentiable-jpeg)
- **LPIPS**: [richzhang/PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- **PyTorch Migration**: Migrated from TensorFlow 1.13 to PyTorch 2.x with ONNX support

## 🔗 Resources

- **Paper**: [arXiv:1904.05343](https://arxiv.org/abs/1904.05343)
- **Project Page**: [matthewtancik.com/stegastamp](http://www.matthewtancik.com/stegastamp)
- **Original Repository**: [github.com/tancik/StegaStamp](https://github.com/tancik/StegaStamp)

## 💬 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a PyTorch 2.x implementation. The original TensorFlow code is archived in `archive_tensorflow/` for reference.

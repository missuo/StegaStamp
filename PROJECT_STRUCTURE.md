# StegaStamp 项目结构

## 📁 目录结构

```
StegaStamp/
├── 📄 核心文件
│   ├── models.py                 # PyTorch 模型定义（Encoder, Decoder, Discriminator）
│   ├── utils.py                  # 工具函数（可微分 JPEG，数据增强）
│   ├── dataset.py                # PyTorch Dataset 类
│   ├── train.py                  # 训练脚本
│   ├── encode_image.py           # 图片编码脚本
│   ├── decode_image.py           # 图片解码脚本
│   ├── export_onnx.py            # ONNX 导出脚本
│   └── onnx_inference.py         # ONNX 推理脚本
│
├── 📚 文档
│   ├── README.md                 # 主文档（英文）
│   ├── TRAINING_GUIDE_CN.md      # 训练指南（中文）
│   ├── README_PYTORCH.md         # PyTorch 迁移详细文档
│   ├── MIGRATION_SUMMARY.md      # 迁移总结
│   └── PROJECT_STRUCTURE.md      # 本文件
│
├── 🧪 测试
│   └── tests/
│       ├── test_models.py        # 模型测试
│       ├── test_utils.py         # 工具函数测试
│       └── run_all_tests.py      # 测试运行器
│
├── 📦 配置
│   └── requirements.txt          # Python 依赖
│
├── 📂 训练输出（训练时自动创建）
│   ├── checkpoints/              # 模型检查点
│   │   └── {experiment_name}/
│   │       ├── {name}_10000.pth
│   │       ├── {name}_20000.pth
│   │       └── {name}_final.pth
│   ├── logs/                     # TensorBoard 日志
│   │   └── {experiment_name}/
│   └── onnx_models/              # 导出的 ONNX 模型
│       ├── encoder.onnx
│       └── decoder.onnx
│
├── 📂 数据（需要手动创建）
│   └── data/
│       └── {dataset_name}/       # 训练图片
│
└── 📦 存档
    └── archive_tensorflow/        # 原始 TensorFlow 实现（已归档）
        ├── models.py
        ├── utils.py
        ├── train.py
        ├── encode_image.py
        ├── decode_image.py
        ├── detector.py
        ├── requirements.txt
        └── README.md
```

## 📄 文件说明

### 核心模块

#### `models.py` (320 行)
PyTorch 模型定义

**类：**
- `StegaStampEncoder`: U-Net 编码器，生成残差
- `StegaStampDecoder`: CNN + STN 解码器，提取秘密
- `Discriminator`: WGAN 判别器，用于对抗训练
- `get_secret_acc`: 计算比特和字符串准确率

**关键方法：**
- `forward()`: 前向传播
- `_initialize_weights()`: 权重初始化

#### `utils.py` (500+ 行)
工具函数和数据增强

**可微分 JPEG：**
- `rgb_to_ycbcr_jpeg()`: RGB → YCbCr 转换
- `ycbcr_to_rgb_jpeg()`: YCbCr → RGB 转换
- `downsampling_420()`: 色度下采样
- `upsampling_420()`: 色度上采样
- `dct_8x8()`: 8×8 DCT 变换
- `idct_8x8()`: 8×8 IDCT 变换
- `y_quantize()`, `c_quantize()`: 量化
- `y_dequantize()`, `c_dequantize()`: 反量化
- `diff_round()`, `round_only_at_0()`: 可微分舍入
- `jpeg_compress_decompress()`: 完整 JPEG 管道

**数据增强：**
- `random_blur_kernel()`: 随机模糊核
- `get_rand_transform_matrix()`: 透视变换矩阵
- `get_rnd_brightness_torch()`: 亮度/色调调整

#### `dataset.py` (90 行)
PyTorch 数据集类

**类：**
- `StegaStampDataset`: 加载图片和生成秘密
  - `__init__()`: 初始化数据集路径
  - `__len__()`: 返回数据集大小
  - `__getitem__()`: 返回 (image, secret) 对

#### `train.py` (600+ 行)
完整训练流程

**主要函数：**
- `train()`: 主训练循环
  - 创建模型和优化器
  - 损失计算和反向传播
  - TensorBoard 日志记录
  - 检查点保存
- `transform_net()`: 数据增强管道
- `apply_perspective_transform()`: 应用透视变换
- `rgb_to_yuv_pytorch()`: RGB → YUV 转换
- `get_ramp_value()`: 损失权重递增

**损失组件：**
1. L2 损失（YUV 色彩空间 + 边缘强调）
2. LPIPS 感知损失
3. 秘密恢复 BCE 损失
4. GAN 对抗损失

### 推理脚本

#### `encode_image.py` (120 行)
将秘密编码到图片中

**功能：**
- 加载训练好的编码器
- 使用 BCH 编码处理秘密消息
- 生成隐写图片和残差
- 支持单张或批量处理

**命令行参数：**
- `checkpoint`: 模型检查点路径
- `--image`: 单张图片路径
- `--images_dir`: 图片目录
- `--save_dir`: 输出目录
- `--secret`: 秘密消息（最多 7 字符）

#### `decode_image.py` (90 行)
从图片中解码秘密

**功能：**
- 加载训练好的解码器
- 提取秘密比特
- BCH 纠错解码
- 输出 UTF-8 字符串

**命令行参数：**
- `checkpoint`: 模型检查点路径
- `--image`: 单张图片路径
- `--images_dir`: 图片目录
- `--secret_size`: 秘密大小（默认 100）

### ONNX 支持

#### `export_onnx.py` (150 行)
导出 PyTorch 模型到 ONNX

**功能：**
- 导出编码器和解码器
- 动态批量大小支持
- ONNX 模型验证
- 数值等价性测试

**主要函数：**
- `export_encoder()`: 导出编码器
- `export_decoder()`: 导出解码器
- `test_onnx_models()`: 测试 ONNX vs PyTorch

#### `onnx_inference.py` (120 行)
使用 ONNX 模型进行推理

**功能：**
- ONNX 编码/解码
- 往返测试（encode → decode）
- 跨平台部署演示

**模式：**
- `--encode`: 编码模式
- `--decode`: 解码模式
- `--test`: 往返测试模式

### 测试套件

#### `tests/test_models.py` (250 行)
模型测试

**测试项：**
- 输出形状验证
- 梯度流测试
- STN 功能测试
- 比特准确率计算
- 设备迁移测试
- 确定性测试

#### `tests/test_utils.py` (280 行)
工具函数测试

**测试项：**
- JPEG 形状保持
- JPEG 可微分性
- JPEG 质量因子效果
- DCT/IDCT 往返精度
- 色度子采样
- 模糊核生成
- 亮度调整
- RGB ↔ YUV 转换

#### `tests/run_all_tests.py` (50 行)
测试运行器

**功能：**
- 运行所有测试
- 汇总测试结果
- 返回退出代码

## 🔄 数据流

### 训练流程

```
1. 数据加载
   Dataset → DataLoader → (image, secret) batch

2. 编码
   secret + image → Encoder → residual
   image + residual → encoded_image

3. 透视变换
   encoded_image → warp → warped_image

4. 数据增强
   warped_image → blur, noise, JPEG, etc. → transformed_image

5. 解码
   transformed_image → Decoder → secret_logits

6. 损失计算
   - L2 loss: ||YUV(encoded) - YUV(original)||²
   - LPIPS: perceptual_loss(encoded, original)
   - Secret: BCE(secret_logits, secret)
   - GAN: discriminator(encoded) vs discriminator(real)

7. 优化
   - Generator (Encoder + Decoder): Adam
   - Discriminator: RMSprop

8. 保存
   - Checkpoint every 10k steps
   - TensorBoard logging
```

### 推理流程（编码）

```
1. 加载图片
   image_path → PIL.Image → numpy → torch.Tensor [1, 3, 400, 400]

2. 准备秘密
   "Hello" → UTF-8 → BCH encode → 100 bits → torch.Tensor [1, 100]

3. 编码
   Encoder(secret, image) → residual [1, 3, 400, 400]

4. 生成隐写图
   encoded = image + residual → clamp [0, 1]

5. 保存
   encoded → numpy → PIL.Image → save as PNG
```

### 推理流程（解码）

```
1. 加载图片
   image_path → PIL.Image → numpy → torch.Tensor [1, 3, 400, 400]

2. 解码
   Decoder(image) → secret_logits [1, 100]

3. 二值化
   sigmoid(logits) → round → binary secret

4. BCH 解码
   binary (96 bits) → BCH decode → 56 data bits → UTF-8

5. 输出
   "Hello" + error correction info
```

## 🛠️ 开发工作流

### 1. 初始设置

```bash
# 克隆仓库
git clone https://github.com/your-repo/StegaStamp.git
cd StegaStamp

# 安装依赖
pip install -r requirements.txt

# 运行测试
cd tests && python run_all_tests.py
```

### 2. 准备数据

```bash
# 创建数据目录
mkdir -p data/train_images

# 下载或复制训练图片
# 修改 train.py 中的 TRAIN_PATH
```

### 3. 训练模型

```bash
# 快速测试
python train.py test --num_steps 1000

# 完整训练
python train.py production --num_steps 140000

# 监控训练
tensorboard --logdir logs
```

### 4. 测试模型

```bash
# 编码测试
python encode_image.py checkpoints/production/production_final.pth \
    --image test.jpg --save_dir output --secret "Test"

# 解码测试
python decode_image.py checkpoints/production/production_final.pth \
    --image output/test_hidden.png
```

### 5. 导出 ONNX

```bash
# 导出
python export_onnx.py checkpoints/production/production_final.pth \
    --output_dir onnx_models --test

# 测试 ONNX
python onnx_inference.py onnx_models/encoder.onnx onnx_models/decoder.onnx \
    --test --image test.jpg --secret "ONNX"
```

## 📊 检查点格式

### PyTorch 检查点 (.pth)

```python
{
    'global_step': int,              # 训练步数
    'encoder': OrderedDict,          # 编码器权重
    'decoder': OrderedDict,          # 解码器权重
    'discriminator': OrderedDict,    # 判别器权重
    'optimizer_G': dict,             # 生成器优化器状态（可选）
    'optimizer_D': dict,             # 判别器优化器状态（可选）
    'args': dict                     # 训练参数（可选）
}
```

### ONNX 模型

- **encoder.onnx**: 独立的编码器模型
  - Input: `secret` [B, 100], `image` [B, 3, 400, 400]
  - Output: `residual` [B, 3, 400, 400]

- **decoder.onnx**: 独立的解码器模型
  - Input: `image` [B, 3, 400, 400]
  - Output: `secret_logits` [B, 100]

## 🔍 代码导航

### 查找功能位置

| 功能 | 文件 | 函数/类 |
|------|------|---------|
| U-Net 编码器 | `models.py` | `StegaStampEncoder` |
| STN 解码器 | `models.py` | `StegaStampDecoder` |
| 判别器 | `models.py` | `Discriminator` |
| 可微分 JPEG | `utils.py` | `jpeg_compress_decompress()` |
| DCT 变换 | `utils.py` | `dct_8x8()`, `idct_8x8()` |
| 数据增强 | `train.py` | `transform_net()` |
| 训练循环 | `train.py` | `train()` |
| 损失计算 | `train.py` | `train()` 函数内 |
| 数据加载 | `dataset.py` | `StegaStampDataset` |
| 编码推理 | `encode_image.py` | `main()` |
| 解码推理 | `decode_image.py` | `main()` |
| ONNX 导出 | `export_onnx.py` | `export_encoder()`, `export_decoder()` |

## 📝 常用命令

```bash
# 训练
python train.py exp_name --num_steps 140000

# 编码
python encode_image.py checkpoint.pth --image in.jpg --save_dir out --secret "Hi"

# 解码
python decode_image.py checkpoint.pth --image encoded.png

# 导出 ONNX
python export_onnx.py checkpoint.pth --output_dir onnx --test

# ONNX 推理
python onnx_inference.py encoder.onnx decoder.onnx --test --image test.jpg

# 测试
cd tests && python run_all_tests.py

# TensorBoard
tensorboard --logdir logs
```

## 🎯 关键路径

### 添加新的数据增强

1. 在 `utils.py` 中实现增强函数
2. 在 `train.py` 的 `transform_net()` 中调用
3. 在 `tests/test_utils.py` 中添加测试

### 修改模型架构

1. 在 `models.py` 中修改模型类
2. 在 `tests/test_models.py` 中更新测试
3. 重新训练模型

### 添加新的损失函数

1. 在 `train.py` 的 `train()` 函数中添加损失计算
2. 添加命令行参数控制损失权重
3. 在 TensorBoard 中记录新损失

---

**最后更新**: 2026-02-01
**版本**: PyTorch 2.x

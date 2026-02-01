# StegaStamp 训练指南（中文）

本指南将详细介绍如何训练 StegaStamp 模型并导出 ONNX 格式。

## 📋 目录

- [环境准备](#环境准备)
- [数据集准备](#数据集准备)
- [开始训练](#开始训练)
- [监控训练](#监控训练)
- [测试模型](#测试模型)
- [导出ONNX](#导出onnx)
- [常见问题](#常见问题)

## 🔧 环境准备

### 1. 系统要求

- **操作系统**: Linux, macOS, or Windows
- **Python**: 3.8 或更高版本
- **GPU**: NVIDIA GPU（推荐，可选）
- **内存**: 至少 8GB RAM
- **硬盘**: 至少 10GB 可用空间

### 2. 安装依赖

```bash
cd /Users/vincent/Projects/StegaStamp

# 安装所有依赖
pip install -r requirements.txt
```

如果遇到权限问题，使用：
```bash
pip install --user -r requirements.txt
```

### 3. 验证安装

```bash
# 运行测试
cd tests
python run_all_tests.py

# 应该看到：
# ✓ ALL TESTS PASSED
```

如果测试通过，说明环境配置成功！

## 📁 数据集准备

### 方法1：使用公开数据集

推荐使用以下数据集：

#### MIR Flickr（推荐）
```bash
# 1. 访问 http://press.liacs.nl/mirflickr/
# 2. 下载图片集（约 1GB）
# 3. 解压到 data/mirflickr/images1/images/

mkdir -p data/mirflickr/images1/images
# 将下载的图片放入此目录
```

#### DIV2K
```bash
# 1. 访问 https://data.vision.ee.ethz.ch/cvl/DIV2K/
# 2. 下载训练集
# 3. 解压到 data/DIV2K/

mkdir -p data/DIV2K
# 将下载的图片放入此目录
```

#### COCO
```bash
# 1. 访问 https://cocodataset.org/
# 2. 下载 train2017
# 3. 解压到 data/coco/

mkdir -p data/coco
# 将下载的图片放入此目录
```

### 方法2：使用自己的图片

```bash
# 创建数据目录
mkdir -p data/my_images

# 将您的图片复制到此目录
# 支持的格式：.jpg, .jpeg, .png, .bmp

# 建议：
# - 至少 1000 张图片
# - 图片尺寸 >= 400x400
# - 多样化的内容（风景、人物、物体等）
```

### 方法3：快速测试（小数据集）

如果只是想测试训练流程：

```bash
# 创建测试目录
mkdir -p data/test_images

# 从网上下载一些测试图片
# 或复制一些现有图片（10-20张即可）
```

### 配置数据路径

编辑 `train.py` 文件，修改第 16 行：

```python
# 将这一行：
TRAIN_PATH = './data/mirflickr/images1/images/'

# 改为您的数据路径：
TRAIN_PATH = './data/my_images/'  # 或其他路径
```

## 🚀 开始训练

### 快速开始（测试配置）

先用少量步数测试，确保一切正常：

```bash
python train.py test_run \
    --secret_size 100 \
    --num_steps 1000 \
    --batch_size 2 \
    --lr 0.0001
```

**预期输出：**
```
Using device: cpu (or cuda)
Dataset initialized with 1234 images from ./data/my_images
Starting training for 1000 steps...

Step 100/1000: Loss=0.8523, BitAcc=0.523, StrAcc=0.000
Step 200/1000: Loss=0.7234, BitAcc=0.612, StrAcc=0.000
...
✓ Saved checkpoint to checkpoints/test_run/test_run_final.pth
```

### 标准训练（推荐配置）

如果测试成功，开始正式训练：

```bash
python train.py stegastamp_v1 \
    --secret_size 100 \
    --num_steps 140000 \
    --batch_size 4 \
    --lr 0.0001 \
    --l2_loss_scale 1.5 \
    --lpips_loss_scale 1.0 \
    --secret_loss_scale 1.0 \
    --G_loss_scale 1.0
```

### 根据硬件调整

#### 如果显存不足：
```bash
python train.py stegastamp_v1 \
    --batch_size 2 \  # 减小 batch size
    --num_steps 140000
```

#### 如果只有 CPU：
```bash
python train.py stegastamp_v1 \
    --batch_size 1 \  # CPU 使用更小的 batch
    --num_steps 50000  # 减少训练步数
```

#### 如果有强大的 GPU：
```bash
python train.py stegastamp_v1 \
    --batch_size 8 \  # 增大 batch size
    --num_steps 140000
```

### 从检查点继续训练

如果训练中断，可以从检查点继续：

```bash
python train.py stegastamp_v1 \
    --pretrained checkpoints/stegastamp_v1/stegastamp_v1_50000.pth \
    --num_steps 140000
```

## 📊 监控训练

### 1. 使用 TensorBoard

在新的终端窗口中：

```bash
cd /Users/vincent/Projects/StegaStamp

# 启动 TensorBoard
tensorboard --logdir logs --port 6006
```

然后在浏览器打开：**http://localhost:6006**

### 2. 重要指标

#### 训练损失（Train Loss）
- **train/loss**: 总损失，应该逐渐下降
- **train/image_loss**: 图像质量损失，应该保持较低
- **train/secret_loss**: 秘密恢复损失，应该下降
- **train/lpips_loss**: 感知损失，应该保持较低

#### 准确率（Accuracy）
- **train/bit_acc**: 比特准确率
  - 开始时：~50%（随机猜测）
  - 10k 步：~70-80%
  - 50k 步：~85-90%
  - 100k 步：**>90%** ✓ 目标

- **train/str_acc**: 字符串准确率
  - 开始时：~0%
  - 50k 步：~30-50%
  - 100k 步：**>70%** ✓ 目标

#### 颜色损失（Color Loss）
- **color_loss/Y_loss**: 亮度损失
- **color_loss/U_loss**: 色度 U 损失
- **color_loss/V_loss**: 色度 V 损失

### 3. 查看生成的图片

在 TensorBoard 的 "IMAGES" 标签页可以看到：
- **input/image**: 原始输入图片
- **encoded/encoded_image**: 编码后的图片（应该看起来与原图相似）
- **encoded/residual**: 添加的残差（应该几乎不可见）
- **transformed/transformed_image**: 经过增强后的图片

### 4. 命令行输出

训练过程中会看到类似输出：

```
Step 100/140000: Loss=0.8234, BitAcc=0.543, StrAcc=0.000
Step 200/140000: Loss=0.7543, BitAcc=0.612, StrAcc=0.000
Step 1000/140000: Loss=0.6234, BitAcc=0.723, StrAcc=0.050
...
Step 10000/140000: Loss=0.4123, BitAcc=0.856, StrAcc=0.125
Saved checkpoint to checkpoints/stegastamp_v1/stegastamp_v1_10000.pth
...
Step 100000/140000: Loss=0.2845, BitAcc=0.923, StrAcc=0.750
```

## 🧪 测试模型

### 1. 准备测试图片

```bash
# 创建测试目录
mkdir -p test_images

# 放入一些测试图片
# 或使用训练集中的图片
```

### 2. 测试编码

```bash
python encode_image.py \
    checkpoints/stegastamp_v1/stegastamp_v1_100000.pth \
    --image test_images/photo.jpg \
    --save_dir output_test \
    --secret "Hello!"
```

**预期输出：**
```
Using device: cpu
Loading checkpoint...
Encoder loaded successfully
Secret message: 'Hello!'

Processing 1/1: test_images/photo.jpg
  Saved: output_test/photo_hidden.png
  Saved: output_test/photo_residual.png

✓ Processed 1 images
```

### 3. 测试解码

```bash
python decode_image.py \
    checkpoints/stegastamp_v1/stegastamp_v1_100000.pth \
    --image output_test/photo_hidden.png
```

**预期输出：**
```
Using device: cpu
Loading checkpoint...
Decoder loaded successfully

output_test/photo_hidden.png: 'Hello!' (corrected 0 bit errors)

✓ Processed 1 images
```

✅ 如果能正确解码出 "Hello!"，说明模型训练成功！

### 4. 批量测试

```bash
# 编码多张图片
python encode_image.py \
    checkpoints/stegastamp_v1/stegastamp_v1_final.pth \
    --images_dir test_images/ \
    --save_dir output_batch \
    --secret "Batch"

# 解码多张图片
python decode_image.py \
    checkpoints/stegastamp_v1/stegastamp_v1_final.pth \
    --images_dir output_batch/
```

## 📦 导出ONNX

### 1. 导出模型

```bash
python export_onnx.py \
    checkpoints/stegastamp_v1/stegastamp_v1_final.pth \
    --output_dir onnx_models \
    --secret_size 100 \
    --opset_version 14 \
    --test
```

**预期输出：**
```
Using device: cpu

Loading checkpoint...
✓ Models loaded successfully

=== Exporting Encoder to ONNX ===
✓ Encoder exported to onnx_models/encoder.onnx
✓ ONNX model verification passed

=== Exporting Decoder to ONNX ===
✓ Decoder exported to onnx_models/decoder.onnx
✓ ONNX model verification passed

=== Testing ONNX Models ===

Testing encoder...
  Max difference: 0.000123
  Mean difference: 0.000012
  ✓ Encoder outputs match (rtol=1e-3)

Testing decoder...
  Max difference: 0.000098
  Mean difference: 0.000008
  ✓ Decoder outputs match (rtol=1e-3)

✓ ONNX model testing complete

==================================================
ONNX Export Complete!
==================================================
Encoder: onnx_models/encoder.onnx
Decoder: onnx_models/decoder.onnx
```

### 2. 测试 ONNX 模型

#### 完整测试（编码→解码）

```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --test \
    --image test_images/photo.jpg \
    --secret "ONNX" \
    --output onnx_test.png
```

**预期输出：**
```
Loading ONNX models...
✓ Models loaded

Testing roundtrip encode/decode with secret 'ONNX'
==================================================

1. Encoding...
✓ Encoded image saved to onnx_test.png

2. Decoding...
✓ Decoded: 'ONNX' (corrected 0 bit errors)

==================================================
ROUNDTRIP TEST RESULTS
==================================================
Original secret: 'ONNX'
Decoded secret:  'ONNX'

✓ SUCCESS: Roundtrip encode/decode successful!
```

#### 仅编码

```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --encode \
    --image test_images/photo.jpg \
    --secret "Test" \
    --output encoded.png
```

#### 仅解码

```bash
python onnx_inference.py \
    onnx_models/encoder.onnx \
    onnx_models/decoder.onnx \
    --decode \
    --image encoded.png
```

## ❓ 常见问题

### 1. CUDA out of memory（显存不足）

**问题：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```bash
# 方案1：减小 batch size
python train.py my_exp --batch_size 2

# 方案2：使用 CPU
CUDA_VISIBLE_DEVICES="" python train.py my_exp

# 方案3：减小图片尺寸（需要修改代码）
```

### 2. 找不到训练图片

**问题：**
```
ValueError: No images found in ./data/...
```

**解决方案：**
```bash
# 检查路径是否正确
ls -la data/mirflickr/images1/images/

# 确保有图片文件
# 修改 train.py 中的 TRAIN_PATH
```

### 3. 训练速度太慢

**问题：** 每步需要很长时间

**解决方案：**
```bash
# 1. 使用 GPU（如果可用）
nvidia-smi  # 检查 GPU

# 2. 减小 batch size
python train.py my_exp --batch_size 2

# 3. 使用更少的训练步数进行测试
python train.py my_exp --num_steps 10000
```

### 4. 解码准确率低

**问题：** BitAcc < 70%

**可能原因和解决方案：**

1. **训练不充分**
   ```bash
   # 继续训练更多步数
   python train.py my_exp --num_steps 200000
   ```

2. **增强太强**
   ```bash
   # 减少 JPEG 压缩
   python train.py my_exp --jpeg_quality 50

   # 减少噪声
   python train.py my_exp --rnd_noise 0.01
   ```

3. **数据集太小**
   ```bash
   # 使用更大的数据集（至少 1000 张图片）
   ```

### 5. TensorBoard 无法访问

**问题：** 浏览器无法打开 http://localhost:6006

**解决方案：**
```bash
# 检查 TensorBoard 是否运行
ps aux | grep tensorboard

# 尝试不同端口
tensorboard --logdir logs --port 6007

# 检查防火墙设置
```

### 6. ONNX 导出失败

**问题：**
```
RuntimeError: ONNX export failed
```

**解决方案：**
```bash
# 确保安装了正确版本
pip install --upgrade onnx onnxruntime

# 检查版本
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxruntime; print(onnxruntime.__version__)"

# 重新导出
python export_onnx.py checkpoints/my_model.pth --output_dir onnx_models
```

### 7. 导入错误

**问题：**
```
ModuleNotFoundError: No module named 'lpips'
```

**解决方案：**
```bash
# 重新安装依赖
pip install -r requirements.txt

# 或单独安装
pip install lpips
```

## 📈 训练时间参考

根据不同硬件配置的预估时间：

| 硬件配置 | Batch Size | 140k 步预计时间 |
|----------|------------|-----------------|
| RTX 4090 | 8 | 10-12 小时 |
| RTX 3090 | 4 | 15-18 小时 |
| RTX 3070 | 4 | 20-24 小时 |
| RTX 2080 Ti | 4 | 24-30 小时 |
| CPU (12核) | 2 | 5-7 天 |
| CPU (6核) | 1 | 10-14 天 |

**建议：**
- 使用 GPU 训练可以节省大量时间
- 可以先用少量步数（10k-20k）快速验证配置
- 然后运行完整的 140k 步训练

## 🎯 训练成功标准

训练成功的标志：

✅ **Bit Accuracy > 90%**
✅ **String Accuracy > 70%**
✅ **编码图片视觉上与原图相似**
✅ **能正确解码测试图片**
✅ **ONNX 导出成功且数值等价**

达到这些标准后，模型就可以用于生产环境了！

## 📚 进阶技巧

### 1. 调整超参数

不同的应用场景可能需要不同的参数：

**高隐蔽性（更难察觉）：**
```bash
python train.py high_stealth \
    --l2_loss_scale 2.0 \      # 增加图像质量损失
    --lpips_loss_scale 1.5 \   # 增加感知损失
    --secret_loss_scale 0.8    # 稍微降低秘密损失
```

**高鲁棒性（更能抵抗破坏）：**
```bash
python train.py high_robust \
    --secret_loss_scale 1.5 \  # 增加秘密损失
    --jpeg_quality 15 \        # 更强的 JPEG 压缩
    --rnd_noise 0.03          # 更多噪声
```

### 2. 分阶段训练

```bash
# 阶段1：仅训练秘密恢复（5k步）
python train.py stage1 \
    --num_steps 5000 \
    --no_im_loss_steps 5000

# 阶段2：添加图像质量约束（继续训练）
python train.py stage2 \
    --pretrained checkpoints/stage1/stage1_final.pth \
    --num_steps 100000

# 阶段3：精细调整（可选）
python train.py stage3 \
    --pretrained checkpoints/stage2/stage2_final.pth \
    --num_steps 140000 \
    --lr 0.00005  # 降低学习率
```

### 3. 使用预训练模型

如果有类似任务的预训练模型：

```bash
python train.py finetune \
    --pretrained path/to/pretrained.pth \
    --num_steps 50000 \
    --lr 0.00005  # 使用较小的学习率微调
```

## 🎬 完整示例脚本

创建自动化脚本 `auto_train.sh`：

```bash
#!/bin/bash

# 配置
EXP_NAME="stegastamp_$(date +%Y%m%d_%H%M%S)"
DATA_PATH="./data/mirflickr/images1/images/"
NUM_STEPS=140000

echo "======================================"
echo "StegaStamp 自动训练脚本"
echo "======================================"
echo "实验名称: $EXP_NAME"
echo "数据路径: $DATA_PATH"
echo "训练步数: $NUM_STEPS"
echo ""

# 检查数据集
if [ ! -d "$DATA_PATH" ]; then
    echo "错误：数据集路径不存在！"
    echo "请确保数据集位于: $DATA_PATH"
    exit 1
fi

# 修改 train.py 中的数据路径
sed -i.bak "s|TRAIN_PATH = .*|TRAIN_PATH = '$DATA_PATH'|" train.py

# 开始训练
echo "开始训练..."
python train.py $EXP_NAME \
    --secret_size 100 \
    --num_steps $NUM_STEPS \
    --batch_size 4 \
    --lr 0.0001 \
    --l2_loss_scale 1.5 \
    --lpips_loss_scale 1.0 \
    --secret_loss_scale 1.0 \
    --G_loss_scale 1.0

# 检查训练是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 训练完成！"

    # 导出 ONNX
    echo "导出 ONNX 模型..."
    python export_onnx.py \
        checkpoints/$EXP_NAME/${EXP_NAME}_final.pth \
        --output_dir onnx_models_$EXP_NAME \
        --test

    if [ $? -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "✓ 全部完成！"
        echo "======================================"
        echo "PyTorch 模型: checkpoints/$EXP_NAME/"
        echo "ONNX 模型: onnx_models_$EXP_NAME/"
        echo ""
        echo "测试命令："
        echo "  python encode_image.py checkpoints/$EXP_NAME/${EXP_NAME}_final.pth --image test.jpg --save_dir output --secret 'Test'"
        echo "  python decode_image.py checkpoints/$EXP_NAME/${EXP_NAME}_final.pth --image output/test_hidden.png"
    fi
else
    echo "✗ 训练失败！请检查错误信息。"
    exit 1
fi
```

使用方法：
```bash
chmod +x auto_train.sh
./auto_train.sh
```

## 📞 获取帮助

如果遇到问题：

1. **查看文档**: 仔细阅读本指南和主 README.md
2. **查看日志**: 检查 TensorBoard 和终端输出
3. **运行测试**: `cd tests && python run_all_tests.py`
4. **GitHub Issues**: 在项目仓库提交 issue

祝训练顺利！🚀

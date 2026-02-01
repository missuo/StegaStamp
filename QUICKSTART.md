# StegaStamp 快速开始指南

**5分钟上手 StegaStamp**

## 📦 安装（1分钟）

```bash
# 克隆项目
git clone https://github.com/tancik/StegaStamp.git
cd StegaStamp

# 安装依赖
pip install -r requirements.txt
```

## ✅ 验证安装（30秒）

```bash
cd tests
python run_all_tests.py
# 应该看到: ✓ ALL TESTS PASSED
```

## 🎯 快速测试（2分钟）

### 使用示例图片测试

```bash
# 下载示例图片（或使用您自己的）
mkdir -p examples
# 将一张图片放入 examples/test.jpg

# 创建小数据集用于测试
mkdir -p data/test_images
cp examples/test.jpg data/test_images/
```

### 快速训练（1000步测试）

```bash
# 修改数据路径（编辑 train.py 第16行）
# TRAIN_PATH = './data/test_images/'

# 快速训练测试
python train.py quick_test --num_steps 1000 --batch_size 2
```

## 🚀 完整工作流（10分钟）

### 1. 准备数据（2分钟）

```bash
# 下载 MIR Flickr 数据集或使用自己的图片
mkdir -p data/train_images
# 放入至少 100 张图片
```

### 2. 开始训练（5-7小时在GPU，或用少量步数测试）

```bash
# 完整训练（推荐）
python train.py stegastamp_v1 --num_steps 140000

# 或快速验证（10分钟）
python train.py quick_verify --num_steps 5000
```

### 3. 监控训练（在新终端）

```bash
tensorboard --logdir logs --port 6006
# 访问 http://localhost:6006
```

### 4. 测试编码/解码（1分钟）

```bash
# 编码
python encode_image.py \
    checkpoints/quick_verify/quick_verify_final.pth \
    --image examples/test.jpg \
    --save_dir output \
    --secret "Hello!"

# 解码
python decode_image.py \
    checkpoints/quick_verify/quick_verify_final.pth \
    --image output/test_hidden.png

# 应该输出: 'Hello!' (corrected X bit errors)
```

### 5. 导出ONNX（30秒）

```bash
python export_onnx.py \
    checkpoints/quick_verify/quick_verify_final.pth \
    --output_dir onnx_models \
    --test
```

## 🎨 使用场景示例

### 场景1: 给照片加水印

```bash
# 训练模型（一次性）
python train.py watermark_model --num_steps 100000

# 给图片加水印
python encode_image.py \
    checkpoints/watermark_model/watermark_model_final.pth \
    --images_dir my_photos/ \
    --save_dir watermarked_photos/ \
    --secret "©2024"
```

### 场景2: 隐藏超链接

```bash
# 隐藏URL（最多7字符，可以用短链接）
python encode_image.py \
    checkpoints/my_model.pth \
    --image poster.jpg \
    --save_dir output/ \
    --secret "bit.ly/x"

# 打印海报，用手机拍照后解码
python decode_image.py \
    checkpoints/my_model.pth \
    --image photo_of_poster.jpg
```

### 场景3: 批量处理

```bash
# 批量编码
python encode_image.py \
    checkpoints/my_model.pth \
    --images_dir input_folder/ \
    --save_dir output_folder/ \
    --secret "Batch"

# 批量解码
python decode_image.py \
    checkpoints/my_model.pth \
    --images_dir output_folder/
```

## 📝 常见快速问题

### Q: 最少需要多少训练图片？
**A:** 建议至少1000张。测试可以用更少（10-100张）。

### Q: 训练需要多长时间？
**A:** GPU: 13-20小时（140k步）。CPU: 5-7天。快速测试: 10分钟（5k步）。

### Q: 可以编码多长的消息？
**A:** 最多7个UTF-8字符（56数据位 + 40纠错位 = 96位 + 4填充 = 100位）。

### Q: 编码后的图片看起来一样吗？
**A:** 是的，肉眼几乎看不出差别（SSIM > 0.98）。

### Q: 打印后还能解码吗？
**A:** 是的！这是StegaStamp的核心特性。训练好的模型可以处理打印-拍照的图片。

### Q: ONNX模型可以在哪里用？
**A:** 任何支持ONNX的平台：移动端、Web、嵌入式设备等。

## 🔗 下一步

- 📖 阅读完整文档: [README.md](README.md)
- 🇨🇳 中文训练指南: [TRAINING_GUIDE_CN.md](TRAINING_GUIDE_CN.md)
- 🏗️ 了解项目结构: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- 🔬 查看测试: `cd tests && python run_all_tests.py`

## 💡 提示

1. **先用少量步数测试**：`--num_steps 1000` 确保配置正确
2. **监控BitAcc**：目标 >90%，低于70%说明训练有问题
3. **使用GPU**：训练速度提升10-100倍
4. **定期保存检查点**：自动每10k步保存一次
5. **使用TensorBoard**：可视化训练进度

## ⚡ 一键脚本

创建 `quick_start.sh`:

```bash
#!/bin/bash
set -e

echo "StegaStamp 快速开始"
echo "=================="

# 安装依赖
echo "1. 安装依赖..."
pip install -q -r requirements.txt

# 运行测试
echo "2. 运行测试..."
cd tests && python run_all_tests.py && cd ..

# 创建示例数据
echo "3. 准备示例数据..."
mkdir -p data/test_images examples output

# 快速训练
echo "4. 快速训练（1000步）..."
python train.py quick_start --num_steps 1000 --batch_size 2

# 测试编码
echo "5. 测试编码..."
# 注意：需要一张测试图片 examples/test.jpg
if [ -f "examples/test.jpg" ]; then
    python encode_image.py \
        checkpoints/quick_start/quick_start_final.pth \
        --image examples/test.jpg \
        --save_dir output \
        --secret "Works!"

    # 测试解码
    echo "6. 测试解码..."
    python decode_image.py \
        checkpoints/quick_start/quick_start_final.pth \
        --image output/test_hidden.png
fi

echo ""
echo "✓ 完成！查看 output/ 目录获取结果"
```

使用：
```bash
chmod +x quick_start.sh
./quick_start.sh
```

---

**现在就开始使用 StegaStamp！** 🚀

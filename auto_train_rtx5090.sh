#!/bin/bash

################################################################################
# StegaStamp 全自动训练脚本 - RTX 5090 优化版
#
# 功能：
# 1. 自动下载训练数据集
# 2. 使用针对 RTX 5090 优化的参数训练
# 3. 实时监控训练进度
# 4. 训练完成后自动导出 ONNX 模型
# 5. 自动测试模型
#
# 使用方法：
#   chmod +x auto_train_rtx5090.sh
#   ./auto_train_rtx5090.sh
################################################################################

set -e  # 遇到错误立即退出

# ============================================================================
# 配置参数
# ============================================================================

# 实验名称（使用时间戳）
EXP_NAME="stegastamp_rtx5090_$(date +%Y%m%d_%H%M%S)"

# 数据集配置
DATASET_NAME="div2k"  # 选项: div2k, coco, flickr30k
DATA_DIR="./data/${DATASET_NAME}"
TRAIN_PATH="${DATA_DIR}/train"

# RTX 5090 优化参数
BATCH_SIZE=16          # RTX 5090 显存充足，使用大 batch size
NUM_STEPS=140000       # 标准训练步数
LEARNING_RATE=0.0001   # 学习率
NUM_WORKERS=8          # 数据加载线程数

# 训练参数
SECRET_SIZE=100
L2_LOSS_SCALE=1.5
LPIPS_LOSS_SCALE=1.0
SECRET_LOSS_SCALE=1.0
G_LOSS_SCALE=1.0
JPEG_QUALITY=25
RND_NOISE=0.02
RND_BRI=0.3

# 路径配置
CHECKPOINT_DIR="./checkpoints/${EXP_NAME}"
ONNX_DIR="./onnx_models/${EXP_NAME}"
LOG_DIR="./logs/${EXP_NAME}"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 工具函数
# ============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 未安装，请先安装"
        exit 1
    fi
}

# ============================================================================
# 检查环境
# ============================================================================

check_environment() {
    print_header "检查环境"

    # 检查必要的命令
    log_info "检查必要工具..."
    check_command python3
    check_command pip3
    check_command wget
    check_command unzip

    # 检查 CUDA
    log_info "检查 CUDA 环境..."
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "未检测到 NVIDIA GPU 驱动"
        exit 1
    fi

    # 显示 GPU 信息
    log_info "GPU 信息："
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

    # 检查 Python 包
    log_info "检查 Python 依赖..."
    python3 -c "import torch; print(f'PyTorch 版本: {torch.__version__}')" || {
        log_error "PyTorch 未安装，正在安装依赖..."
        pip3 install -r requirements.txt
    }

    # 检查 CUDA 是否可用
    python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(f'CUDA 可用，设备数量: {torch.cuda.device_count()}')"

    log_success "环境检查完成"
}

# ============================================================================
# 下载数据集
# ============================================================================

download_div2k() {
    print_header "下载 DIV2K 数据集"

    local url="http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
    local zip_file="${DATA_DIR}/DIV2K_train_HR.zip"

    mkdir -p "${DATA_DIR}"

    if [ -d "${TRAIN_PATH}" ] && [ "$(ls -A ${TRAIN_PATH} 2>/dev/null | wc -l)" -gt 100 ]; then
        log_warning "数据集已存在，跳过下载"
        local num_images=$(find ${TRAIN_PATH} -name "*.png" -o -name "*.jpg" | wc -l)
        log_info "现有图片数量: ${num_images}"
        return 0
    fi

    log_info "下载 DIV2K 训练集 (~3.5GB)..."
    log_info "URL: ${url}"

    wget -c "${url}" -O "${zip_file}" --progress=bar:force 2>&1 | \
        grep --line-buffered "%" | \
        sed -u -e "s,\.,,g" | \
        awk '{printf("\r下载进度: %s", $2); fflush()}'
    echo ""

    log_info "解压数据集..."
    unzip -q "${zip_file}" -d "${DATA_DIR}"

    # DIV2K 解压后的目录结构: DIV2K_train_HR/*.png
    if [ -d "${DATA_DIR}/DIV2K_train_HR" ]; then
        mv "${DATA_DIR}/DIV2K_train_HR" "${TRAIN_PATH}"
    fi

    log_info "清理压缩包..."
    rm -f "${zip_file}"

    local num_images=$(find ${TRAIN_PATH} -name "*.png" -o -name "*.jpg" | wc -l)
    log_success "数据集下载完成，共 ${num_images} 张图片"
}

download_coco() {
    print_header "下载 COCO 数据集"

    local url="http://images.cocodataset.org/zips/train2017.zip"
    local zip_file="${DATA_DIR}/train2017.zip"

    mkdir -p "${DATA_DIR}"

    if [ -d "${TRAIN_PATH}" ] && [ "$(ls -A ${TRAIN_PATH} 2>/dev/null | wc -l)" -gt 1000 ]; then
        log_warning "数据集已存在，跳过下载"
        local num_images=$(find ${TRAIN_PATH} -name "*.jpg" -o -name "*.png" | wc -l)
        log_info "现有图片数量: ${num_images}"
        return 0
    fi

    log_info "下载 COCO train2017 (~18GB，这会需要一些时间)..."
    log_info "URL: ${url}"

    wget -c "${url}" -O "${zip_file}" --progress=bar:force

    log_info "解压数据集..."
    unzip -q "${zip_file}" -d "${DATA_DIR}"

    # COCO 解压后的目录结构: train2017/*.jpg
    if [ -d "${DATA_DIR}/train2017" ]; then
        mv "${DATA_DIR}/train2017" "${TRAIN_PATH}"
    fi

    log_info "清理压缩包..."
    rm -f "${zip_file}"

    local num_images=$(find ${TRAIN_PATH} -name "*.jpg" -o -name "*.png" | wc -l)
    log_success "数据集下载完成，共 ${num_images} 张图片"
}

download_dataset() {
    case ${DATASET_NAME} in
        div2k)
            download_div2k
            ;;
        coco)
            download_coco
            ;;
        *)
            log_error "未知的数据集: ${DATASET_NAME}"
            log_info "支持的数据集: div2k, coco"
            exit 1
            ;;
    esac
}

# ============================================================================
# 更新训练脚本中的数据路径
# ============================================================================

update_train_path() {
    print_header "配置训练路径"

    log_info "更新 train.py 中的数据路径为: ${TRAIN_PATH}"

    # 备份原文件
    cp train.py train.py.bak

    # 更新路径（使用 sed）
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|TRAIN_PATH = .*|TRAIN_PATH = '${TRAIN_PATH}'|" train.py
    else
        # Linux
        sed -i "s|TRAIN_PATH = .*|TRAIN_PATH = '${TRAIN_PATH}'|" train.py
    fi

    # 验证更改
    grep "TRAIN_PATH" train.py

    log_success "路径配置完成"
}

# ============================================================================
# 启动 TensorBoard
# ============================================================================

start_tensorboard() {
    print_header "启动 TensorBoard"

    # 检查是否已经在运行
    if pgrep -f "tensorboard" > /dev/null; then
        log_warning "TensorBoard 可能已在运行"
    fi

    log_info "启动 TensorBoard 监控..."
    nohup tensorboard --logdir logs --port 6006 --bind_all > tensorboard.log 2>&1 &

    sleep 2

    log_success "TensorBoard 已启动"
    log_info "访问: http://localhost:6006"
}

# ============================================================================
# 训练模型
# ============================================================================

train_model() {
    print_header "开始训练"

    log_info "实验名称: ${EXP_NAME}"
    log_info "训练参数:"
    echo "  - Batch Size: ${BATCH_SIZE}"
    echo "  - Training Steps: ${NUM_STEPS}"
    echo "  - Learning Rate: ${LEARNING_RATE}"
    echo "  - Secret Size: ${SECRET_SIZE}"
    echo "  - Dataset: ${DATASET_NAME} (${TRAIN_PATH})"
    echo ""

    log_info "开始训练... (预计 10-15 小时)"
    log_info "可以通过 TensorBoard 监控进度: http://localhost:6006"
    echo ""

    # 记录开始时间
    start_time=$(date +%s)

    # 运行训练
    python3 train.py "${EXP_NAME}" \
        --secret_size ${SECRET_SIZE} \
        --num_steps ${NUM_STEPS} \
        --batch_size ${BATCH_SIZE} \
        --lr ${LEARNING_RATE} \
        --l2_loss_scale ${L2_LOSS_SCALE} \
        --lpips_loss_scale ${LPIPS_LOSS_SCALE} \
        --secret_loss_scale ${SECRET_LOSS_SCALE} \
        --G_loss_scale ${G_LOSS_SCALE} \
        --jpeg_quality ${JPEG_QUALITY} \
        --rnd_noise ${RND_NOISE} \
        --rnd_bri ${RND_BRI} \
        2>&1 | tee "training_${EXP_NAME}.log"

    # 计算训练时间
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))

    log_success "训练完成！用时: ${hours}小时${minutes}分钟"

    # 检查最终模型
    if [ -f "${CHECKPOINT_DIR}/${EXP_NAME}_final.pth" ]; then
        local file_size=$(ls -lh "${CHECKPOINT_DIR}/${EXP_NAME}_final.pth" | awk '{print $5}')
        log_success "最终模型已保存: ${CHECKPOINT_DIR}/${EXP_NAME}_final.pth (${file_size})"
    else
        log_error "未找到最终模型文件"
        exit 1
    fi
}

# ============================================================================
# 测试模型
# ============================================================================

test_model() {
    print_header "测试模型"

    # 创建测试目录
    mkdir -p test_output

    # 从训练集中随机选择一张图片作为测试
    local test_image=$(find ${TRAIN_PATH} -name "*.png" -o -name "*.jpg" | head -1)

    if [ -z "${test_image}" ]; then
        log_warning "未找到测试图片，跳过测试"
        return 0
    fi

    log_info "使用测试图片: ${test_image}"

    # 测试编码
    log_info "测试编码..."
    python3 encode_image.py \
        "${CHECKPOINT_DIR}/${EXP_NAME}_final.pth" \
        --image "${test_image}" \
        --save_dir test_output \
        --secret "RTX5090"

    # 测试解码
    log_info "测试解码..."
    local encoded_image=$(ls test_output/*_hidden.png 2>/dev/null | head -1)

    if [ -n "${encoded_image}" ]; then
        python3 decode_image.py \
            "${CHECKPOINT_DIR}/${EXP_NAME}_final.pth" \
            --image "${encoded_image}"

        log_success "模型测试完成"
    else
        log_error "编码失败，未找到输出图片"
    fi
}

# ============================================================================
# 导出 ONNX
# ============================================================================

export_onnx() {
    print_header "导出 ONNX 模型"

    log_info "导出到: ${ONNX_DIR}"

    python3 export_onnx.py \
        "${CHECKPOINT_DIR}/${EXP_NAME}_final.pth" \
        --output_dir "${ONNX_DIR}" \
        --secret_size ${SECRET_SIZE} \
        --opset_version 14 \
        --test

    if [ $? -eq 0 ]; then
        log_success "ONNX 导出完成"

        # 显示文件大小
        log_info "ONNX 模型:"
        ls -lh "${ONNX_DIR}"/*.onnx

        # 测试 ONNX 模型
        log_info "测试 ONNX 模型..."

        local test_image=$(find ${TRAIN_PATH} -name "*.png" -o -name "*.jpg" | head -1)

        if [ -n "${test_image}" ]; then
            python3 onnx_inference.py \
                "${ONNX_DIR}/encoder.onnx" \
                "${ONNX_DIR}/decoder.onnx" \
                --test \
                --image "${test_image}" \
                --secret "ONNX" \
                --output test_output/onnx_test.png

            log_success "ONNX 模型测试通过"
        fi
    else
        log_error "ONNX 导出失败"
        exit 1
    fi
}

# ============================================================================
# 生成报告
# ============================================================================

generate_report() {
    print_header "生成训练报告"

    local report_file="report_${EXP_NAME}.txt"

    cat > "${report_file}" << EOF
StegaStamp 训练报告
================================================================================

实验名称: ${EXP_NAME}
训练时间: $(date)

训练配置:
--------
- GPU: RTX 5090
- 数据集: ${DATASET_NAME}
- 训练步数: ${NUM_STEPS}
- Batch Size: ${BATCH_SIZE}
- Learning Rate: ${LEARNING_RATE}

输出文件:
--------
- PyTorch 检查点: ${CHECKPOINT_DIR}/${EXP_NAME}_final.pth
- ONNX 编码器: ${ONNX_DIR}/encoder.onnx
- ONNX 解码器: ${ONNX_DIR}/decoder.onnx
- TensorBoard 日志: ${LOG_DIR}
- 训练日志: training_${EXP_NAME}.log

使用方法:
--------
1. 编码图片:
   python3 encode_image.py ${CHECKPOINT_DIR}/${EXP_NAME}_final.pth \\
       --image input.jpg --save_dir output --secret "Hello"

2. 解码图片:
   python3 decode_image.py ${CHECKPOINT_DIR}/${EXP_NAME}_final.pth \\
       --image output/input_hidden.png

3. ONNX 推理:
   python3 onnx_inference.py \\
       ${ONNX_DIR}/encoder.onnx \\
       ${ONNX_DIR}/decoder.onnx \\
       --test --image test.jpg --secret "Test"

TensorBoard:
-----------
查看训练过程: tensorboard --logdir logs --port 6006

================================================================================
EOF

    log_success "报告已生成: ${report_file}"

    # 显示报告
    cat "${report_file}"
}

# ============================================================================
# 清理函数
# ============================================================================

cleanup() {
    log_info "清理临时文件..."

    # 恢复训练脚本
    if [ -f "train.py.bak" ]; then
        mv train.py.bak train.py
        log_info "已恢复 train.py"
    fi
}

# 设置 trap 以在退出时清理
trap cleanup EXIT

# ============================================================================
# 主函数
# ============================================================================

main() {
    # 打印欢迎信息
    clear
    cat << "EOF"
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                StegaStamp 全自动训练脚本 - RTX 5090 优化版                   ║
║                                                                            ║
║  功能:                                                                      ║
║    • 自动下载训练数据集                                                       ║
║    • 使用针对 RTX 5090 优化的参数训练                                          ║
║    • 实时 TensorBoard 监控                                                  ║
║    • 训练完成后自动导出 ONNX                                                  ║
║    • 自动测试和生成报告                                                       ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
EOF

    echo ""
    log_info "开始执行全自动训练流程..."
    sleep 2

    # 1. 检查环境
    check_environment

    # 2. 下载数据集
    download_dataset

    # 3. 更新训练路径
    update_train_path

    # 4. 启动 TensorBoard
    start_tensorboard

    # 5. 训练模型
    train_model

    # 6. 测试模型
    test_model

    # 7. 导出 ONNX
    export_onnx

    # 8. 生成报告
    generate_report

    # 完成
    print_header "全部完成！"

    log_success "训练、测试和导出全部完成！"
    echo ""
    log_info "输出文件:"
    echo "  - PyTorch 模型: ${CHECKPOINT_DIR}/${EXP_NAME}_final.pth"
    echo "  - ONNX 模型: ${ONNX_DIR}/encoder.onnx, ${ONNX_DIR}/decoder.onnx"
    echo "  - 训练报告: report_${EXP_NAME}.txt"
    echo "  - 训练日志: training_${EXP_NAME}.log"
    echo ""
    log_info "下一步:"
    echo "  1. 查看训练报告了解详情"
    echo "  2. 使用模型进行编码/解码测试"
    echo "  3. 部署 ONNX 模型到生产环境"
    echo ""
    log_success "感谢使用 StegaStamp！🚀"
}

# 运行主函数
main "$@"

# ViTPose 人体姿态识别框架使用说明

## 框架概述

ViTPose 是一个基于 Vision Transformer 的人体姿态估计框架，提供了高精度的 2D/3D 人体姿态识别功能。该框架支持多种应用场景，包括人体姿态、手部姿态、面部关键点等。

## 主要特性

### 1. 支持的姿态类型
- **2D 人体姿态**：COCO、MPII、AIC、CrowdPose 等数据集
- **全身姿态**：包含面部、手部、身体关键点
- **手部姿态**：2D/3D 手部关键点识别
- **面部关键点**：面部特征点检测
- **动物姿态**：AP-10K、APT-36K 数据集

### 2. 模型性能
- **ViTPose-S**：73.8 AP (COCO)
- **ViTPose-B**：75.8 AP (COCO)
- **ViTPose-L**：78.3 AP (COCO)
- **ViTPose-H**：79.1 AP (COCO)

## 环境配置

### 基础环境要求
```bash
# Python 环境
Python 3.7+
PyTorch 1.9.0+
CUDA 11.1+

# 核心依赖
pip install torch torchvision
pip install opencv-python
pip install mmcv==1.3.9
pip install timm==0.4.9 einops
```

### 安装步骤
```bash
# 1. 安装 MMCV
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..

# 2. 安装 ViTPose
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .

# 3. 安装额外依赖
pip install timm==0.4.9 einops
```

## 快速开始

### 1. 图像姿态检测

#### 使用预训练模型进行推理
```bash
python demo/top_down_img_demo.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    /path/to/checkpoint.pth \
    --img-root /path/to/images \
    --json-file /path/to/annotations.json \
    --out-img-root /path/to/output \
    --show
```

#### 参数说明
- `pose_config`：模型配置文件路径
- `pose_checkpoint`：预训练模型权重路径
- `--img-root`：图像文件夹路径
- `--json-file`：COCO格式标注文件
- `--out-img-root`：输出图像保存路径
- `--show`：是否显示结果
- `--device`：推理设备 (默认: cuda:0)
- `--kpt-thr`：关键点置信度阈值 (默认: 0.3)

### 2. 视频姿态检测

#### 使用人体检测器进行视频处理
```bash
python demo/top_down_video_demo_with_mmdet.py \
    configs/detection/mmdet_coco/faster_rcnn_r50_fpn_coco.py \
    /path/to/detection_checkpoint.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    /path/to/pose_checkpoint.pth \
    --video-path /path/to/video.mp4 \
    --out-video-root /path/to/output \
    --show
```

#### 不使用检测器的视频处理
```bash
python demo/top_down_video_demo_full_frame_without_det.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    /path/to/pose_checkpoint.pth \
    --video-path /path/to/video.mp4 \
    --out-video-root /path/to/output \
    --show
```

### 3. 实时摄像头检测
```bash
python demo/webcam_demo.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    /path/to/pose_checkpoint.pth \
    --camera-id 0 \
    --show
```

## 模型配置

### 1. 配置文件结构
```
configs/
├── body/                    # 人体姿态
│   ├── 2d_kpt_sview_rgb_img/
│   │   └── topdown_heatmap/
│   │       ├── coco/       # COCO数据集配置
│   │       ├── mpii/       # MPII数据集配置
│   │       └── aic/        # AIC数据集配置
├── hand/                    # 手部姿态
├── face/                    # 面部关键点
├── wholebody/              # 全身姿态
└── animal/                 # 动物姿态
```

### 2. 常用模型配置
- **ViTPose-S**: `ViTPose_small_coco_256x192.py`
- **ViTPose-B**: `ViTPose_base_coco_256x192.py`
- **ViTPose-L**: `ViTPose_large_coco_256x192.py`
- **ViTPose-H**: `ViTPose_huge_coco_256x192.py`

## 预训练模型下载

### 官方模型权重
所有预训练模型可通过以下链接下载：
- **OneDrive**: https://1drv.ms/u/s!AimBgYV7JjTlgccZeiFjh4DJ7gjYyg?e=iTMdMq

### 推荐模型选择
- **快速推理**: ViTPose-S (73.8 AP)
- **平衡性能**: ViTPose-B (75.8 AP)
- **高精度**: ViTPose-L (78.3 AP)
- **最佳性能**: ViTPose-H (79.1 AP)

## 自定义开发

### 1. 训练自定义模型
```bash
# 单机训练
bash tools/dist_train.sh \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    4 \
    --cfg-options model.pretrained=/path/to/pretrained.pth \
    --seed 0

# 多机训练
python -m torch.distributed.launch \
    --nnodes 2 \
    --node_rank 0 \
    --nproc_per_node 4 \
    --master_addr localhost \
    --master_port 29500 \
    tools/train.py \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    --cfg-options model.pretrained=/path/to/pretrained.pth \
    --launcher pytorch \
    --seed 0
```

### 2. 模型测试
```bash
bash tools/dist_test.sh \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py \
    /path/to/checkpoint.pth \
    4
```

## 输出格式

### 关键点数据结构
```python
pose_results = [
    {
        'keypoints': np.array([[x1, y1, conf1], [x2, y2, conf2], ...]),
        'bbox': [x, y, w, h],
        'score': float
    },
    ...
]
```

### COCO 关键点定义 (17个关键点)
1. 鼻子 (nose)
2. 左眼 (left_eye)
3. 右眼 (right_eye)
4. 左耳 (left_ear)
5. 右耳 (right_ear)
6. 左肩 (left_shoulder)
7. 右肩 (right_shoulder)
8. 左肘 (left_elbow)
9. 右肘 (right_elbow)
10. 左手腕 (left_wrist)
11. 右手腕 (right_wrist)
12. 左髋 (left_hip)
13. 右髋 (right_hip)
14. 左膝 (left_knee)
15. 右膝 (right_knee)
16. 左脚踝 (left_ankle)
17. 右脚踝 (right_ankle)

## 性能优化

### 1. 推理速度优化
- 使用较小的模型 (ViTPose-S)
- 降低输入分辨率
- 使用 GPU 加速
- 启用 TensorRT 优化

### 2. 内存优化
- 使用 FP16 推理
- 批处理大小调整
- 模型量化

### 3. 精度优化
- 使用更大的模型 (ViTPose-H)
- 提高输入分辨率
- 数据增强训练

## 常见问题

### 1. 安装问题
**Q: MMCV 安装失败**
A: 确保使用正确的 MMCV 版本 (1.3.9)，并安装 MMCV_WITH_OPS

**Q: CUDA 版本不匹配**
A: 检查 PyTorch 和 CUDA 版本兼容性

### 2. 推理问题
**Q: 模型加载失败**
A: 检查模型权重文件路径和格式

**Q: 推理速度慢**
A: 使用 GPU 加速，选择较小的模型

### 3. 训练问题
**Q: 显存不足**
A: 减小 batch_size，使用梯度累积

**Q: 训练不收敛**
A: 检查学习率设置，数据预处理

## 集成到心理行为分析系统

### 1. 姿态检测集成
```python
from mmpose.apis import inference_top_down_pose_model, init_pose_model

# 初始化模型
pose_model = init_pose_model(
    config_file,
    checkpoint_file,
    device='cuda:0'
)

# 推理
pose_results, _ = inference_top_down_pose_model(
    pose_model,
    image,
    person_results,
    bbox_thr=0.3,
    format='xyxy'
)
```

### 2. 关键点提取
```python
def extract_keypoints(pose_results):
    """提取关键点坐标"""
    keypoints = []
    for pose in pose_results:
        kpts = pose['keypoints']
        # 提取头部、手部、身体关键点
        head_kpts = kpts[:5]  # 头部关键点
        arm_kpts = kpts[5:11]  # 手臂关键点
        body_kpts = kpts[11:]  # 身体关键点
        keypoints.append({
            'head': head_kpts,
            'arms': arm_kpts,
            'body': body_kpts
        })
    return keypoints
```

### 3. 姿态分类
```python
def classify_pose(keypoints):
    """基于关键点位置进行姿态分类"""
    # 实现40种基本姿态的分类逻辑
    # 例如：点头、摇头、耸肩等
    pass
```

## 总结

ViTPose 是一个功能强大的人体姿态识别框架，提供了：
- 高精度的 2D/3D 姿态检测
- 多种预训练模型选择
- 完整的训练和推理流程
- 丰富的应用场景支持

通过合理配置和使用，可以快速集成到心理行为分析系统中，为姿态识别提供可靠的技术基础。 
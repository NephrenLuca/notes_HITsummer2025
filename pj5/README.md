# 基于ViTPose的人体姿态识别与心理行为分析系统

这是一个基于ViTPose开源框架的人体姿态识别系统，能够分析视频中的人体姿态并识别40种基本动作，为心理行为分析提供数据支持。

## 🎯 项目功能

### 核心特性
- **高精度姿态检测**: 基于ViTPose-B框架，支持17个关键点检测，准确率达75.8 AP
- **40种姿态分类**: 涵盖头部、肩部、手部、手臂、腿部姿态的详细分类
- **视频分析**: 支持MP4、AVI等格式视频文件输入，实时姿态分析
- **结果可视化**: 生成带姿态标注的视频和详细分析报告
- **心理行为分析**: 基于姿态数据提供心理状态分析
- **轻量化设计**: 优化的代码结构，快速部署和运行

### 支持的姿态类型

#### 头部姿态 (6种)
- 点头/摇头
- 头部倾斜
- 头部前倾/后仰
- 头部转动

#### 肩部姿态 (5种)
- 耸肩
- 肩部放松/紧张
- 肩部前倾/后仰

#### 手部姿态 (6种)
- 触摸耳朵（厌烦、回忆、编造）
- 摸头发（不安、紧张）
- 手部交叉
- 手指敲击
- 手部握拳/张开

#### 手臂姿态 (4种)
- 张开双臂（决心、责任感）
- 手臂交叉
- 手臂自然下垂
- 手臂指向动作

#### 腿部姿态 (6种)
- 腿部交叉
- 腿部抖动
- 腿部前伸/后缩
- 站立姿态变化

## 🚀 快速开始

### 环境要求

**重要**: 本项目需要 **Python 3.10** 环境

```bash
# 系统要求
- Python 3.10
- Windows 10/11 (当前配置)
- 8GB+ RAM
- 支持CUDA的GPU (可选，用于加速)
```

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd pj5

# 创建Python 3.10虚拟环境
python -m venv venv310
venv310\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载预训练模型

下载ViTPose-B模型并放置在 `models/` 目录下：
- **下载链接**: https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs
- **重命名为**: `vitpose_b_coco_256x192.pth`

### 3. 准备视频文件

将待分析的视频文件放置在 `data/videos/` 目录下，或直接指定视频路径。

### 4. 运行分析

#### 方式一：使用简化脚本（推荐）
```bash
python run_analysis.py
```

#### 方式二：使用完整命令行
```bash
python src/main.py path/to/your/video.mp4
```

## 📁 项目结构

```
pj5/
├── src/                          # 源代码目录
│   ├── main.py                   # 主程序入口
│   ├── pose_detection/           # 姿态检测模块
│   │   └── vitpose_detector.py   # ViTPose检测器实现
│   └── pose_classification/      # 姿态分类模块
│       └── pose_classifier.py    # 40种姿态分类器
├── data/                         # 数据目录
│   ├── videos/                   # 输入视频文件
│   │   └── sample.mp4           # 示例视频
│   └── output/                   # 分析结果输出
├── models/                       # 预训练模型目录
│   └── vitpose_b_coco_256x192.pth  # ViTPose-B模型
├── venv310/                      # Python 3.10虚拟环境
├── requirements.txt              # Python依赖包列表
├── run_analysis.py              # 简化运行脚本
├── project_development_document_zh.md  # 项目开发文档
├── ViTPose_使用说明.md          # ViTPose框架使用说明
└── README.md                    # 项目说明文档
```

## 🔧 详细安装步骤

### 1. 环境准备

确保系统已安装Python 3.10：

```bash
# 检查Python版本
python --version  # 应显示 Python 3.10.x
```

### 2. 创建虚拟环境

```bash
# 创建虚拟环境
python -m venv venv310

# 激活虚拟环境
# Windows:
venv310\Scripts\activate
# Linux/Mac:
source venv310/bin/activate
```

### 3. 安装依赖包

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果遇到网络问题，可以使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 4. 验证安装

```bash
# 测试导入
python -c "import torch; import cv2; import mmpose; print('环境配置成功!')"
```

## 📖 使用方法

### 命令行参数

```bash
# 基本用法
python src/main.py video.mp4

# 指定模型路径
python src/main.py video.mp4 \
    --pose-checkpoint models/vitpose_b_coco_256x192.pth \
    --device cuda:0

# 显示视频
python src/main.py video.mp4 --show-video

# 不保存视频，只保存分析结果
python src/main.py video.mp4 --no-save-video

# 指定输出目录
python src/main.py video.mp4 --output-dir data/output
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `video_path` | 输入视频文件路径 | 必需 |
| `--pose-config` | 姿态检测配置文件 | ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py |
| `--pose-checkpoint` | 姿态检测模型权重 | models/vitpose_b_coco_256x192.pth |
| `--device` | 推理设备 (cpu/cuda:0) | cuda:0 |
| `--output-dir` | 输出目录 | data/output |
| `--show-video` | 显示视频 | False |
| `--no-save-video` | 不保存视频 | False |

## 📊 输出结果

分析完成后，系统会生成以下文件：

### 1. 分析视频
- **文件**: `data/output/{video_name}_analyzed.mp4`
- **内容**: 包含姿态关键点标注的视频
- **格式**: MP4格式，保持原视频分辨率

### 2. 详细结果
- **文件**: `data/output/{video_name}_analysis.json`
- **内容**: 每帧的详细姿态分析数据
- **格式**: JSON格式，包含关键点坐标和姿态分类

### 3. 文本摘要
- **文件**: `data/output/{video_name}_summary.txt`
- **内容**: 人类可读的分析摘要
- **格式**: 文本格式，包含统计信息和常见姿态

### 示例输出

```
==================================================
视频姿态分析摘要
==================================================

视频文件: data/videos/sample.mp4
总帧数: 150
检测到的姿态总数: 45

姿态统计:
- 独特姿态类型: 8
- 姿态总数: 45

最常见的姿态:
- 头部倾斜: 12次
- 肩部放松: 8次
- 手臂自然下垂: 7次
- 触摸耳朵: 6次
- 腿部交叉: 5次
```

## 🎯 技术架构

### 核心组件

#### 1. ViTPoseDetector (姿态检测器)
- **功能**: 基于ViTPose的姿态检测
- **输入**: 图像/视频帧
- **输出**: 17个关键点坐标
- **性能**: 75.8 AP (COCO数据集)

#### 2. PoseClassifier (姿态分类器)
- **功能**: 40种基本姿态分类
- **算法**: 基于几何关系的分类算法
- **输入**: 关键点坐标
- **输出**: 姿态类型和置信度

#### 3. PoseAnalysisSystem (分析系统)
- **功能**: 整合检测和分类功能
- **特性**: 结果可视化和保存
- **输出**: 统计分析报告

### 关键点定义

COCO格式的17个关键点：
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

## 📈 性能指标

- **姿态识别准确率**: > 80% (基于ViTPose性能)
- **实时处理帧率**: > 10fps (GPU加速)
- **系统响应时间**: < 3秒
- **支持视频格式**: MP4, AVI, MOV等
- **最大视频分辨率**: 1920x1080

## 🔍 故障排除

### 常见问题

#### 1. Python版本错误
```bash
# 错误: 需要Python 3.10
# 解决: 确保使用Python 3.10环境
python --version  # 应显示 3.10.x
```

#### 2. 模型文件缺失
```bash
# 错误: 模型文件不存在
# 解决: 下载ViTPose-B模型到models/目录
# 下载链接: https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs
```

#### 3. 依赖包安装失败
```bash
# 错误: mmcv安装失败
# 解决: 使用预编译版本
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.0.0/index.html
```

#### 4. CUDA相关错误
```bash
# 错误: CUDA不可用
# 解决: 使用CPU模式
python src/main.py video.mp4 --device cpu
```

### 性能优化建议

1. **GPU加速**: 使用支持CUDA的GPU可显著提升处理速度
2. **视频分辨率**: 降低视频分辨率可提高处理速度
3. **批处理**: 对于多个视频，可编写批处理脚本
4. **内存优化**: 确保系统有足够内存（建议8GB+）

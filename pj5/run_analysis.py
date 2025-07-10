#!/usr/bin/env python3
"""
简化的视频姿态分析脚本
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def get_video_path():
    """获取视频文件路径"""
    # 检查data/videos目录
    videos_dir = Path("data/videos")
    if videos_dir.exists():
        video_files = list(videos_dir.glob("*.mp4")) + list(videos_dir.glob("*.avi"))
        if video_files:
            print(f"发现视频文件: {video_files[0]}")
            return str(video_files[0])
    
    # 用户输入
    video_path = input("请输入视频文件路径: ").strip()
    if not video_path:
        print("未提供视频文件路径")
        return None
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return None
    
    return video_path

def check_model_files():
    """检查模型文件"""
    print("检查模型文件...")
    
    # 检查姿态检测模型
    pose_model_path = "models/vitpose_b_coco_256x192.pth"
    if not os.path.exists(pose_model_path):
        print(f"错误: 姿态检测模型文件不存在: {pose_model_path}")
        print("请下载ViTPose-B模型并放置在 models/ 目录下")
        print("下载链接: https://1drv.ms/u/s!AimBgYV7JjTlgSMjp1_NrV3VRSmK?e=Q1uZKs")
        print("下载后重命名为: vitpose_b_coco_256x192.pth")
        return False
    
    print("✓ 模型文件检查通过")
    return True

def run_pose_analysis(video_path):
    """运行姿态分析"""
    print(f"开始分析视频: {video_path}")
    
    try:
        # 导入torch
        import torch
        
        # 导入分析系统
        from src.main import PoseAnalysisSystem
        
        # 初始化系统
        system = PoseAnalysisSystem(
            pose_config='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
            pose_checkpoint='models/vitpose_b_coco_256x192.pth',
            device='cuda:0' if torch.cuda.is_available() else 'cpu'
        )
        
        # 生成输出路径
        video_name = Path(video_path).stem
        output_video_path = f"data/output/{video_name}_analyzed.mp4"
        
        # 确保输出目录存在
        os.makedirs("data/output", exist_ok=True)
        
        # 分析视频
        analysis_result = system.analyze_video(
            video_path=video_path,
            output_path=output_video_path,
            show_video=False,
            save_video=True,
            save_results=True
        )
        
        # 显示结果
        print("\n" + "=" * 50)
        print("分析完成!")
        print("=" * 50)
        
        overall = analysis_result['overall_summary']
        print(f"检测到的姿态类型: {overall['unique_pose_types']}")
        print(f"姿态总数: {overall['total_poses']}")
        print("\n最常见的姿态:")
        for pose_type, count in overall['most_common_poses'][:5]:
            print(f"  - {pose_type}: {count}次")
        
        print(f"\n结果已保存到: data/output/")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖包已正确安装")
        return False
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def main():
    """主函数"""
    print("=" * 50)
    print("基于ViTPose的人体姿态分析系统")
    print("=" * 50)
    
    # 获取视频文件
    video_path = get_video_path()
    if not video_path:
        return
    
    # 检查模型文件
    if not check_model_files():
        return
    
    # 运行分析
    success = run_pose_analysis(video_path)
    
    if success:
        print("\n✓ 分析完成!")
    else:
        print("\n✗ 分析失败，请检查错误信息")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
基于ViTPose的人体姿态识别主程序
读取视频文件并输出检测到的动作
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict
import cv2
import numpy as np
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.pose_detection.vitpose_detector import ViTPoseDetector
from src.pose_classification.pose_classifier import PoseClassifier

class PoseAnalysisSystem:
    """姿态分析系统"""
    
    def __init__(self, 
                 pose_config: str,
                 pose_checkpoint: str,
                 det_config: str = None,
                 det_checkpoint: str = None,
                 device: str = 'cuda:0'):
        """
        初始化姿态分析系统
        
        Args:
            pose_config: 姿态检测配置文件路径
            pose_checkpoint: 姿态检测模型权重路径
            det_config: 人体检测配置文件路径
            det_checkpoint: 人体检测模型权重路径
            device: 推理设备
        """
        print("正在初始化姿态分析系统...")
        
        # 初始化姿态检测器
        self.detector = ViTPoseDetector(
            pose_config=pose_config,
            pose_checkpoint=pose_checkpoint,
            det_config=det_config,
            det_checkpoint=det_checkpoint,
            device=device
        )
        
        # 初始化姿态分类器
        self.classifier = PoseClassifier()
        
        print("✓ 姿态分析系统初始化完成")
    
    def analyze_video(self, 
                     video_path: str,
                     output_path: str = None,
                     show_video: bool = False,
                     save_video: bool = True,
                     save_results: bool = True) -> Dict:
        """
        分析视频中的姿态
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            show_video: 是否显示视频
            save_video: 是否保存视频
            save_results: 是否保存分析结果
            
        Returns:
            分析结果字典
        """
        print(f"开始分析视频: {video_path}")
        
        # 检测视频中的姿态
        pose_results = self.detector.detect_pose_from_video(
            video_path=video_path,
            output_path=output_path,
            show_video=show_video,
            save_video=save_video
        )
        
        # 分析每帧的姿态
        frame_analysis = []
        total_poses = 0
        
        print("正在分析姿态...")
        for frame_idx, frame_poses in enumerate(tqdm(pose_results)):
            frame_analysis_result = {
                'frame': frame_idx,
                'poses': [],
                'summary': {}
            }
            
            # 分析每个检测到的人
            for person_pose in frame_poses:
                if 'keypoints' in person_pose:
                    keypoints = person_pose['keypoints']
                    
                    # 分类姿态
                    classified_poses = self.classifier.classify_pose(keypoints)
                    
                    if classified_poses:
                        frame_analysis_result['poses'].extend(classified_poses)
                        total_poses += len(classified_poses)
            
            # 获取帧摘要
            frame_analysis_result['summary'] = self.classifier.get_pose_summary(
                frame_analysis_result['poses']
            )
            frame_analysis.append(frame_analysis_result)
        
        # 生成总体分析结果
        analysis_result = {
            'video_path': video_path,
            'total_frames': len(pose_results),
            'total_poses_detected': total_poses,
            'frame_analysis': frame_analysis,
            'overall_summary': self._generate_overall_summary(frame_analysis)
        }
        
        # 保存结果
        if save_results:
            self._save_analysis_results(analysis_result, video_path)
        
        print(f"✓ 视频分析完成，共检测到 {total_poses} 个姿态")
        return analysis_result
    
    def _generate_overall_summary(self, frame_analysis: List[Dict]) -> Dict:
        """生成总体摘要"""
        all_poses = []
        for frame in frame_analysis:
            all_poses.extend(frame['poses'])
        
        # 统计所有姿态类型
        pose_counts = {}
        for pose in all_poses:
            pose_type = pose['type'].value
            if pose_type in pose_counts:
                pose_counts[pose_type] += 1
            else:
                pose_counts[pose_type] = 1
        
        # 按频率排序
        sorted_poses = sorted(pose_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'total_poses': len(all_poses),
            'unique_pose_types': len(pose_counts),
            'most_common_poses': sorted_poses[:10],  # 前10个最常见的姿态
            'pose_frequency': pose_counts
        }
    
    def _save_analysis_results(self, analysis_result: Dict, video_path: str):
        """保存分析结果"""
        # 创建输出目录
        output_dir = Path("data/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成输出文件名
        video_name = Path(video_path).stem
        output_file = output_dir / f"{video_name}_analysis.json"
        
        # 保存JSON结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        
        # 生成文本摘要
        summary_file = output_dir / f"{video_name}_summary.txt"
        self._save_text_summary(analysis_result, summary_file)
        
        print(f"✓ 分析结果已保存到: {output_file}")
        print(f"✓ 文本摘要已保存到: {summary_file}")
    
    def _save_text_summary(self, analysis_result: Dict, output_file: Path):
        """保存文本摘要"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write("视频姿态分析摘要\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"视频文件: {analysis_result['video_path']}\n")
            f.write(f"总帧数: {analysis_result['total_frames']}\n")
            f.write(f"检测到的姿态总数: {analysis_result['total_poses_detected']}\n\n")
            
            # 总体摘要
            overall = analysis_result['overall_summary']
            f.write("姿态统计:\n")
            f.write(f"- 独特姿态类型: {overall['unique_pose_types']}\n")
            f.write(f"- 姿态总数: {overall['total_poses']}\n\n")
            
            # 最常见的姿态
            f.write("最常见的姿态:\n")
            for pose_type, count in overall['most_common_poses']:
                f.write(f"- {pose_type}: {count}次\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.write("详细分析完成\n")
            f.write("=" * 50 + "\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基于ViTPose的人体姿态分析系统")
    parser.add_argument('video_path', help='输入视频文件路径')
    parser.add_argument('--pose-config', 
                       default='ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
                       help='姿态检测配置文件路径')
    parser.add_argument('--pose-checkpoint', 
                       default='models/vitpose_b_coco_256x192.pth',
                       help='姿态检测模型权重路径')
    parser.add_argument('--det-config',
                       default='ViTPose/configs/detection/mmdet_coco/faster_rcnn_r50_fpn_coco.py',
                       help='人体检测配置文件路径')
    parser.add_argument('--det-checkpoint',
                       default='models/faster_rcnn_r50_fpn_1x_coco.pth',
                       help='人体检测模型权重路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--output-dir', default='data/output', help='输出目录')
    parser.add_argument('--show-video', action='store_true', help='显示视频')
    parser.add_argument('--no-save-video', action='store_true', help='不保存视频')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.video_path):
        print(f"错误: 视频文件不存在: {args.video_path}")
        return
    
    # 检查模型文件
    if not os.path.exists(args.pose_checkpoint):
        print(f"错误: 姿态检测模型文件不存在: {args.pose_checkpoint}")
        print("请下载预训练模型并放置在 models/ 目录下")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出视频路径
    video_name = Path(args.video_path).stem
    output_video_path = output_dir / f"{video_name}_analyzed.mp4"
    
    try:
        # 初始化系统
        system = PoseAnalysisSystem(
            pose_config=args.pose_config,
            pose_checkpoint=args.pose_checkpoint,
            det_config=args.det_config,
            det_checkpoint=args.det_checkpoint,
            device=args.device
        )
        
        # 分析视频
        analysis_result = system.analyze_video(
            video_path=args.video_path,
            output_path=str(output_video_path) if not args.no_save_video else None,
            show_video=args.show_video,
            save_video=not args.no_save_video,
            save_results=True
        )
        
        # 打印摘要
        print("\n" + "=" * 50)
        print("分析完成!")
        print("=" * 50)
        overall = analysis_result['overall_summary']
        print(f"检测到的姿态类型: {overall['unique_pose_types']}")
        print(f"姿态总数: {overall['total_poses']}")
        print("\n最常见的姿态:")
        for pose_type, count in overall['most_common_poses'][:5]:
            print(f"  - {pose_type}: {count}次")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 
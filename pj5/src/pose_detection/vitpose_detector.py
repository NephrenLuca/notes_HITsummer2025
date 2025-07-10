"""
ViTPose姿态检测器
基于ViTPose框架的人体姿态检测模块
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
    from mmpose.datasets import DatasetInfo
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import process_mmdet_results
    HAS_MMPOSE = True
except ImportError:
    HAS_MMPOSE = False
    print("警告: 未安装mmpose，请先安装相关依赖")

class ViTPoseDetector:
    """ViTPose姿态检测器"""
    
    def __init__(self, 
                 pose_config: str,
                 pose_checkpoint: str,
                 det_config: Optional[str] = None,
                 det_checkpoint: Optional[str] = None,
                 device: str = 'cuda:0'):
        """
        初始化ViTPose检测器
        
        Args:
            pose_config: 姿态检测配置文件路径
            pose_checkpoint: 姿态检测模型权重路径
            det_config: 人体检测配置文件路径（可选）
            det_checkpoint: 人体检测模型权重路径（可选）
            device: 推理设备
        """
        if not HAS_MMPOSE:
            raise ImportError("请先安装mmpose相关依赖")
        
        self.device = device
        self.pose_model = None
        self.det_model = None
        self.dataset_info = None
        
        # 初始化姿态检测模型
        self._init_pose_model(pose_config, pose_checkpoint)
        
        # 初始化人体检测模型（如果提供）
        if det_config and det_checkpoint:
            self._init_det_model(det_config, det_checkpoint)
    
    def _init_pose_model(self, config_path: str, checkpoint_path: str):
        """初始化姿态检测模型"""
        print(f"正在加载姿态检测模型: {checkpoint_path}")
        self.pose_model = init_pose_model(
            config_path, 
            checkpoint_path, 
            device=self.device
        )
        
        # 获取数据集信息
        dataset = self.pose_model.cfg.data['test']['type']
        dataset_info = self.pose_model.cfg.data['test'].get('dataset_info', None)
        if dataset_info is not None:
            self.dataset_info = DatasetInfo(dataset_info)
        
        print("✓ 姿态检测模型加载完成")
    
    def _init_det_model(self, config_path: str, checkpoint_path: str):
        """初始化人体检测模型"""
        print(f"正在加载人体检测模型: {checkpoint_path}")
        self.det_model = init_detector(
            config_path, 
            checkpoint_path, 
            device=self.device
        )
        print("✓ 人体检测模型加载完成")
    
    def detect_pose_from_image(self, 
                              image: np.ndarray,
                              person_results: Optional[List[Dict]] = None,
                              bbox_thr: float = 0.3,
                              kpt_thr: float = 0.3) -> List[Dict]:
        """
        从图像中检测人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            person_results: 人体检测结果（如果为None，将使用全图检测）
            bbox_thr: 边界框置信度阈值
            kpt_thr: 关键点置信度阈值
            
        Returns:
            姿态检测结果列表
        """
        if person_results is None and self.det_model is not None:
            # 使用人体检测模型
            det_results = inference_detector(self.det_model, image)
            person_results = process_mmdet_results(det_results, cat_id=1)  # person class
        elif person_results is None:
            # 使用全图检测
            h, w = image.shape[:2]
            person_results = [{'bbox': [0, 0, w, h]}]
        
        # 姿态检测
        pose_results, _ = inference_top_down_pose_model(
            self.pose_model,
            image,
            person_results,
            bbox_thr=bbox_thr,
            format='xyxy',
            dataset=self.pose_model.cfg.data['test']['type'],
            dataset_info=self.dataset_info,
            return_heatmap=False,
            outputs=None
        )
        
        return pose_results
    
    def detect_pose_from_video(self, 
                              video_path: str,
                              output_path: Optional[str] = None,
                              show_video: bool = False,
                              save_video: bool = True,
                              bbox_thr: float = 0.3,
                              kpt_thr: float = 0.3) -> List[List[Dict]]:
        """
        从视频中检测人体姿态
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径
            show_video: 是否显示视频
            save_video: 是否保存视频
            bbox_thr: 边界框置信度阈值
            kpt_thr: 关键点置信度阈值
            
        Returns:
            每帧的姿态检测结果列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, {total_frames}帧")
        
        # 设置输出视频
        video_writer = None
        if save_video and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_pose_results = []
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                print(f"处理第 {frame_count}/{total_frames} 帧...")
                
                # 检测姿态
                pose_results = self.detect_pose_from_image(
                    frame, 
                    bbox_thr=bbox_thr, 
                    kpt_thr=kpt_thr
                )
                all_pose_results.append(pose_results)
                
                # 可视化结果
                if show_video or save_video:
                    vis_frame = self._visualize_pose_results(frame, pose_results, kpt_thr)
                    
                    if show_video:
                        cv2.imshow('Pose Detection', vis_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    if save_video and video_writer:
                        video_writer.write(vis_frame)
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if show_video:
                cv2.destroyAllWindows()
        
        print(f"✓ 视频处理完成，共处理 {len(all_pose_results)} 帧")
        return all_pose_results
    
    def _visualize_pose_results(self, 
                               image: np.ndarray, 
                               pose_results: List[Dict],
                               kpt_thr: float = 0.3) -> np.ndarray:
        """
        可视化姿态检测结果
        
        Args:
            image: 原始图像
            pose_results: 姿态检测结果
            kpt_thr: 关键点置信度阈值
            
        Returns:
            可视化后的图像
        """
        from mmpose.apis import vis_pose_result
        
        vis_image = vis_pose_result(
            self.pose_model,
            image,
            pose_results,
            dataset=self.pose_model.cfg.data['test']['type'],
            dataset_info=self.dataset_info,
            kpt_score_thr=kpt_thr,
            radius=4,
            thickness=2,
            show=False
        )
        
        return vis_image
    
    def extract_keypoints(self, pose_results: List[Dict]) -> List[np.ndarray]:
        """
        提取关键点坐标
        
        Args:
            pose_results: 姿态检测结果
            
        Returns:
            关键点坐标列表
        """
        keypoints_list = []
        for pose in pose_results:
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                # 过滤低置信度的关键点
                valid_keypoints = keypoints[keypoints[:, 2] > 0.3]
                keypoints_list.append(valid_keypoints)
        
        return keypoints_list
    
    def get_pose_statistics(self, pose_results: List[Dict]) -> Dict:
        """
        获取姿态统计信息
        
        Args:
            pose_results: 姿态检测结果
            
        Returns:
            统计信息字典
        """
        if not pose_results:
            return {}
        
        stats = {
            'num_persons': len(pose_results),
            'keypoints_per_person': [],
            'confidence_scores': []
        }
        
        for pose in pose_results:
            if 'keypoints' in pose:
                keypoints = pose['keypoints']
                num_keypoints = len(keypoints)
                avg_confidence = np.mean(keypoints[:, 2]) if num_keypoints > 0 else 0
                
                stats['keypoints_per_person'].append(num_keypoints)
                stats['confidence_scores'].append(avg_confidence)
        
        if stats['confidence_scores']:
            stats['avg_confidence'] = np.mean(stats['confidence_scores'])
            stats['max_confidence'] = np.max(stats['confidence_scores'])
            stats['min_confidence'] = np.min(stats['confidence_scores'])
        
        return stats 
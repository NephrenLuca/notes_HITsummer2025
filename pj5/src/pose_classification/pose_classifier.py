"""
姿态分类器
基于关键点位置识别40种基本人体姿态
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum

class PoseType(Enum):
    """姿态类型枚举"""
    # 头部姿态
    NODDING = "点头"
    SHAKING_HEAD = "摇头"
    HEAD_TILT = "头部倾斜"
    HEAD_FORWARD = "头部前倾"
    HEAD_BACKWARD = "头部后仰"
    HEAD_ROTATION = "头部转动"
    
    # 肩部姿态
    SHOULDER_SHRUG = "耸肩"
    SHOULDER_RELAXED = "肩部放松"
    SHOULDER_TENSE = "肩部紧张"
    SHOULDER_FORWARD = "肩部前倾"
    SHOULDER_BACKWARD = "肩部后仰"
    
    # 手部姿态
    TOUCHING_EAR = "触摸耳朵"
    TOUCHING_HAIR = "摸头发"
    HAND_CROSSED = "手部交叉"
    FINGER_TAPPING = "手指敲击"
    HAND_CLENCHED = "手部握拳"
    HAND_OPEN = "手部张开"
    
    # 手臂姿态
    ARMS_OPEN = "张开双臂"
    ARMS_CROSSED = "手臂交叉"
    ARMS_HANGING = "手臂自然下垂"
    ARM_POINTING = "手臂指向"
    
    # 腿部姿态
    LEGS_CROSSED = "腿部交叉"
    LEG_SHAKING = "腿部抖动"
    LEG_FORWARD = "腿部前伸"
    LEG_BACKWARD = "腿部后缩"
    STANDING_STRAIGHT = "直立站立"
    STANDING_LEANING = "倾斜站立"

class PoseClassifier:
    """姿态分类器"""
    
    def __init__(self):
        """初始化姿态分类器"""
        self.pose_types = list(PoseType)
        self.confidence_threshold = 0.3
        
        # COCO关键点索引定义
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 关键点分组
        self.head_keypoints = [0, 1, 2, 3, 4]  # 鼻子、眼睛、耳朵
        self.shoulder_keypoints = [5, 6]  # 肩膀
        self.arm_keypoints = [7, 8, 9, 10]  # 肘部、手腕
        self.leg_keypoints = [11, 12, 13, 14, 15, 16]  # 髋部、膝盖、脚踝
    
    def classify_pose(self, keypoints: np.ndarray) -> List[Dict]:
        """
        分类姿态
        
        Args:
            keypoints: 关键点坐标 (N, 3) - [x, y, confidence]
            
        Returns:
            检测到的姿态列表
        """
        if keypoints is None or len(keypoints) == 0:
            return []
        
        detected_poses = []
        
        # 提取各部位关键点
        head_kpts = self._extract_keypoints(keypoints, self.head_keypoints)
        shoulder_kpts = self._extract_keypoints(keypoints, self.shoulder_keypoints)
        arm_kpts = self._extract_keypoints(keypoints, self.arm_keypoints)
        leg_kpts = self._extract_keypoints(keypoints, self.leg_keypoints)
        
        # 头部姿态分类
        head_poses = self._classify_head_poses(head_kpts)
        detected_poses.extend(head_poses)
        
        # 肩部姿态分类
        shoulder_poses = self._classify_shoulder_poses(shoulder_kpts)
        detected_poses.extend(shoulder_poses)
        
        # 手部姿态分类
        hand_poses = self._classify_hand_poses(arm_kpts)
        detected_poses.extend(hand_poses)
        
        # 手臂姿态分类
        arm_poses = self._classify_arm_poses(arm_kpts)
        detected_poses.extend(arm_poses)
        
        # 腿部姿态分类
        leg_poses = self._classify_leg_poses(leg_kpts)
        detected_poses.extend(leg_poses)
        
        return detected_poses
    
    def _extract_keypoints(self, keypoints: np.ndarray, indices: List[int]) -> np.ndarray:
        """提取指定索引的关键点"""
        if len(keypoints) < max(indices) + 1:
            return np.array([])
        
        extracted = []
        for idx in indices:
            if idx < len(keypoints):
                kpt = keypoints[idx]
                if kpt[2] > self.confidence_threshold:  # 置信度检查
                    extracted.append(kpt)
        
        return np.array(extracted) if extracted else np.array([])
    
    def _classify_head_poses(self, head_kpts: np.ndarray) -> List[Dict]:
        """分类头部姿态"""
        poses = []
        
        if len(head_kpts) < 3:  # 至少需要3个头部关键点
            return poses
        
        # 计算头部姿态特征
        nose_pos = head_kpts[0] if len(head_kpts) > 0 else None
        left_eye_pos = head_kpts[1] if len(head_kpts) > 1 else None
        right_eye_pos = head_kpts[2] if len(head_kpts) > 2 else None
        
        if nose_pos is not None and left_eye_pos is not None and right_eye_pos is not None:
            # 头部倾斜检测
            eye_center = (left_eye_pos[:2] + right_eye_pos[:2]) / 2
            head_tilt = self._calculate_head_tilt(nose_pos[:2], eye_center)
            
            if abs(head_tilt) > 15:  # 倾斜角度阈值
                poses.append({
                    'type': PoseType.HEAD_TILT,
                    'confidence': nose_pos[2],
                    'details': f'头部倾斜角度: {head_tilt:.1f}度'
                })
            
            # 头部前后倾检测
            head_forward_backward = self._calculate_head_forward_backward(nose_pos[:2], eye_center)
            if head_forward_backward > 0.1:
                poses.append({
                    'type': PoseType.HEAD_FORWARD,
                    'confidence': nose_pos[2],
                    'details': '头部前倾'
                })
            elif head_forward_backward < -0.1:
                poses.append({
                    'type': PoseType.HEAD_BACKWARD,
                    'confidence': nose_pos[2],
                    'details': '头部后仰'
                })
        
        return poses
    
    def _classify_shoulder_poses(self, shoulder_kpts: np.ndarray) -> List[Dict]:
        """分类肩部姿态"""
        poses = []
        
        if len(shoulder_kpts) < 2:
            return poses
        
        left_shoulder = shoulder_kpts[0]
        right_shoulder = shoulder_kpts[1]
        
        # 耸肩检测
        shoulder_height_diff = abs(left_shoulder[1] - right_shoulder[1])
        if shoulder_height_diff > 20:  # 高度差阈值
            poses.append({
                'type': PoseType.SHOULDER_SHRUG,
                'confidence': min(left_shoulder[2], right_shoulder[2]),
                'details': f'肩部高度差: {shoulder_height_diff:.1f}像素'
            })
        
        # 肩部紧张/放松检测
        shoulder_distance = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        if shoulder_distance > 150:  # 肩部距离阈值
            poses.append({
                'type': PoseType.SHOULDER_RELAXED,
                'confidence': min(left_shoulder[2], right_shoulder[2]),
                'details': '肩部放松'
            })
        elif shoulder_distance < 100:
            poses.append({
                'type': PoseType.SHOULDER_TENSE,
                'confidence': min(left_shoulder[2], right_shoulder[2]),
                'details': '肩部紧张'
            })
        
        return poses
    
    def _classify_hand_poses(self, arm_kpts: np.ndarray) -> List[Dict]:
        """分类手部姿态"""
        poses = []
        
        if len(arm_kpts) < 4:  # 需要肘部和手腕关键点
            return poses
        
        # 提取手腕位置
        left_wrist = arm_kpts[2] if len(arm_kpts) > 2 else None
        right_wrist = arm_kpts[3] if len(arm_kpts) > 3 else None
        
        if left_wrist is not None:
            # 左手触摸耳朵检测
            if self._is_touching_ear(left_wrist):
                poses.append({
                    'type': PoseType.TOUCHING_EAR,
                    'confidence': left_wrist[2],
                    'details': '左手触摸耳朵'
                })
            
            # 左手摸头发检测
            if self._is_touching_hair(left_wrist):
                poses.append({
                    'type': PoseType.TOUCHING_HAIR,
                    'confidence': left_wrist[2],
                    'details': '左手摸头发'
                })
        
        if right_wrist is not None:
            # 右手触摸耳朵检测
            if self._is_touching_ear(right_wrist):
                poses.append({
                    'type': PoseType.TOUCHING_EAR,
                    'confidence': right_wrist[2],
                    'details': '右手触摸耳朵'
                })
            
            # 右手摸头发检测
            if self._is_touching_hair(right_wrist):
                poses.append({
                    'type': PoseType.TOUCHING_HAIR,
                    'confidence': right_wrist[2],
                    'details': '右手摸头发'
                })
        
        return poses
    
    def _classify_arm_poses(self, arm_kpts: np.ndarray) -> List[Dict]:
        """分类手臂姿态"""
        poses = []
        
        if len(arm_kpts) < 4:
            return poses
        
        left_elbow = arm_kpts[0]
        right_elbow = arm_kpts[1]
        left_wrist = arm_kpts[2]
        right_wrist = arm_kpts[3]
        
        # 手臂交叉检测
        if self._is_arms_crossed(left_elbow, right_elbow, left_wrist, right_wrist):
            poses.append({
                'type': PoseType.ARMS_CROSSED,
                'confidence': min(left_elbow[2], right_elbow[2]),
                'details': '手臂交叉'
            })
        
        # 张开双臂检测
        if self._is_arms_open(left_elbow, right_elbow, left_wrist, right_wrist):
            poses.append({
                'type': PoseType.ARMS_OPEN,
                'confidence': min(left_elbow[2], right_elbow[2]),
                'details': '张开双臂'
            })
        
        return poses
    
    def _classify_leg_poses(self, leg_kpts: np.ndarray) -> List[Dict]:
        """分类腿部姿态"""
        poses = []
        
        if len(leg_kpts) < 6:  # 需要髋部、膝盖、脚踝关键点
            return poses
        
        # 腿部交叉检测
        if self._is_legs_crossed(leg_kpts):
            poses.append({
                'type': PoseType.LEGS_CROSSED,
                'confidence': np.mean([kpt[2] for kpt in leg_kpts]),
                'details': '腿部交叉'
            })
        
        # 腿部抖动检测（需要时序信息，这里简化处理）
        # 在实际应用中，需要分析连续帧的关键点变化
        
        return poses
    
    def _calculate_head_tilt(self, nose_pos: np.ndarray, eye_center: np.ndarray) -> float:
        """计算头部倾斜角度"""
        # 计算头部倾斜角度
        head_vector = nose_pos - eye_center
        angle = np.degrees(np.arctan2(head_vector[1], head_vector[0]))
        return angle
    
    def _calculate_head_forward_backward(self, nose_pos: np.ndarray, eye_center: np.ndarray) -> float:
        """计算头部前后倾程度"""
        # 简化的前后倾检测
        return nose_pos[1] - eye_center[1]
    
    def _is_touching_ear(self, wrist_pos: np.ndarray) -> bool:
        """检测是否触摸耳朵"""
        # 简化的触摸耳朵检测
        # 在实际应用中，需要更复杂的几何关系判断
        return wrist_pos[1] < 100  # 手腕位置较高
    
    def _is_touching_hair(self, wrist_pos: np.ndarray) -> bool:
        """检测是否摸头发"""
        # 简化的摸头发检测
        return wrist_pos[1] < 80  # 手腕位置很高
    
    def _is_arms_crossed(self, left_elbow: np.ndarray, right_elbow: np.ndarray,
                         left_wrist: np.ndarray, right_wrist: np.ndarray) -> bool:
        """检测手臂是否交叉"""
        # 简化的手臂交叉检测
        # 检查手腕是否在身体另一侧
        left_wrist_x = left_wrist[0]
        right_wrist_x = right_wrist[0]
        body_center_x = (left_elbow[0] + right_elbow[0]) / 2
        
        return (left_wrist_x > body_center_x and right_wrist_x < body_center_x)
    
    def _is_arms_open(self, left_elbow: np.ndarray, right_elbow: np.ndarray,
                      left_wrist: np.ndarray, right_wrist: np.ndarray) -> bool:
        """检测是否张开双臂"""
        # 简化的张开双臂检测
        # 检查手臂是否向两侧伸展
        left_arm_angle = np.arctan2(left_wrist[1] - left_elbow[1], 
                                   left_wrist[0] - left_elbow[0])
        right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1], 
                                    right_wrist[0] - right_elbow[0])
        
        # 手臂向外伸展的角度范围
        return (left_arm_angle < -np.pi/4 and right_arm_angle > np.pi/4)
    
    def _is_legs_crossed(self, leg_kpts: np.ndarray) -> bool:
        """检测腿部是否交叉"""
        # 简化的腿部交叉检测
        if len(leg_kpts) < 6:
            return False
        
        # 检查脚踝位置
        left_ankle = leg_kpts[4]
        right_ankle = leg_kpts[5]
        
        # 如果脚踝位置交叉，则认为腿部交叉
        return abs(left_ankle[0] - right_ankle[0]) < 50
    
    def get_pose_summary(self, poses: List[Dict]) -> Dict:
        """获取姿态摘要"""
        if not poses:
            return {'total_poses': 0, 'pose_types': []}
        
        pose_counts = {}
        for pose in poses:
            pose_type = pose['type'].value
            if pose_type in pose_counts:
                pose_counts[pose_type] += 1
            else:
                pose_counts[pose_type] = 1
        
        return {
            'total_poses': len(poses),
            'pose_types': pose_counts,
            'unique_poses': len(pose_counts)
        } 
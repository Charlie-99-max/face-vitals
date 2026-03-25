import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os


class FaceDetector:
    """人脸检测器 - 使用MediaPipe Face Landmarker"""

    def __init__(self):
        # 获取模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', 'assets', 'face_landmarker.task')

        # 创建FaceLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # 关键点索引 (MediaPipe Face Landmarker)
        self.FOREHEAD_POINTS = list(range(10, 21))
        self.LEFT_CHEEK_POINTS = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
        self.RIGHT_CHEEK_POINTS = [280, 281, 282, 283, 284, 285, 286, 287, 288, 289]

        # 表情关键点
        self.LEFT_EYE_POINTS = [33, 133, 160, 144, 153, 154, 155, 157, 158, 159]
        self.RIGHT_EYE_POINTS = [362, 263, 387, 373, 380, 374, 381, 382, 384, 385]
        self.LEFT_EYEBROW_POINTS = [107, 70, 63, 105, 66]
        self.RIGHT_EYEBROW_POINTS = [336, 296, 283, 295, 300]
        self.MOUTH_POINTS = [13, 14, 78, 308]
        self.NOSE_POINTS = [1, 4, 5, 6, 19]

        # 用于存储上一帧的时间戳
        self._timestamp = 0

    def detect(self, frame):
        """
        检测人脸
        Returns:
            landmarks: 面部关键点列表
            face_roi: 人脸区域 (x, y, w, h)
        """
        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测
        self._timestamp += 1
        result = self.detector.detect_for_video(mp_image, self._timestamp * 10)
    
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None, None, None # 多返回一个 None

        landmarks = result.face_landmarks[0]
        blendshapes = result.face_blendshapes[0] if result.face_blendshapes else None

        # 计算人脸边界框
        h, w = frame.shape[:2]
        x_coords = [lm.x * w for lm in landmarks]
        y_coords = [lm.y * h for lm in landmarks]

        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))

        # 添加边距
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        face_roi = (x_min, y_min, x_max - x_min, y_max - y_min)

        return landmarks, face_roi, blendshapes

    def extract_roi_pixels(self, frame, landmarks):
        """使用多边形掩膜提取精确的皮肤区域像素均值"""
        h, w = frame.shape[:2]

        def get_poly_mean(indices):
            """获取精确多边形区域内的像素均值"""
            points = []
            for idx in indices:
                if idx < len(landmarks):
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    points.append([x, y])

            if len(points) < 3:
                return None

            pts = np.array([points], dtype=np.int32)
            # 创建纯黑掩膜
            mask = np.zeros((h, w), dtype=np.uint8)
            # 填充多边形为纯白
            cv2.fillPoly(mask, pts, 255)
            # 仅计算白色掩膜区域的 BGR 均值
            mean_val = cv2.mean(frame, mask=mask)[:3]
            return mean_val

        # 使用多边形掩膜获取 BGR 均值
        forehead_mean = get_poly_mean(self.FOREHEAD_POINTS[:5])
        left_cheek_mean = get_poly_mean(self.LEFT_CHEEK_POINTS[:5])
        right_cheek_mean = get_poly_mean(self.RIGHT_CHEEK_POINTS[:5])

        # 返回均值 (B, G, R) 的 tuple
        return forehead_mean, left_cheek_mean, right_cheek_mean

    def extract_expression_features(self, landmarks, blendshapes, frame_shape):
        """
        提取面部表情特征 - 使用 Blendshapes
        优先使用 MediaPipe 的 Blendshapes（标准化，不受距离影响）
        """
        features = {}

        # ===== 优先使用 Blendshapes（推荐） =====
        if blendshapes:
            # 创建 blendshape 名称到值的映射
            blendshape_dict = {bs.category_name: bs.score for bs in blendshapes}

            # 关键情绪指标 - 标准化的 0-1 值
            features['blendshapes'] = blendshape_dict

            # 常用情绪 blendshapes
            features['smile'] = blendshape_dict.get('smile', 0)
            features['mouth_open'] = blendshape_dict.get('jawOpen', 0)
            features['brow_down'] = blendshape_dict.get('browOuterDownLeft', 0) + blendshape_dict.get('browOuterDownRight', 0)
            features['eye_blink'] = blendshape_dict.get('eyeBlinkLeft', 0) + blendshape_dict.get('eyeBlinkRight', 0)
            features['cheek_raise'] = blendshape_dict.get('cheekPuff', 0) + blendshape_dict.get('cheekSquintLeft', 0) + blendshape_dict.get('cheekSquintRight', 0)
            features['mouth_frown'] = blendshape_dict.get('mouthFrownLeft', 0) + blendshape_dict.get('mouthFrownRight', 0)

            # 如果有 blendshapes，直接返回（不依赖绝对距离）
            return features

        # ===== 备用方案：绝对像素距离（不推荐） =====
        h, frame_w = frame_shape[:2]

        def get_point(idx):
            if idx < len(landmarks):
                return (int(landmarks[idx].x * frame_w),
                        int(landmarks[idx].y * h))
            return None

        # 眼睛开合度
        left_eye_top = get_point(159)
        left_eye_bottom = get_point(145)
        right_eye_top = get_point(386)
        right_eye_bottom = get_point(374)

        left_eye_openness = 0
        right_eye_openness = 0
        if left_eye_top and left_eye_bottom:
            left_eye_openness = abs(left_eye_top[1] - left_eye_bottom[1])
        if right_eye_top and right_eye_bottom:
            right_eye_openness = abs(right_eye_top[1] - right_eye_bottom[1])

        features['eye_openness'] = (left_eye_openness + right_eye_openness) / 2

        # 嘴巴宽度和高度
        mouth_left = get_point(61)
        mouth_right = get_point(291)
        mouth_top = get_point(13)
        mouth_bottom = get_point(14)

        if mouth_left and mouth_right:
            features['mouth_width'] = abs(mouth_right[0] - mouth_left[0])
        else:
            features['mouth_width'] = 0

        if mouth_top and mouth_bottom:
            features['mouth_height'] = abs(mouth_bottom[1] - mouth_top[1])
        else:
            features['mouth_height'] = 0

        # 眉毛倾斜度
        left_eyebrow_inner = get_point(107)
        left_eyebrow_outer = get_point(70)
        right_eyebrow_inner = get_point(336)
        right_eyebrow_outer = get_point(296)

        if left_eyebrow_inner and left_eyebrow_outer:
            features['left_eyebrow_angle'] = (left_eyebrow_outer[1] - left_eyebrow_inner[1]) / \
                                 (left_eyebrow_outer[0] - left_eyebrow_inner[0] + 1e-6)
        else:
            features['left_eyebrow_angle'] = 0

        if right_eyebrow_inner and right_eyebrow_outer:
            features['right_eyebrow_angle'] = (right_eyebrow_outer[1] - right_eyebrow_inner[1]) / \
                                   (right_eyebrow_outer[0] - right_eyebrow_inner[0] + 1e-6)
        else:
            features['right_eyebrow_angle'] = 0

        # 标记没有 blendshapes
        features['blendshapes'] = None

        return features

    def draw_face_mesh(self, frame, landmarks, face_roi):
        """绘制人脸网格和ROI"""
        output = frame.copy()

        # 绘制人脸框
        x, y, w, h = face_roi
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制额头关键点
        if landmarks:
            h_frame, w_frame = frame.shape[:2]

            for idx in self.FOREHEAD_POINTS[:5]:
                if idx < len(landmarks):
                    px = int(landmarks[idx].x * w_frame)
                    py = int(landmarks[idx].y * h_frame)
                    cv2.circle(output, (px, py), 3, (0, 255, 255), -1)

        return output

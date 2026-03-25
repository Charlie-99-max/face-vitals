import numpy as np
from collections import deque


class EmotionRecognizer:
    """
    基于生理信号和面部表情的情绪识别器
    使用多维度融合方法
    """

    def __init__(self):
        self.emotions = ['Calm', 'Happy', 'Anxious', 'Stressed', 'Sad']
        self.emotion_colors = {
            'Calm': '#4ADE80',
            'Happy': '#FBBF24',
            'Anxious': '#F97316',
            'Stressed': '#E94560',
            'Sad': '#3B82F6',
        }

        self.feature_history = deque(maxlen=60)
        self.emotion_history = deque(maxlen=20)

        self.current_emotion = None
        self.confidence = 0

        self.hr_history = deque(maxlen=30)
        self.hrv_history = deque(maxlen=30)

        # 情绪特征参考值（基于生理研究）
        # 心率和HRV与情绪的关系
        self.hr_baseline = 75  # 默认静息心率
        self.hrv_baseline = 40  # 默认HRV

    def reset(self):
        self.feature_history.clear()
        self.emotion_history.clear()
        self.hr_history.clear()
        self.hrv_history.clear()
        self.current_emotion = None
        self.confidence = 0

    def recognize(self, expression_features, hrv_status=None, heart_rate=None):
        """识别情绪 - 多维度融合，优先使用 Blendshapes"""
        if expression_features is None:
            return None, 0

        # 更新心率历史
        if heart_rate:
            self.hr_history.append(heart_rate)

        # 更新HRV历史
        if hrv_status:
            hrv_map = {'high': 60, 'normal': 40, 'low': 20}
            self.hrv_history.append(hrv_map.get(hrv_status, 40))

        # ===== 优先使用 Blendshapes（标准化 0-1 值） =====
        blendshapes = expression_features.get('blendshapes')
        if blendshapes:
            # 使用 blendshapes 进行情绪识别
            return self._recognize_with_blendshapes(
                expression_features, hrv_status, heart_rate
            )

        # ===== 备用方案：绝对像素距离 =====
        eye_openness = expression_features.get('eye_openness', 30)
        mouth_width = expression_features.get('mouth_width', 50)
        mouth_height = expression_features.get('mouth_height', 10)
        eyebrow_angle = expression_features.get('left_eyebrow_angle', 0)

        self.feature_history.append({
            'eye': eye_openness,
            'mouth_w': mouth_width,
            'mouth_h': mouth_height,
            'eyebrow': eyebrow_angle,
        })

        if len(self.feature_history) < 20:
            return None, 0

        # 计算特征统计
        avg_eye = np.mean([f['eye'] for f in self.feature_history])
        avg_mouth_w = np.mean([f['mouth_w'] for f in self.feature_history])
        avg_mouth_h = np.mean([f['mouth_h'] for f in self.feature_history])
        avg_eyebrow = np.mean([f['eyebrow'] for f in self.feature_history])

        # 眼睛和嘴巴的变化趋势
        eye_trend = self._compute_trend([f['eye'] for f in self.feature_history])
        mouth_trend = self._compute_trend([f['mouth_w'] for f in self.feature_history])

        # 心率趋势
        hr_trend = 0
        if len(self.hr_history) >= 5:
            hr_trend = self._compute_trend(list(self.hr_history))

        # 多维度情绪分类
        scores = self._classify_emotion_multidimensional(
            avg_eye, avg_mouth_w, avg_mouth_h, avg_eyebrow,
            eye_trend, mouth_trend, hr_trend,
            hrv_status, heart_rate
        )

        # 选择最高分的情绪
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]

        # 平滑
        self.emotion_history.append(emotion)

        if len(self.emotion_history) >= 10:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
            count = emotion_counts[smoothed_emotion]
            self.current_emotion = smoothed_emotion
            self.confidence = min(1.0, count / len(self.emotion_history))
        else:
            self.current_emotion = emotion
            self.confidence = confidence * 0.6

        return self.current_emotion, self.confidence

    def _recognize_with_blendshapes(self, expression_features, hrv_status, heart_rate):
        """使用 Blendshapes 进行情绪识别 - 标准化值更稳定"""
        # 提取 blendshape 特征
        smile = expression_features.get('smile', 0)
        mouth_open = expression_features.get('mouth_open', 0)
        brow_down = expression_features.get('brow_down', 0)
        eye_blink = expression_features.get('eye_blink', 0)
        cheek_raise = expression_features.get('cheek_raise', 0)
        mouth_frown = expression_features.get('mouth_frown', 0)

        # 记录历史
        self.feature_history.append({
            'smile': smile,
            'mouth_open': mouth_open,
            'brow_down': brow_down,
            'eye_blink': eye_blink,
            'cheek_raise': cheek_raise,
            'mouth_frown': mouth_frown,
        })

        if len(self.feature_history) < 20:
            return None, 0

        # 计算滑动平均
        avg_smile = np.mean([f['smile'] for f in self.feature_history])
        avg_mouth_open = np.mean([f['mouth_open'] for f in self.feature_history])
        avg_brow_down = np.mean([f['brow_down'] for f in self.feature_history])
        avg_cheek_raise = np.mean([f['cheek_raise'] for f in self.feature_history])
        avg_mouth_frown = np.mean([f['mouth_frown'] for f in self.feature_history])

        # 趋势计算
        smile_trend = self._compute_trend([f['smile'] for f in self.feature_history])
        brow_trend = self._compute_trend([f['brow_down'] for f in self.feature_history])

        # 心率趋势
        hr_trend = 0
        if len(self.hr_history) >= 5:
            hr_trend = self._compute_trend(list(self.hr_history))

        # ===== 基于 Blendshapes 的情绪分类 =====
        scores = {e: 0.1 for e in self.emotions}

        # === 生理信号权重最高 ===
        # HRV分析
        if hrv_status == 'high':
            scores['Calm'] += 0.6
            scores['Happy'] += 0.2
        elif hrv_status == 'low':
            scores['Stressed'] += 0.5
            scores['Anxious'] += 0.4
        else:
            scores['Calm'] += 0.3
            scores['Happy'] += 0.2

        # 心率分析
        if heart_rate:
            hr_change = heart_rate - self.hr_baseline
            if hr_change > 20:
                if hrv_status == 'high':
                    scores['Happy'] += 0.4
                else:
                    scores['Anxious'] += 0.4
                    scores['Stressed'] += 0.3
            elif hr_change < -15:
                scores['Calm'] += 0.4
                if hrv_status == 'low':
                    scores['Sad'] += 0.3

            if hr_trend > 2:
                if hrv_status == 'high':
                    scores['Happy'] += 0.2
                else:
                    scores['Anxious'] += 0.2
            elif hr_trend < -2:
                scores['Calm'] += 0.2

        # === Blendshape 表情分析 ===
        # 微笑（最重要的情绪指标）
        if avg_smile > 0.5:
            scores['Happy'] += 0.8
        elif avg_smile > 0.3:
            scores['Happy'] += 0.4
        elif avg_smile > 0.1:
            scores['Happy'] += 0.2

        # 笑容趋势
        if smile_trend > 0.01:
            scores['Happy'] += 0.3

        # 脸颊抬起（辅助微笑检测）
        if avg_cheek_raise > 0.3:
            scores['Happy'] += 0.3

        # 皱眉（压力/焦虑/悲伤）
        if avg_brow_down > 0.5:
            scores['Stressed'] += 0.5
            scores['Anxious'] += 0.3
        elif avg_brow_down > 0.3:
            scores['Anxious'] += 0.3
            scores['Sad'] += 0.2

        # 眉毛趋势
        if brow_trend > 0.01:
            scores['Anxious'] += 0.2

        # 嘴角下垂（悲伤/沮丧）
        if avg_mouth_frown > 0.3:
            scores['Sad'] += 0.5
            scores['Anxious'] += 0.2

        # 嘴巴张开（惊讶/焦虑）
        if avg_mouth_open > 0.4:
            scores['Anxious'] += 0.3
        elif avg_mouth_open > 0.2:
            scores['Happy'] += 0.1

        # 归一化
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}

        # 选择最高分的情绪
        emotion = max(scores, key=scores.get)
        confidence = scores[emotion]

        # 平滑
        self.emotion_history.append(emotion)

        if len(self.emotion_history) >= 10:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            smoothed_emotion = max(emotion_counts, key=emotion_counts.get)
            count = emotion_counts[smoothed_emotion]
            self.current_emotion = smoothed_emotion
            self.confidence = min(1.0, count / len(self.emotion_history))
        else:
            self.current_emotion = emotion
            self.confidence = confidence * 0.6

        return self.current_emotion, self.confidence

    def _compute_trend(self, data):
        """计算数据趋势"""
        if len(data) < 5:
            return 0
        x = np.arange(len(data))
        try:
            slope, _ = np.polyfit(x, data, 1)
            return slope
        except:
            return 0

    def _classify_emotion_multidimensional(self, eye, mouth_w, mouth_h, eyebrow,
                                           eye_trend, mouth_trend, hr_trend,
                                           hrv_status, heart_rate):
        """多维度情绪分类"""
        scores = {e: 0.1 for e in self.emotions}

        mouth_ratio = mouth_h / (mouth_w + 1e-6)

        # === 生理信号权重最高 ===
        # 1. HRV分析
        if hrv_status == 'high':
            scores['Calm'] += 0.6
            scores['Happy'] += 0.2
        elif hrv_status == 'low':
            scores['Stressed'] += 0.5
            scores['Anxious'] += 0.4
        else:  # normal
            scores['Calm'] += 0.3
            scores['Happy'] += 0.2

        # 2. 心率分析
        if heart_rate:
            # 心率相对于基线的变化
            hr_change = heart_rate - self.hr_baseline

            if hr_change > 20:  # 心率显著升高
                if hrv_status == 'high':
                    scores['Happy'] += 0.4  # 兴奋
                else:
                    scores['Anxious'] += 0.4
                    scores['Stressed'] += 0.3
            elif hr_change < -15:  # 心率显著降低
                scores['Calm'] += 0.4
                if hrv_status == 'low':
                    scores['Sad'] += 0.3

            # 心率趋势
            if hr_trend > 2:  # 心率上升
                if hrv_status == 'high':
                    scores['Happy'] += 0.2
                else:
                    scores['Anxious'] += 0.2
            elif hr_trend < -2:  # 心率下降
                scores['Calm'] += 0.2

        # === 面部表情 ===
        # 眼睛
        if eye > 35:
            scores['Happy'] += 0.15
            scores['Anxious'] += 0.1
        elif eye < 18:
            scores['Sad'] += 0.25
            scores['Calm'] += 0.1

        # 眼睛趋势
        if eye_trend < -1:  # 眼睛逐渐变小
            scores['Sad'] += 0.15
            scores['Calm'] += 0.1
        elif eye_trend > 1:  # 眼睛逐渐变大
            scores['Happy'] += 0.15

        # 嘴巴 - 微笑检测
        if mouth_ratio < 0.18 and mouth_w > 40:
            scores['Happy'] += 0.4  # 明显微笑
        elif mouth_ratio > 0.35:
            scores['Anxious'] += 0.25  # 嘴巴张开

        # 嘴巴趋势
        if mouth_trend > 2:
            scores['Happy'] += 0.1

        # 眉毛
        if eyebrow < -0.4:
            scores['Anxious'] += 0.2
            scores['Sad'] += 0.15
        elif eyebrow > 0.3:
            scores['Stressed'] += 0.3
            scores['Anxious'] += 0.15

        # 归一化
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}

        return scores

    def update_baseline(self, heart_rate, hrv_value):
        """更新生理基线"""
        if heart_rate:
            # 使用指数移动平均更新
            self.hr_baseline = 0.9 * self.hr_baseline + 0.1 * heart_rate
        if hrv_value:
            self.hrv_baseline = 0.9 * self.hrv_baseline + 0.1 * hrv_value

    def get_current_emotion(self):
        return self.current_emotion

    def get_confidence(self):
        return self.confidence

    def get_emotion_color(self, emotion):
        return self.emotion_colors.get(emotion, '#94A3B8')

    def get_emotion_description(self):
        descriptions = {
            'Calm': '放松平静',
            'Happy': '愉悦开心',
            'Anxious': '焦虑紧张',
            'Stressed': '压力山大',
            'Sad': '低落悲伤',
        }
        if self.current_emotion:
            return descriptions.get(self.current_emotion, '未知')
        return '等待分析...'

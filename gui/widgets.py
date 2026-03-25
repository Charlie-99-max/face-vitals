from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QFont, QPainterPath
import numpy as np


class MetricCard(QWidget):
    """指标卡片组件"""

    def __init__(self, title, unit="", accent_color="#E94560", parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.accent_color = accent_color
        self.value = None
        self.status = None
        self.subtitle = ""

        self.setFixedHeight(180)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #1E1E2E;
                border-radius: 12px;
            }}
        """)

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(8)

        # 标题
        self.title_label = QLabel(self.title)
        self.title_label.setStyleSheet("color: #94A3B8; font-size: 14px;")
        layout.addWidget(self.title_label)

        # 主值
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("""
            color: #FFFFFF;
            font-size: 48px;
            font-weight: bold;
            font-family: SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif;
        """)
        layout.addWidget(self.value_label)

        # 单位
        self.unit_label = QLabel(self.unit)
        self.unit_label.setStyleSheet("color: #94A3B8; font-size: 16px;")
        layout.addWidget(self.unit_label)

        # 状态/副标题
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #4ADE80; font-size: 12px;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def set_value(self, value):
        """设置值"""
        if value is not None:
            self.value = value
            self.value_label.setText(f"{value:.1f}" if isinstance(value, float) else str(value))
        else:
            self.value_label.setText("--")

    def set_status(self, status, color=None):
        """设置状态"""
        self.status = status
        if color:
            self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")
        self.status_label.setText(status)

    def set_subtitle(self, text):
        """设置副标题"""
        self.subtitle = text
        self.status_label.setText(text)


class SignalWidget(QWidget):
    """信号波形显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.max_points = 200
        self.line_color = QColor("#4ADE80")
        self.background_color = QColor("#1E1E2E")

        self.setMinimumHeight(100)
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E2E;
                border-radius: 8px;
            }
        """)

    def set_data(self, data):
        """设置数据"""
        if len(data) > self.max_points:
            self.data = data[-self.max_points:]
        else:
            self.data = data
        self.update()

    def set_color(self, color_hex):
        """设置线条颜色"""
        self.line_color = QColor(color_hex)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), self.background_color)

        if len(self.data) < 2:
            return

        # 绘制波形
        painter.setPen(QPen(self.line_color, 2))

        w = self.width()
        h = self.height()

        # 归一化数据
        data_min = min(self.data)
        data_max = max(self.data)
        data_range = data_max - data_min if data_max != data_min else 1

        points = []
        for i, val in enumerate(self.data):
            x = int(i / len(self.data) * w)
            y = int((1 - (val - data_min) / data_range) * h * 0.8 + h * 0.1)
            points.append((x, y))

        # 绘制连线
        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1],
                           points[i+1][0], points[i+1][1])


class PulsingLabel(QLabel):
    """脉冲动画标签"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 16px;
                padding: 10px;
            }
        """)

    def start_pulse(self):
        """开始脉冲动画"""
        self.setStyleSheet("""
            QLabel {
                color: #E94560;
                font-size: 16px;
                padding: 10px;
                animation: pulse 1s infinite;
            }
        """)

    def stop_pulse(self):
        """停止脉冲动画"""
        self.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 16px;
                padding: 10px;
            }
        """)


class EmotionDisplay(QWidget):
    """情绪显示组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.emotion = None
        self.confidence = 0

        self.setMinimumHeight(180)
        self.setStyleSheet("""
            QWidget {
                background-color: #1E1E2E;
                border-radius: 12px;
            }
        """)

    def set_emotion(self, emotion, confidence=0):
        """设置情绪"""
        self.emotion = emotion
        self.confidence = confidence
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor("#1E1E2E"))

        if not self.emotion:
            # 显示等待文字
            painter.setPen(QColor("#94A3B8"))
            font = QFont()
            font.setPixelSize(16)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "等待分析...")
            return

        # 绘制情绪图标和文字
        center_x = self.width() // 2
        center_y = self.height() // 2

        # 情绪颜色
        colors = {
            'Calm': QColor("#4ADE80"),
            'Happy': QColor("#FBBF24"),
            'Anxious': QColor("#F97316"),
            'Stressed': QColor("#E94560"),
            'Sad': QColor("#3B82F6"),
        }
        emotion_color = colors.get(self.emotion, QColor("#94A3B8"))

        # 绘制圆圈
        painter.setPen(QPen(emotion_color, 3))
        emotion_color_with_alpha = QColor(emotion_color)
        emotion_color_with_alpha.setAlpha(30)
        painter.setBrush(QBrush(emotion_color_with_alpha))
        radius = 50
        painter.drawEllipse(center_x - radius, center_y - radius - 20,
                          radius * 2, radius * 2)

        # 绘制表情
        painter.setPen(QPen(emotion_color, 3))
        self._draw_emotion_face(painter, center_x, center_y - 20, self.emotion)

        # 绘制文字
        painter.setPen(QColor("#FFFFFF"))
        font = QFont()
        font.setPixelSize(20)
        font.setBold(True)
        painter.setFont(font)
        emotion_text = {
            'Calm': '平静',
            'Happy': '开心',
            'Anxious': '焦虑',
            'Stressed': '压力',
            'Sad': '悲伤',
        }
        text = emotion_text.get(self.emotion, self.emotion)
        painter.drawText(self.rect().adjusted(0, 80, 0, 0),
                        Qt.AlignmentFlag.AlignCenter, text)

        # 置信度
        painter.setPen(QColor("#94A3B8"))
        font.setPixelSize(12)
        font.setBold(False)
        painter.setFont(font)
        conf_text = f"置信度: {self.confidence*100:.0f}%"
        painter.drawText(self.rect().adjusted(0, 110, 0, 0),
                        Qt.AlignmentFlag.AlignCenter, conf_text)

    def _draw_emotion_face(self, painter, cx, cy, emotion):
        """绘制简化的表情"""
        y_offset = cy

        if emotion == 'Calm':
            # 平静 - 微笑
            painter.drawArc(cx - 20, y_offset - 10, 40, 20, 0, 180 * 16)
        elif emotion == 'Happy':
            # 开心 - 大笑
            painter.drawArc(cx - 25, y_offset - 15, 50, 30, 0, 180 * 16)
        elif emotion == 'Anxious':
            # 焦虑 - 紧张
            painter.drawLine(cx - 15, y_offset, cx + 15, y_offset)
        elif emotion == 'Stressed':
            # 压力 - 皱眉
            painter.drawArc(cx - 20, y_offset, 40, 15, 180 * 16, 180 * 16)
        elif emotion == 'Sad':
            # 悲伤 - 嘴角下弯
            painter.drawArc(cx - 20, y_offset, 40, 20, 0, 180 * 16)


class CameraComboBox:
    """摄像头选择器 (使用QComboBox)"""
    pass

import cv2
import numpy as np
import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QApplication
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QFont

# 导入组件
from widgets import MetricCard, SignalWidget, EmotionDisplay


class VideoPanel(QFrame):
    """视频面板组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #1E1E2E;
                border-radius: 12px;
                border: 2px solid #2A2A3E;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 视频标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("color: #94A3B8; font-size: 14px;")
        self.video_label.setText("请选择摄像头并开始监测")
        layout.addWidget(self.video_label)

        # 信号质量条
        self.quality_layout = QHBoxLayout()
        self.quality_label = QLabel("信号质量:")
        self.quality_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        self.quality_layout.addWidget(self.quality_label)

        self.quality_bar = QLabel()
        self.quality_bar.setFixedSize(100, 10)
        self.quality_bar.setStyleSheet("""
            background-color: #2A2A3E;
            border-radius: 5px;
        """)
        self.quality_layout.addWidget(self.quality_bar)
        self.quality_layout.addStretch()

        self.quality_value = QLabel("0%")
        self.quality_value.setStyleSheet("color: #4ADE80; font-size: 12px;")
        self.quality_layout.addWidget(self.quality_value)

        layout.addLayout(self.quality_layout)

    def set_frame(self, frame):
        """设置视频帧"""
        if frame is None:
            return

        # 转换颜色
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        # 创建QImage
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # 缩放以适应标签
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled_pixmap)

    def set_quality(self, quality):
        """设置信号质量"""
        if quality is None:
            quality = 0

        # 颜色根据质量变化
        if quality > 70:
            color = "#4ADE80"  # 绿色
        elif quality > 40:
            color = "#FBBF24"  # 黄色
        else:
            color = "#E94560"  # 红色

        self.quality_value.setText(f"{int(quality)}%")
        self.quality_value.setStyleSheet(f"color: {color}; font-size: 12px;")


class MainWindow(QWidget):
    """主窗口"""

    def __init__(self):
        super().__init__()

        # 核心组件
        self.video_capture = None
        self.timer = None

        # 状态
        self.is_running = False
        self.current_camera = 0

        # 初始化UI
        self.setWindowTitle("FaceVitals - 心率 HRV 心情预测")
        self.setMinimumSize(1024, 768)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self._bg_color()};
                color: #FFFFFF;
            }}
            QPushButton {{
                background-color: #E94560;
                color: #FFFFFF;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #F25B74;
            }}
            QPushButton:pressed {{
                background-color: #D13B54;
            }}
            QPushButton:disabled {{
                background-color: #4A4A5A;
                color: #8A8A9A;
            }}
            QComboBox {{
                background-color: #2A2A3E;
                color: #FFFFFF;
                border: 1px solid #3A3A4E;
                padding: 8px 12px;
                border-radius: 6px;
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox::down-arrow {{
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #94A3B8;
            }}
        """)

        self._init_ui()

    def _bg_color(self):
        return "#0F0F1A"

    def _init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ===== 标题栏 =====
        header_layout = QHBoxLayout()

        title = QLabel("FaceVitals")
        title.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            font-family: SF Pro Display, -apple-system, BlinkMacSystemFont, sans-serif;
            color: #FFFFFF;
        """)
        header_layout.addWidget(title)

        header_layout.addStretch()

        # 摄像头选择
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedWidth(150)
        self.camera_combo.addItem("摄像头 0", 0)
        self._populate_cameras()
        header_layout.addWidget(QLabel("选择摄像头:"))
        header_layout.addWidget(self.camera_combo)

        # 开始/停止按钮
        self.start_button = QPushButton("开始监测")
        self.start_button.clicked.connect(self.toggle_monitoring)
        header_layout.addWidget(self.start_button)

        main_layout.addLayout(header_layout)

        # ===== 主内容区 =====
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # 左侧 - 视频面板
        self.video_panel = VideoPanel()
        self.video_panel.setMinimumWidth(500)
        content_layout.addWidget(self.video_panel)

        # 右侧 - 指标面板
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)

        # 心率卡片
        self.hr_card = MetricCard("心率", "BPM", "#E94560")
        self.hr_card.set_value(None)
        right_panel.addWidget(self.hr_card)

        # 心率波形
        self.hr_graph = SignalWidget()
        self.hr_graph.setMinimumHeight(80)
        right_panel.addWidget(self.hr_graph)

        # HRV卡片
        self.hrv_card = MetricCard("心率变异性 (RMSSD)", "ms", "#4ADE80")
        self.hrv_card.set_value(None)
        right_panel.addWidget(self.hrv_card)

        # 心情卡片
        self.emotion_display = EmotionDisplay()
        right_panel.addWidget(self.emotion_display)

        content_layout.addLayout(right_panel)

        main_layout.addLayout(content_layout)

        # ===== 状态栏 =====
        status_layout = QHBoxLayout()

        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        status_layout.addWidget(self.status_label)

        status_layout.addStretch()

        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        status_layout.addWidget(self.fps_label)

        main_layout.addLayout(status_layout)

        # 定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.last_fps_update = 0

    def _populate_cameras(self):
        """检测可用摄像头"""
        self.camera_combo.clear()

        # 尝试检测多个摄像头
        available_cameras = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        if not available_cameras:
            available_cameras = [0]

        for cam in available_cameras:
            self.camera_combo.addItem(f"摄像头 {cam}", cam)

    def toggle_monitoring(self):
        """切换监测状态"""
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        """开始监测"""
        camera_index = self.camera_combo.currentData()
        self.video_capture = cv2.VideoCapture(camera_index)

        if not self.video_capture.isOpened():
            self.status_label.setText("无法打开摄像头")
            return

        # 设置分辨率
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        self.start_button.setText("停止监测")
        self.status_label.setText("正在监测...")
        self.timer.start(33)  # ~30 FPS

    def stop_monitoring(self):
        """停止监测"""
        self.is_running = False
        self.start_button.setText("开始监测")
        self.status_label.setText("已停止")
        self.timer.stop()

        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # 重置显示
        self.hr_card.set_value(None)
        self.hr_card.set_status("")
        self.hrv_card.set_value(None)
        self.hrv_card.set_status("")
        self.emotion_display.set_emotion(None, 0)
        self.video_panel.set_quality(0)
        self.hr_graph.set_data([])

    def update_frame(self):
        """更新帧"""
        if not self.is_running or not self.video_capture:
            return

        ret, frame = self.video_capture.read()
        if not ret:
            return

        # 显示原始帧
        display_frame = frame.copy()
        self.video_panel.set_frame(display_frame)

        # FPS计算
        self.frame_count += 1
        import time
        current_time = time.time()
        if current_time - self.last_fps_update >= 1:
            fps = self.frame_count / (current_time - self.last_fps_update)
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_fps_update = current_time

    def closeEvent(self, event):
        """关闭事件"""
        self.stop_monitoring()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

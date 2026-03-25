#!/usr/bin/env python3
"""
FaceVitals - 心率 HRV 心情预测系统

该程序通过摄像头实时检测心率、HRV和情绪状态。
基于rPPG（远程光电容积描记）技术实现无接触式生理指标测量。
"""

# ============================================================
# 导入必要的库
# ============================================================

import cv2                      # OpenCV，用于摄像头读取和图像处理
import sys                      # 系统相关操作
import time                     # 时间相关操作，用于FPS计算

# PyQt6 相关组件
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,  # 基础UI组件
    QComboBox, QFrame, QApplication, QMessageBox, QDialog,   # 对话框和选择框
    QLineEdit, QDialogButtonBox, QFormLayout,                # 表单输入组件
    QListWidget, QAbstractItemView                           # 列表组件
)
from PyQt6.QtCore import Qt, QTimer, QSize     # Qt常量和定时器
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor  # 图形绘制

# 核心算法模块
from core.face_detector import FaceDetector           # 人脸检测模块
from core.advanced_rppg import AdvancedRPPGProcessor  # rPPG信号处理模块
from core.hrv_analyzer import HRVAnalyzer            # HRV心率变异性分析模块
from core.emotion_recognizer import EmotionRecognizer # 情绪识别模块


# ============================================================
# 全局配置
# ============================================================

# 默认摄像头配置列表
# source: 摄像头索引（0=系统默认摄像头，1=第二个摄像头，以此类推）
# type: 'local'表示本地摄像头，'url'表示网络摄像头
DEFAULT_CAMERAS = [
    {'name': '💻 电脑摄像头', 'source': '0', 'type': 'local'},
    {'name': '📱 手机后置', 'source': '1', 'type': 'local'},
    {'name': '📱 手机前置', 'source': '2', 'type': 'local'},
]


# ============================================================
# UI组件类
# ============================================================

class VideoPanel(QFrame):
    """
    视频显示面板组件

    功能：
    - 显示摄像头捕获的视频画面
    - 支持水平翻转（适配前置摄像头）
    - iOS风格的黑色圆角设计
    """

    def __init__(self, parent=None):
        """初始化视频面板"""
        super().__init__(parent)

        # 设置面板样式：黑色背景，圆角边框
        self.setStyleSheet("""
            QFrame {
                background-color: #000000;
                border-radius: 20px;
            }
        """)

        # 创建垂直布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)  # 无边距

        # 创建视频显示标签
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐
        self.video_label.setMinimumSize(640, 480)  # 最小尺寸
        self.video_label.setStyleSheet("color: #8E8E93; font-size: 17px; border: none; background: transparent;")
        self.video_label.setText("等待启动监测...")  # 初始提示文字
        layout.addWidget(self.video_label)

    def set_frame(self, frame, flip=True):
        """
        更新视频画面

        参数:
            frame: OpenCV读取的BGR格式图像
            flip: 是否水平翻转（默认True，适配前置摄像头）
        """
        if frame is None:
            return

        # 水平翻转（镜像效果）
        if flip:
            frame = cv2.flip(frame, 1)

        # BGR转RGB（OpenCV使用BGR，Qt使用RGB）
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        # 创建Qt图像并转换为Pixmap显示
        qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        scaled = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,  # 保持宽高比
            Qt.TransformationMode.SmoothTransformation  # 平滑缩放
        )
        self.video_label.setPixmap(scaled)


class MetricCard(QFrame):
    """
    指标显示卡片组件

    功能：
    - 显示单个指标（如心率、HRV）的数值
    - 包含图标、标题、数值和单位
    - 固定高度防止布局抖动
    """

    def __init__(self, icon, title, unit, color="#FF3B30", parent=None):
        """
        初始化指标卡片

        参数:
            icon: emoji图标
            title: 指标名称
            unit: 单位
            color: 数值颜色（十六进制）
        """
        super().__init__(parent)

        # 白色背景，圆角16px
        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 16px;
            }
        """)

        # 固定高度140px，防止数值变化时布局抖动
        self.setMinimumHeight(140)
        self.setMaximumHeight(140)

        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)  # 内边距
        layout.setSpacing(2)  # 元素间距

        # 顶部区域：图标 + 标题
        top_layout = QHBoxLayout()
        top_layout.setSpacing(8)

        icon_label = QLabel(icon)
        icon_label.setStyleSheet("font-size: 18px;")
        top_layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #8E8E93; font-size: 13px; font-weight: 500;")
        top_layout.addWidget(title_label)
        top_layout.addStretch()  # 右侧留白
        layout.addLayout(top_layout)

        # 数值显示标签
        self.value_label = QLabel("--")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.value_label.setMinimumSize(QSize(120, 56))  # 固定最小尺寸防止抖动
        self.value_label.setStyleSheet(f"""
            color: {color};
            font-size: 48px;
            font-weight: 700;
            letter-spacing: -2px;
            font-family: 'SF Pro Display', -apple-system, sans-serif;
        """)
        layout.addWidget(self.value_label)

        # 单位标签
        unit_label = QLabel(unit)
        unit_label.setStyleSheet("color: #8E8E93; font-size: 13px; font-weight: 500;")
        layout.addWidget(unit_label)

    def set_value(self, value):
        """更新显示的数值"""
        self.value_label.setText(str(value))


class SignalGraph(QFrame):
    """
    心率波形显示组件

    功能：
    - 实时绘制心率波形曲线
    - 红色iOS风格线条
    - 固定高度防止抖动
    """

    def __init__(self, parent=None):
        """初始化波形图"""
        super().__init__(parent)
        self.data = []  # 存储波形数据点

        # 白色背景，圆角16px
        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 16px;
            }
        """)

        # 固定高度90px
        self.setMinimumHeight(90)
        self.setMaximumHeight(90)

    def set_data(self, data):
        """
        更新波形数据

        参数:
            data: 数值列表，最多保留100个点
        """
        self.data = data[-100:] if len(data) > 100 else data
        self.update()  # 触发重绘

    def paintEvent(self, event):
        """
        绘制波形曲线（自定义绘制）

        使用QPainter绘制平滑的红色曲线
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)  # 抗锯齿

        # 白色背景
        painter.fillRect(self.rect(), QColor(255, 255, 255))

        if len(self.data) < 2:
            return  # 数据点不足则不绘制

        # 设置画笔：红色，2.5px宽度，圆角端点
        pen = QPen(QColor(255, 59, 48), 2.5)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        w = self.width()
        h = self.height()
        step = w / max(len(self.data) - 1, 1)  # 每个数据点的水平间距

        # 计算数值范围用于归一化
        min_val = min(self.data)
        max_val = max(self.data)
        range_val = max_val - min_val if max_val != min_val else 1

        # 计算所有点的坐标
        path_points = []
        for i, val in enumerate(self.data):
            x = i * step
            # Y坐标反转（屏幕坐标系Y向下）
            y = h - ((val - min_val) / range_val * (h - 20)) - 10
            path_points.append((x, y))

        # 绘制线条
        for i in range(len(path_points) - 1):
            painter.drawLine(
                int(path_points[i][0]), int(path_points[i][1]),
                int(path_points[i + 1][0]), int(path_points[i + 1][1])
            )


class EmotionCard(QFrame):
    """
    情绪状态显示卡片

    功能：
    - 显示检测到的情绪状态
    - 显示情绪识别置信度
    - 固定高度防止抖动
    """

    def __init__(self, parent=None):
        """初始化情绪卡片"""
        super().__init__(parent)

        self.setStyleSheet("""
            QFrame {
                background-color: #FFFFFF;
                border-radius: 16px;
            }
        """)

        # 固定高度100px
        self.setMinimumHeight(100)
        self.setMaximumHeight(100)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 14, 20, 14)
        layout.setSpacing(4)

        # 顶部区域
        top_layout = QHBoxLayout()
        icon_label = QLabel("🧠")
        icon_label.setStyleSheet("font-size: 18px;")
        top_layout.addWidget(icon_label)

        title = QLabel("情绪状态")
        title.setStyleSheet("color: #8E8E93; font-size: 13px; font-weight: 500;")
        top_layout.addWidget(title)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # 情绪名称
        self.emotion_label = QLabel("--")
        self.emotion_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.emotion_label.setMinimumSize(QSize(100, 32))
        self.emotion_label.setStyleSheet("color: #000000; font-size: 26px; font-weight: 600;")
        layout.addWidget(self.emotion_label)

        # 置信度
        self.confidence_label = QLabel("")
        self.confidence_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        layout.addWidget(self.confidence_label)

    def set_emotion(self, emotion, confidence):
        """
        更新情绪显示

        参数:
            emotion: 情绪名称
            confidence: 置信度（0-1）
        """
        self.emotion_label.setText(emotion)
        if confidence:
            self.confidence_label.setText(f"置信度: {int(confidence * 100)}%")


# ============================================================
# 摄像头管理对话框
# ============================================================

class CameraManagerDialog(QDialog):
    """
    摄像头管理对话框

    功能：
    - 添加新摄像头配置
    - 编辑现有摄像头配置
    - 删除摄像头配置
    """

    def __init__(self, cameras, parent=None):
        """
        初始化对话框

        参数:
            cameras: 摄像头配置列表
        """
        super().__init__(parent)
        self.cameras = cameras
        self.setWindowTitle("摄像头设置")
        self.setFixedSize(420, 380)

        # iOS灰色风格样式
        self.setStyleSheet("""
            QDialog { background-color: #F2F2F7; }
            QLabel { color: #000000; }
            QLineEdit {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
            }
            QComboBox {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
                padding: 12px;
                font-size: 15px;
            }
        """)

        self._init_ui()

    def _init_ui(self):
        """初始化对话框UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(16)

        # 标题
        title = QLabel("摄像头设置")
        title.setStyleSheet("font-size: 22px; font-weight: 700; color: #000000;")
        layout.addWidget(title)

        # 说明文字
        info_label = QLabel("本地摄像头输入编号(0,1,2)，手机摄像头输入URL")
        info_label.setStyleSheet("color: #8E8E93; font-size: 13px;")
        layout.addWidget(info_label)

        # 摄像头列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.list_widget.setStyleSheet("""
            QListWidget {
                background-color: #FFFFFF;
                border: none;
                border-radius: 12px;
            }
            QListWidget::item {
                padding: 14px;
                border-bottom: 1px solid #E5E5EA;
                font-size: 15px;
                color: #000000;
            }
            QListWidget::item:last { border-bottom: none; }
            QListWidget::item:selected { background-color: #007AFF; color: #FFFFFF; }
        """)
        self._refresh_list()
        layout.addWidget(self.list_widget)

        # 按钮行：添加、编辑、删除、完成
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        for text, method in [("添加", self._add_camera), ("编辑", self._edit_camera), ("删除", self._remove_camera)]:
            btn = QPushButton(text)
            btn.setFixedHeight(40)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E5E5EA;
                    color: #007AFF;
                    border: none;
                    border-radius: 10px;
                    font-size: 15px;
                    font-weight: 600;
                }
                QPushButton:hover { background-color: #D1D1D6; }
            """)
            btn.clicked.connect(method)
            btn_layout.addWidget(btn)

        ok_btn = QPushButton("完成")
        ok_btn.setFixedHeight(40)
        ok_btn.setFixedWidth(80)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #0071E3; }
        """)
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        layout.addLayout(btn_layout)

    def _refresh_list(self):
        """刷新摄像头列表显示"""
        self.list_widget.clear()
        for cam in self.cameras:
            self.list_widget.addItem(f"{cam['name']}  →  {cam['source']}")

    def _add_camera(self):
        """添加新摄像头对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("添加摄像头")
        dialog.setFixedSize(360, 220)

        layout = QFormLayout(dialog)
        layout.setSpacing(12)

        name_input = QLineEdit()
        name_input.setPlaceholderText("例如: 手机前置")
        source_input = QLineEdit()
        source_input.setPlaceholderText("0 或 URL地址")
        type_combo = QComboBox()
        type_combo.addItem("本地", "local")
        type_combo.addItem("URL", "url")

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow("名称:", name_input)
        layout.addRow("源:", source_input)
        layout.addRow("类型:", type_combo)
        layout.addRow(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            name = name_input.text().strip()
            source = source_input.text().strip()
            if name and source:
                self.cameras.append({'name': name, 'source': source, 'type': type_combo.currentData()})
                self._refresh_list()

    def _edit_camera(self):
        """编辑选中的摄像头"""
        row = self.list_widget.currentRow()
        if row < 0:
            QMessageBox.warning(self, "提示", "请先选择一个摄像头")
            return

        cam = self.cameras[row]
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑摄像头")
        dialog.setFixedSize(360, 220)

        layout = QFormLayout(dialog)
        layout.setSpacing(12)

        name_input = QLineEdit(cam['name'])
        source_input = QLineEdit(cam['source'])
        type_combo = QComboBox()
        type_combo.addItem("本地", "local")
        type_combo.addItem("URL", "url")
        type_combo.setCurrentIndex(0 if cam['type'] == 'local' else 1)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow("名称:", name_input)
        layout.addRow("源:", source_input)
        layout.addRow("类型:", type_combo)
        layout.addRow(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.cameras[row] = {
                'name': name_input.text().strip(),
                'source': source_input.text().strip(),
                'type': type_combo.currentData()
            }
            self._refresh_list()

    def _remove_camera(self):
        """删除选中的摄像头"""
        row = self.list_widget.currentRow()
        if row >= 0:
            self.cameras.pop(row)
            self._refresh_list()

    def get_cameras(self):
        """获取更新后的摄像头列表"""
        return self.cameras


# ============================================================
# 主应用类
# ============================================================

class FaceVitalsApp(QWidget):
    """
    FaceVitals 主应用窗口

    功能：
    - 管理摄像头连接和视频流
    - 实时人脸检测和生理指标计算
    - 显示心率、HRV、情绪状态
    - 摄像头管理和切换
    """

    def __init__(self):
        """初始化主应用"""
        super().__init__()

        # 初始化核心算法模块
        self.face_detector = FaceDetector()                      # 人脸检测器
        self.rppg_processor = AdvancedRPPGProcessor(buffer_size=900, fs=30)  # rPPG处理器
        self.hrv_analyzer = HRVAnalyzer(fs=30)                  # HRV分析器
        self.emotion_recognizer = EmotionRecognizer()            # 情绪识别器

        # 状态变量
        self.is_running = False           # 是否正在监测
        self.video_capture = None        # 视频捕获对象
        self.frame_count = 0             # 帧计数（用于FPS计算）
        self.last_fps_time = time.time() # 上次FPS计算时间
        self.camera_list = DEFAULT_CAMERAS.copy()  # 摄像头配置列表

        # 窗口基本设置
        self.setWindowTitle("FaceVitals")
        self.setMinimumSize(1200, 820)

        # 全局样式表 - iOS风格
        self.setStyleSheet("""
            * {
                font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif;
            }
            QWidget {
                background-color: #F2F2F7;
                color: #000000;
            }
        """)

        # 初始化UI并填充摄像头列表
        self._init_ui()
        self._populate_cameras()

    def _init_ui(self):
        """初始化主界面UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 12, 16, 12)
        main_layout.setSpacing(12)

        # ============================================================
        # 顶部导航栏
        # ============================================================
        nav_bar = QFrame()
        nav_bar.setStyleSheet("QFrame { background-color: #F2F2F7; border-radius: 16px; }")
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(20, 14, 20, 14)
        nav_layout.setSpacing(20)

        # 标题
        title = QLabel("FaceVitals")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #000000;")
        nav_layout.addWidget(title)
        nav_layout.addStretch()

        # 摄像头选择标签
        cam_label = QLabel("摄像头:")
        cam_label.setStyleSheet("color: #8E8E93; font-size: 14px;")
        nav_layout.addWidget(cam_label)

        # 摄像头下拉选择框
        self.camera_combo = QComboBox()
        self.camera_combo.setFixedWidth(180)
        self.camera_combo.setStyleSheet("""
            QComboBox {
                background-color: #FFFFFF;
                color: #000000;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
                padding: 10px 14px;
                font-size: 15px;
            }
            QComboBox:hover { border-color: #C7C7CC; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #8E8E93;
            }
            QComboBox QAbstractItemView {
                background-color: #FFFFFF;
                border: 1px solid #E5E5EA;
                border-radius: 10px;
                selection-background-color: #007AFF;
                selection-color: #FFFFFF;
            }
        """)
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        nav_layout.addWidget(self.camera_combo)

        # 设置按钮
        settings_btn = QPushButton("⚙ 设置")
        settings_btn.setFixedHeight(38)
        settings_btn.setStyleSheet("""
            QPushButton { background-color: #FFFFFF; border: none; border-radius: 10px; font-size: 14px; color: #007AFF; }
            QPushButton:hover { background-color: #E5E5EA; }
        """)
        settings_btn.clicked.connect(self._manage_cameras)
        nav_layout.addWidget(settings_btn)

        # 切换按钮
        switch_btn = QPushButton("🔄 切换")
        switch_btn.setFixedHeight(38)
        switch_btn.setStyleSheet("""
            QPushButton { background-color: #FFFFFF; border: none; border-radius: 10px; font-size: 14px; color: #007AFF; }
            QPushButton:hover { background-color: #E5E5EA; }
        """)
        switch_btn.clicked.connect(self._toggle_camera)
        nav_layout.addWidget(switch_btn)

        # 开始/停止按钮
        self.start_btn = QPushButton("▶ 开始")
        self.start_btn.setFixedSize(90, 38)
        self.start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #0071E3; }
        """)
        self.start_btn.clicked.connect(self.toggle_monitoring)
        nav_layout.addWidget(self.start_btn)

        main_layout.addWidget(nav_bar)

        # ============================================================
        # 主内容区：左侧视频 + 右侧指标
        # ============================================================
        content = QHBoxLayout()
        content.setSpacing(12)

        # 左侧视频面板（占3/5宽度）
        self.video_panel = VideoPanel()
        self.video_panel.setMinimumSize(500, 400)
        content.addWidget(self.video_panel, 3)

        # 右侧指标面板（占2/5宽度）
        right_container = QVBoxLayout()
        right_container.setSpacing(10)

        # 心率卡片
        self.hr_card = MetricCard("❤️", "心率", "BPM", "#FF3B30")
        right_container.addWidget(self.hr_card)

        # 心率波形图
        self.hr_graph = SignalGraph()
        right_container.addWidget(self.hr_graph)

        # HRV卡片
        self.hrv_card = MetricCard("💓", "HRV", "RMSSD", "#34C759")
        right_container.addWidget(self.hrv_card)

        # 情绪卡片
        self.emotion_card = EmotionCard()
        right_container.addWidget(self.emotion_card)

        right_container.addStretch()
        content.addLayout(right_container, 2)

        main_layout.addLayout(content, 1)

        # ============================================================
        # 底部状态栏
        # ============================================================
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(0, 4, 0, 0)

        # 提示文字
        self.status_label = QLabel("💡 光线充足环境效果更佳")
        self.status_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        # 信号质量指示器
        signal_layout = QHBoxLayout()
        signal_layout.setSpacing(8)
        signal_icon = QLabel("📶")
        signal_icon.setStyleSheet("font-size: 14px;")
        signal_layout.addWidget(signal_icon)

        # 质量条背景
        self.quality_bar_bg = QFrame()
        self.quality_bar_bg.setFixedSize(120, 5)
        self.quality_bar_bg.setStyleSheet("background-color: #E5E5EA; border-radius: 2.5px;")
        bar_layout = QHBoxLayout(self.quality_bar_bg)
        bar_layout.setContentsMargins(0, 0, 0, 0)
        # 质量条填充
        self.quality_fill = QFrame()
        self.quality_fill.setFixedSize(0, 5)
        self.quality_fill.setStyleSheet("background-color: #34C759; border-radius: 2.5px;")
        bar_layout.addWidget(self.quality_fill)
        signal_layout.addWidget(self.quality_bar_bg)

        self.quality_label = QLabel("0%")
        self.quality_label.setStyleSheet("color: #8E8E93; font-size: 12px;")
        signal_layout.addWidget(self.quality_label)
        status_layout.addLayout(signal_layout)

        # FPS显示
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #8E8E93; font-size: 12px; margin-left: 16px;")
        status_layout.addWidget(self.fps_label)
        main_layout.addLayout(status_layout)

        # 定时器：每33ms触发一次（约30fps）
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _populate_cameras(self):
        """
        填充摄像头下拉列表

        检测系统中所有可用摄像头，并与配置列表匹配显示
        """
        self.camera_combo.blockSignals(True)  # 阻止信号触发
        self.camera_combo.clear()

        # 检测所有可用摄像头
        detected = []
        for i in range(10):  # 最多检测10个
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                detected.append({'index': i, 'width': w, 'height': h})
                cap.release()

        # 按配置顺序添加摄像头到下拉列表
        for cam in self.camera_list:
            if cam['type'] == 'local':
                idx = cam['source']
                try:
                    idx_int = int(idx)
                    # 查找是否检测到该摄像头
                    found = next((c for c in detected if c['index'] == idx_int), None)
                    if found:
                        label = f"{cam['name']} ({found['width']}×{found['height']})"
                    else:
                        label = f"{cam['name']} (未检测)"
                except ValueError:
                    label = f"{cam['name']} ({idx})"
                self.camera_combo.addItem(label, idx)

        self.camera_combo.blockSignals(False)

    def _manage_cameras(self):
        """打开摄像头管理对话框"""
        if self.is_running:
            QMessageBox.warning(self, "提示", "请先停止监测")
            return
        dialog = CameraManagerDialog(self.camera_list, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.camera_list = dialog.get_cameras()
            self._populate_cameras()

    def _on_camera_changed(self, index):
        """
        摄像头下拉框选择变化处理

        如果正在监测，自动重启以切换摄像头
        """
        if self.camera_combo.signalsBlocked():
            return
        if self.is_running:
            self.stop_monitoring()
            QTimer.singleShot(800, self.start_monitoring)  # 延迟800ms后重启

    def _toggle_camera(self):
        """
        切换到下一个摄像头

        循环切换下拉列表中的摄像头
        """
        count = self.camera_combo.count()
        if count > 1:
            if self.is_running:
                self.stop_monitoring()
            # 切换到下一个
            next_idx = (self.camera_combo.currentIndex() + 1) % count
            self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentIndex(next_idx)
            self.camera_combo.blockSignals(False)
            if self.is_running:
                QTimer.singleShot(800, self.start_monitoring)

    def toggle_monitoring(self):
        """切换监测状态（开始/停止）"""
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()

    def start_monitoring(self):
        """
        开始监测

        初始化视频捕获、重置算法状态、启动定时器
        """
        # 释放之前的摄像头
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None

        # 获取选中的摄像头索引
        camera_data = self.camera_combo.currentData()
        try:
            camera_index = int(camera_data)
        except (ValueError, TypeError):
            QMessageBox.warning(self, "错误", "无效的摄像头配置")
            return

        # 尝试打开摄像头（优先使用AVFoundation后端）
        self.video_capture = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
        if not self.video_capture.isOpened():
            self.video_capture = cv2.VideoCapture(camera_index)

        # 检查是否成功打开
        if not self.video_capture or not self.video_capture.isOpened():
            QMessageBox.warning(self, "错误", f"无法打开摄像头 index {camera_index}")
            self.is_running = False
            return

        # 设置分辨率
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # 重置所有算法模块
        self.rppg_processor.reset()
        self.hrv_analyzer.reset()
        self.emotion_recognizer.reset()

        # 更新状态
        self.is_running = True
        self.start_btn.setText("⏹ 停止")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF3B30;
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #E5342B; }
        """)

        # 启动定时器（约30fps）
        self.timer.start(33)

    def stop_monitoring(self):
        """停止监测"""
        self.is_running = False
        self.start_btn.setText("▶ 开始")
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #007AFF;
                color: #FFFFFF;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #0071E3; }
        """)

        # 停止定时器
        self.timer.stop()

        # 释放摄像头
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
            self.video_capture = None

    def update_frame(self):
        """
        定时器触发：处理每一帧

        流程：
        1. 读取视频帧
        2. 人脸检测
        3. 提取rPPG信号
        4. 处理信号并更新UI
        """
        # 安全检查
        if not self.is_running or not self.video_capture or not self.video_capture.isOpened():
            return

        # 读取帧
        ret, frame = self.video_capture.read()
        if not ret:
            return

        # 人脸检测
        landmarks, face_roi, blendshapes = self.face_detector.detect(frame)
        display_frame = frame.copy()

        # 如果检测到人脸
        if face_roi:
            x, y, w, h = face_roi
            # 绘制人脸框（绿色）
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (52, 199, 89), 2)

            if landmarks:
                # 提取ROI像素均值
                f_mean, lc_mean, rc_mean = self.face_detector.extract_roi_pixels(frame, landmarks)
                # 添加到rPPG处理器
                self.rppg_processor.add_frame(f_mean, lc_mean, rc_mean)

                # 提取表情特征
                expr_features = self.face_detector.extract_expression_features(landmarks, blendshapes, frame.shape)
                # 处理信号并更新UI
                self._process_signals(expr_features)

        # 显示视频帧（水平翻转）
        self.video_panel.set_frame(display_frame, flip=True)

        # FPS计算
        self.frame_count += 1
        curr_time = time.time()
        if curr_time - self.last_fps_time >= 1:
            self.fps_label.setText(f"FPS: {self.frame_count / (curr_time - self.last_fps_time):.1f}")
            self.frame_count = 0
            self.last_fps_time = curr_time

    def _process_signals(self, expression_features):
        """
        处理生理信号并更新UI

        参数:
            expression_features: 表情特征向量
        """
        # 处理rPPG信号获取心率
        hr, quality = self.rppg_processor.process()

        if hr and hr > 0:
            # 更新心率显示
            self.hr_card.set_value(str(int(hr)))

            # 更新信号质量
            q_val = min(100, int(quality * 100)) if quality else 0
            self.quality_fill.setFixedWidth(int(1.6 * q_val))
            self.quality_label.setText(f"{q_val}%")

            # 更新心率波形图
            sig_data = self.rppg_processor.get_signal_for_display()
            if sig_data:
                self.hr_graph.set_data(sig_data)
        else:
            self.hr_card.set_value("--")

        # 处理HRV
        rppg_sig = self.rppg_processor.get_rppg_signal()
        if rppg_sig is not None and len(rppg_sig) > 100:
            rmssd, _ = self.hrv_analyzer.analyze(rppg_sig)
            if rmssd and rmssd > 0:
                self.hrv_card.set_value(str(int(rmssd)))

        # 处理情绪
        hrv_status = self.hrv_analyzer.get_status()
        emotion, conf = self.emotion_recognizer.recognize(
            expression_features, hrv_status, self.rppg_processor.current_hr
        )
        if emotion:
            self.emotion_card.set_emotion(emotion, conf)


# ============================================================
# 程序入口
# ============================================================

def main():
    """程序入口函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # 使用Fusion风格
    window = FaceVitalsApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

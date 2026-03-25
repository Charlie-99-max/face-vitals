# 配置文件

# 颜色配置 (HEX)
COLORS = {
    'primary': '#1A1A2E',
    'secondary': '#16213E',
    'accent': '#E94560',
    'success': '#4ADE80',
    'warning': '#FBBF24',
    'background': '#0F0F1A',
    'text_primary': '#FFFFFF',
    'text_secondary': '#94A3B8',
    'card_bg': '#1E1E2E',
}

# 窗口配置
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800
MIN_WIDTH = 1024
MIN_HEIGHT = 768

# rPPG配置
RPPG_CONFIG = {
    'sample_rate': 30,  # 视频帧率
    'buffer_size': 450,  # 30秒 * 15fps = 450帧
    'min_analysis_frames': 150,  # 至少5秒数据
    'fft_window': 128,
    'hr_range': (40, 180),  # 心率范围
}

# HRV配置
HRV_CONFIG = {
    'min_rr_intervals': 10,
    'rmsd_threshold': 30,
}

# 情绪配置
EMOTION_CONFIG = {
    'emotions': ['Calm', 'Happy', 'Anxious', 'Stressed', 'Sad'],
    'emotion_colors': {
        'Calm': '#4ADE80',
        'Happy': '#FBBF24',
        'Anxious': '#F97316',
        'Stressed': '#E94560',
        'Sad': '#3B82F6',
    }
}

# 面部ROI配置
FACE_ROI = {
    'forehead': (10, 20),  # 额头区域
    'left_cheek': (50, 60),
    'right_cheek': (280, 290),
}

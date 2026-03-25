import numpy as np
from collections import deque
import heartpy as hp


class RPPGProcessor:
    """
    基于heartpy的专业心率检测
    使用多算法融合
    """

    def __init__(self, buffer_size=600, fs=30):
        self.buffer_size = buffer_size
        self.fs = fs

        # 信号缓冲区
        self.r_buffer = deque(maxlen=buffer_size)
        self.g_buffer = deque(maxlen=buffer_size)
        self.b_buffer = deque(maxlen=buffer_size)
        self.skin_signal_buffer = deque(maxlen=buffer_size)

        # 状态
        self.current_hr = None
        self.signal_quality = 0

        # 心率历史
        self.hr_history = deque(maxlen=15)

    def reset(self):
        self.r_buffer.clear()
        self.g_buffer.clear()
        self.b_buffer.clear()
        self.skin_signal_buffer.clear()
        self.hr_history.clear()
        self.current_hr = None
        self.signal_quality = 0

    def add_frame(self, forehead_roi, left_cheek_roi, right_cheek_roi):
        """添加一帧数据"""
        if forehead_roi is None and left_cheek_roi is None and right_cheek_roi is None:
            return

        try:
            regions = []
            if forehead_roi is not None and forehead_roi.size > 0:
                forehead_mean = np.mean(forehead_roi, axis=(0, 1))
                regions.append(('forehead', forehead_mean))
            if left_cheek_roi is not None and left_cheek_roi.size > 0:
                left_cheek_mean = np.mean(left_cheek_roi, axis=(0, 1))
                regions.append(('left_cheek', left_cheek_mean))
            if right_cheek_roi is not None and right_cheek_roi.size > 0:
                right_cheek_mean = np.mean(right_cheek_roi, axis=(0, 1))
                regions.append(('right_cheek', right_cheek_mean))

            if regions:
                total = np.zeros(3)
                weights = {'forehead': 0.4, 'left_cheek': 0.3, 'right_cheek': 0.3}
                for name, color in regions:
                    # OpenCV使用BGR格式: color[0]=B, color[1]=G, color[2]=R
                    total += color * weights[name]

                # 正确提取RGB通道 (total是BGR格式)
                self.b_buffer.append(total[0])  # B通道
                self.g_buffer.append(total[1])  # G通道 - 心率检测主要用这个
                self.r_buffer.append(total[2])  # R通道
                self.skin_signal_buffer.append(total[1])  # 使用绿色通道

        except Exception as e:
            pass

    def process(self):
        """使用heartpy处理信号"""
        if len(self.skin_signal_buffer) < 180:  # 至少6秒
            return None, 0

        signal = np.array(self.skin_signal_buffer)

        # 预处理
        try:
            # 滤波 - 使用改进的参数
            signal = hp.filtering.filter_signal(signal,
                                                cutoff=[0.7, 4.0],
                                                filtertype='bandpass',
                                                fs=self.fs,
                                                order=3)

            # 使用heartpy检测峰值
            wd, m = hp.process(signal, self.fs)

            if 'bpm' in m and m['bpm']:
                hr = m['bpm']

                # 验证心率范围
                if 40 <= hr <= 180:
                    # 平滑
                    self.hr_history.append(hr)
                    if len(self.hr_history) > 3:
                        smoothed_hr = np.median(list(self.hr_history)[-5:])
                    else:
                        smoothed_hr = hr

                    self.current_hr = smoothed_hr

                    # 信号质量 - 改进评估
                    if 'snr' in m:
                        self.signal_quality = min(100, m['snr'] * 15)
                    elif 'quality' in m:
                        self.signal_quality = min(100, m['quality'] * 100)
                    else:
                        # 基于信号波动估计质量
                        signal_var = np.var(signal)
                        if signal_var > 0:
                            self.signal_quality = min(100, signal_var * 50)
                        else:
                            self.signal_quality = 50

                    return self.current_hr, self.signal_quality

        except Exception as e:
            # 如果heartpy失败，使用备用方法
            pass

        return self._fallback_process()

    def _fallback_process(self):
        """备用处理方法"""
        if len(self.skin_signal_buffer) < 200:
            return None, 0

        from utils.signal_utils import (
            detrend_signal, bandpass_filter, compute_frequency_hr
        )

        signal = np.array(self.skin_signal_buffer)
        signal = detrend_signal(signal)
        signal = bandpass_filter(signal, 0.7, 4, self.fs)

        hr, quality = compute_frequency_hr(signal, self.fs)

        if hr and 40 <= hr <= 180:
            self.hr_history.append(hr)
            if len(self.hr_history) > 3:
                smoothed_hr = np.median(list(self.hr_history)[-5:])
            else:
                smoothed_hr = hr
            self.current_hr = smoothed_hr
            self.signal_quality = quality / 10

        return self.current_hr, self.signal_quality

    def get_rppg_signal(self):
        """获取rPPG信号用于HRV分析"""
        if len(self.skin_signal_buffer) < 200:
            return None
        return np.array(self.skin_signal_buffer)

    def get_buffer_length(self):
        return len(self.skin_signal_buffer)

    def get_signal_for_display(self):
        if len(self.skin_signal_buffer) < 10:
            return []
        return list(self.skin_signal_buffer)[-100:]

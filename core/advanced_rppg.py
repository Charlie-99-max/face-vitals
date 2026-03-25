"""
Advanced rPPG Processor - 基于深度信号处理
使用多区域融合、自适应滤波和峰值检测
"""

import numpy as np
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class AdvancedRPPGProcessor:
    """
    高级rPPG处理器
    核心算法：多区域信号提取 + 自适应滤波 + 峰值对齐融合
    """

    def __init__(self, buffer_size=900, fs=30):
        self.buffer_size = buffer_size  # 30秒数据
        self.fs = fs

        # 多区域信号缓冲区
        self.forehead_r = deque(maxlen=buffer_size)
        self.forehead_g = deque(maxlen=buffer_size)
        self.forehead_b = deque(maxlen=buffer_size)

        self.cheek_r = deque(maxlen=buffer_size)
        self.cheek_g = deque(maxlen=buffer_size)
        self.cheek_b = deque(maxlen=buffer_size)

        # 组合信号
        self.combined_r = deque(maxlen=buffer_size)
        self.combined_g = deque(maxlen=buffer_size)
        self.combined_b = deque(maxlen=buffer_size)

        # 状态
        self.current_hr = None
        self.signal_quality = 0

        # 心率历史（用于平滑）
        self.hr_history = deque(maxlen=20)
        self.last_hr = None

    def reset(self):
        self.forehead_r.clear()
        self.forehead_g.clear()
        self.forehead_b.clear()
        self.cheek_r.clear()
        self.cheek_g.clear()
        self.cheek_b.clear()
        self.combined_r.clear()
        self.combined_g.clear()
        self.combined_b.clear()
        self.hr_history.clear()
        self.current_hr = None
        self.signal_quality = 0
        self.last_hr = None

    def add_frame(self, forehead_mean, left_cheek_mean, right_cheek_mean):
        """从多个区域提取信号 - 适配多边形掩膜返回的BGR均值"""
        try:
            # forehead_mean, left_cheek_mean, right_cheek_mean 现在是 BGR 均值 tuple (B, G, R)
            # 额头区域
            if forehead_mean is not None and len(forehead_mean) >= 3:
                self.forehead_b.append(forehead_mean[0])  # B
                self.forehead_g.append(forehead_mean[1])  # G
                self.forehead_r.append(forehead_mean[2])  # R
            else:
                self.forehead_r.append(0)
                self.forehead_g.append(0)
                self.forehead_b.append(0)

            # 脸颊区域 - 取左右脸颊均值
            if left_cheek_mean is not None and right_cheek_mean is not None:
                if len(left_cheek_mean) >= 3 and len(right_cheek_mean) >= 3:
                    c_b = (left_cheek_mean[0] + right_cheek_mean[0]) / 2
                    c_g = (left_cheek_mean[1] + right_cheek_mean[1]) / 2
                    c_r = (left_cheek_mean[2] + right_cheek_mean[2]) / 2
                    self.cheek_b.append(c_b)
                    self.cheek_g.append(c_g)
                    self.cheek_r.append(c_r)
                else:
                    self.cheek_r.append(0)
                    self.cheek_g.append(0)
                    self.cheek_b.append(0)
            else:
                self.cheek_r.append(0)
                self.cheek_g.append(0)
                self.cheek_b.append(0)

            # 组合信号（额头权重高）
            if len(self.forehead_r) > 0:
                cr = 0.5 * list(self.forehead_r)[-1] + 0.5 * list(self.cheek_r)[-1]
                cg = 0.5 * list(self.forehead_g)[-1] + 0.5 * list(self.cheek_g)[-1]
                cb = 0.5 * list(self.forehead_b)[-1] + 0.5 * list(self.cheek_b)[-1]
                self.combined_r.append(cr)
                self.combined_g.append(cg)
                self.combined_b.append(cb)

        except Exception as e:
            pass

    def process(self):
        """处理信号 - 核心算法"""
        min_frames = int(10 * self.fs)  # 至少10秒数据
        if len(self.combined_g) < min_frames:
            return None, 0

        # 获取原始信号
        r = np.array(self.combined_r)
        g = np.array(self.combined_g)
        b = np.array(self.combined_b)

        # 验证信号有效性
        valid_frames = np.sum((g > 0) & (g < 255))
        if valid_frames < min_frames * 0.5:
            return None, 0

        # 预处理：去除异常值
        g = self._remove_outliers(g)
        r = self._remove_outliers(r)
        b = self._remove_outliers(b)

        # 步骤1: 颜色空间归一化
        rgb_mean = np.mean([r, g, b], axis=1)
        rgb_std = np.std([r, g, b], axis=1)

        # 避免除零
        rgb_std[rgb_std < 1] = 1

        r_p = (r - rgb_mean[0]) / rgb_std[0]
        g_p = (g - rgb_mean[1]) / rgb_std[1]
        b_p = (b - rgb_mean[2]) / rgb_std[2]

        # 步骤2: 提取脉搏信号 (多算法融合)

        # 算法1: Green通道（最常用）
        signal1 = self._extract_green(g_p)

        # 算法2: CHROM
        signal2 = self._extract_chrom(r_p, g_p, b_p)

        # 算法3: POS
        signal3 = self._extract_pos(r_p, g_p, b_p)

        # 步骤3: 融合信号
        combined_signal = (signal1 + signal2 + signal3) / 3

        # 步骤4: 自适应滤波
        filtered = self._adaptive_filter(combined_signal)

        # 步骤5: 心率提取
        hr, quality = self._extract_heart_rate(filtered)

        if hr is None:
            return None, 0

        # 步骤6: 平滑处理
        hr = self._smooth_hr(hr)

        self.current_hr = hr
        self.signal_quality = min(100, quality)

        return self.current_hr, self.signal_quality

    def _remove_outliers(self, data, threshold=3):
        """去除异常值"""
        if len(data) < 10:
            return data

        median = np.median(data)
        mad = np.median(np.abs(data - median))

        if mad == 0:
            return data

        outliers = np.abs(data - median) > threshold * mad * 1.4826
        data_clean = data.copy()
        data_clean[outliers] = median

        return data_clean

    def _extract_green(self, g):
        """绿色通道提取 - 改进版"""
        from scipy import signal as sig

        # 归一化
        g = g - np.mean(g)

        # 先用低通滤波去除高频噪声
        b_lp, a_lp = sig.butter(3, 5 / (self.fs/2), btype='low')
        try:
            g = sig.filtfilt(b_lp, a_lp, g)
        except:
            pass

        # 带通滤波 (0.7-4 Hz) - 心率范围
        b, a = sig.butter(3, [0.7 / (self.fs/2), 4 / (self.fs/2)], btype='band')
        try:
            g = sig.filtfilt(b, a, g)
        except:
            pass

        return g

    def _extract_chrom(self, r, g, b):
        """CHROM算法 - 改进版"""
        from scipy import signal as sig

        # 归一化
        r = r - np.mean(r)
        g = g - np.mean(g)
        b = b - np.mean(b)

        # 投影 - 使用改进的CHROM公式
        X = np.stack([r, g, b])

        # 标准CHROM算法
        S = 3 * X[0] - 2 * X[1]

        # 归一化
        S = S - np.mean(S)
        S_std = np.std(S)
        if S_std > 0:
            S = S / S_std

        # 带通滤波 (0.7-4 Hz)
        b_coef, a_coef = sig.butter(3, [0.7 / (self.fs/2), 4 / (self.fs/2)], btype='band')
        try:
            S = sig.filtfilt(b_coef, a_coef, S)
        except:
            pass

        return S

    def _extract_pos(self, r, g, b):
        """POS算法 - 改进版"""
        # 皮肤正交平面
        cn = np.stack([r, g, b], axis=1)
        cn = cn - cn.mean(axis=0)

        # 标准POS算法
        S = cn[:, 1] - cn[:, 0]
        St = cn[:, 1] - 0.5 * cn[:, 0] - 0.5 * cn[:, 2]

        # 正交组合
        signal = S - np.mean(S)

        # 滤波 (0.7-4 Hz)
        from scipy import signal as sig
        b_coef, a_coef = sig.butter(3, [0.7 / (self.fs/2), 4 / (self.fs/2)], btype='band')
        try:
            signal = sig.filtfilt(b_coef, a_coef, signal)
        except:
            pass

        return signal

    def _adaptive_filter(self, signal_data):
        """自适应滤波 - 改进版"""
        from scipy import signal as sig

        # 去除趋势 - 使用更高的多项式阶数
        x = np.arange(len(signal_data))
        try:
            coeffs = np.polyfit(x, signal_data, 2)
            trend = np.polyval(coeffs, x)
            signal_data = signal_data - trend
        except:
            signal_data = signal_data - np.mean(signal_data)

        # 带通滤波 (0.7-4 Hz)
        b_coef, a_coef = sig.butter(3, [0.7 / (self.fs/2), 4 / (self.fs/2)], btype='band')
        try:
            signal_data = sig.filtfilt(b_coef, a_coef, signal_data)
        except:
            pass

        # 额外的去噪 - 中值滤波去除脉冲噪声
        try:
            signal_data = sig.medfilt(signal_data, kernel_size=3)
        except:
            pass

        return signal_data

    def _extract_heart_rate(self, signal_data):
        """提取心率 - 改进版Welch方法"""
        if len(signal_data) < 128:
            return None, 0

        # Welch's method - 使用更大的窗口以获得更好的频率分辨率
        nperseg = min(512, len(signal_data) // 2)
        if nperseg < 64:
            return None, 0

        try:
            f, psd = signal.welch(signal_data, fs=self.fs, nperseg=nperseg, noverlap=nperseg//2)
        except:
            return None, 0

        # 心率范围 0.7-4 Hz (42-240 BPM)
        valid = (f >= 0.7) & (f <= 4)
        if not np.any(valid):
            return None, 0

        f_valid = f[valid]
        psd_valid = psd[valid]

        if len(f_valid) == 0:
            return None, 0

        # 找峰值 - 使用更好的峰值检测
        peak_idx = np.argmax(psd_valid)
        hr_freq = f_valid[peak_idx]
        hr = hr_freq * 60

        # 验证心率范围
        if not (40 <= hr <= 180):
            return None, 0

        # 计算信号质量 - 改进的信噪比估计
        signal_power = psd_valid[peak_idx]
        total_power = np.sum(psd_valid)

        # 计算相邻频率的功率作为噪声估计
        if peak_idx > 0 and peak_idx < len(psd_valid) - 1:
            noise_estimate = (psd_valid[peak_idx-1] + psd_valid[peak_idx+1]) / 2
        else:
            noise_estimate = total_power / len(psd_valid)

        if noise_estimate > 0:
            snr = signal_power / (noise_estimate + 1e-6)
            quality = min(100, snr * 20)
        else:
            quality = signal_power / (total_power + 1e-6) * 100

        return hr, quality

    def _smooth_hr(self, hr):
        """心率平滑"""
        # 如果与上次差异太大，怀疑是噪声
        if self.last_hr is not None:
            if abs(hr - self.last_hr) > 20:
                # 使用历史平均
                if len(self.hr_history) > 3:
                    median_hr = np.median(list(self.hr_history)[-5:])
                    hr = 0.7 * median_hr + 0.3 * hr

        self.hr_history.append(hr)
        self.last_hr = hr

        # 移动平均
        if len(self.hr_history) >= 5:
            window = list(self.hr_history)[-5:]
            smoothed = np.median(window)
        else:
            smoothed = hr

        return smoothed

    def get_rppg_signal(self):
        """获取处理后的信号"""
        if len(self.combined_g) < 200:
            return None
        return np.array(self.combined_g)

    def get_buffer_length(self):
        return len(self.combined_g)

    def get_signal_for_display(self):
        if len(self.combined_g) < 10:
            return []
        return list(self.combined_g)[-100:]

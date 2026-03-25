import numpy as np
from collections import deque
import heartpy as hp
from scipy.interpolate import CubicSpline

class HRVAnalyzer:
    """
    基于 heartpy 的专业 HRV 分析
    加入了三次样条插值，提升 HRV 计算的毫秒级精度
    """

    def __init__(self, fs=30):
        self.fs = fs
        self.current_rmssd = None
        self.current_hrv_status = None
        self.rr_intervals_history = deque(maxlen=100)
        self.is_valid = False
        self.rmssd_history = deque(maxlen=10)

    def reset(self):
        self.rr_intervals_history.clear()
        self.rmssd_history.clear()
        self.current_rmssd = None
        self.current_hrv_status = None
        self.is_valid = False

    def analyze(self, rppg_signal):
        if rppg_signal is None or len(rppg_signal) < 300:
            return None, None

        try:
            # 1. 原始滤波
            signal = hp.filtering.filter_signal(rppg_signal, cutoff=[0.7, 4.0], filtertype='bandpass', fs=self.fs, order=3)

            # 2. ==== 核心优化：插值上采样到 250Hz ====
            target_fs = 250
            duration = len(signal) / self.fs
            # 创建原始时间轴和新的高频时间轴
            x_old = np.linspace(0, duration, len(signal))
            x_new = np.linspace(0, duration, int(len(signal) * (target_fs / self.fs)))
            
            # 使用三次样条插值生成平滑的高频信号
            cs = CubicSpline(x_old, signal)
            signal_upsampled = cs(x_new)

            # 3. 使用高频信号和高频采样率进行峰值检测
            wd, m = hp.process(signal_upsampled, target_fs)

            if 'rr_list' in m and len(m['rr_list']) > 3:  # 至少需要几个间隔
                rr_intervals = np.array(m['rr_list'])

                # 过滤异常RR间期
                valid_rr = self._filter_rr(rr_intervals)

                if len(valid_rr) >= 3: 
                    try:
                        # process 返回的 rr_list 已经是毫秒级，直接计算指标
                        rr_m = hp.process_rr(valid_rr)

                        if 'rmssd' in rr_m:
                            rmssd = rr_m['rmssd']

                            # 扩大过滤范围
                            if 10 < rmssd < 250:
                                # 平滑
                                self.rmssd_history.append(rmssd)
                                if len(self.rmssd_history) >= 2:
                                    smoothed_rmssd = np.median(list(self.rmssd_history))
                                else:
                                    smoothed_rmssd = rmssd

                                self.current_rmssd = smoothed_rmssd

                                # 状态
                                if smoothed_rmssd > 50:
                                    status = 'high'
                                elif smoothed_rmssd >= 30:
                                    status = 'normal'
                                else:
                                    status = 'low'

                                self.current_hrv_status = status
                                self.is_valid = True

                                return smoothed_rmssd, status

                    except Exception as inner_e:
                        pass

        except Exception as e:
            pass

        return self._fallback_analyze(rppg_signal)

    def _filter_rr(self, rr_intervals):
        """过滤异常RR间期"""
        if len(rr_intervals) < 3:
            return rr_intervals

        median_rr = np.median(rr_intervals)
        mad = np.median(np.abs(rr_intervals - median_rr))

        if mad == 0:
            return rr_intervals

        threshold = 3.0
        lower = median_rr - threshold * mad * 1.4826
        upper = median_rr + threshold * mad * 1.4826

        # 心率范围检查：毫秒级的 RR 间期限制
        lower = max(lower, 333)   # 对应约 180 BPM
        upper = min(upper, 1500)  # 对应约 40 BPM

        valid = [rr for rr in rr_intervals if lower <= rr <= upper]
        return np.array(valid)

    def _fallback_analyze(self, rppg_signal):
        """备用HRV分析"""
        from utils.signal_utils import (
            find_peaks_in_signal, compute_rr_intervals,
            compute_rmssd, bandpass_filter, detrend_signal
        )

        if rppg_signal is None or len(rppg_signal) < 200: 
            return None, None

        signal = detrend_signal(rppg_signal)
        signal = bandpass_filter(signal, 0.7, 4, self.fs)

        peaks = find_peaks_in_signal(signal, self.fs)
        if len(peaks) < 3:  
            return None, None

        rr_intervals = compute_rr_intervals(peaks, self.fs)
        if len(rr_intervals) < 2:  
            return None, None

        valid_rr = self._filter_rr(rr_intervals)
        if len(valid_rr) < 2: 
            return None, None

        rmssd = compute_rmssd(valid_rr)

        if rmssd > 250 or rmssd < 10: 
            return None, None

        self.rmssd_history.append(rmssd)
        if len(self.rmssd_history) >= 2: 
            smoothed = np.median(list(self.rmssd_history))
        else:
            smoothed = rmssd

        self.current_rmssd = smoothed

        if smoothed > 50:
            status = 'high'
        elif smoothed >= 30:
            status = 'normal'
        else:
            status = 'low'

        self.current_hrv_status = status
        self.is_valid = True

        return smoothed, status

    def get_rmssd(self):
        return self.current_rmssd

    def get_status(self):
        return self.current_hrv_status

    def get_status_description(self):
        if self.current_hrv_status == 'high':
            return "放松状态 - 副交感神经活跃"
        elif self.current_hrv_status == 'normal':
            return "正常状态 - 平衡的自主神经"
        elif self.current_hrv_status == 'low':
            return "压力状态 - 交感神经占主导"
        return "等待数据..."

    def is_ready(self):
        return self.is_valid

    def get_rr_data(self):
        if len(self.rr_intervals_history) < 5:
            return []
        return list(self.rr_intervals_history)
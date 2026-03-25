import numpy as np
from collections import deque
from scipy import signal
from scipy.fft import fft, fftfreq
from sklearn.decomposition import FastICA


def butter_bandpass(lowcut, highcut, fs, order=4):
    """设计带通滤波器"""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """应用带通滤波器"""
    if len(data) < 32:
        return data
    try:
        b, a = butter_bandpass(lowcut, highcut, fs, order)
        return signal.filtfilt(b, a, data)
    except:
        return data


def detrend_signal(signal_data):
    """去除信号趋势"""
    if len(signal_data) < 10:
        return signal_data
    x = np.arange(len(signal_data))
    try:
        coeffs = np.polyfit(x, signal_data, 1)
        trend = np.polyval(coeffs, x)
        return signal_data - trend
    except:
        return signal_data - np.mean(signal_data)


def compute_signal_quality(green_channel):
    """计算信号质量"""
    if len(green_channel) < 30:
        return 0
    signal_power = np.var(green_channel)
    noise_estimate = np.mean(np.abs(np.diff(green_channel)))
    if noise_estimate == 0:
        return 0
    snr = signal_power / (noise_estimate ** 2)
    quality = min(100, snr / 50)
    return quality


def find_peaks_in_signal(signal_data, fs, min_hr=40, max_hr=180):
    """峰值检测"""
    if len(signal_data) < 64:
        return np.array([])
    min_distance = int(60 / max_hr * fs)
    max_distance = int(60 / min_hr * fs)
    signal_smooth = signal.medfilt(signal_data, 5)
    threshold = np.mean(signal_smooth) + 0.3 * np.std(signal_smooth)
    peaks, _ = signal.find_peaks(
        signal_smooth,
        distance=min_distance,
        height=threshold,
        prominence=np.std(signal_data) * 0.3
    )
    return peaks


def compute_rr_intervals(peaks, fs):
    """计算RR间期"""
    if len(peaks) < 2:
        return np.array([])
    return np.diff(peaks) / fs * 1000


def compute_rmssd(rr_intervals):
    """计算RMSSD"""
    if len(rr_intervals) < 2:
        return 0
    successive_diffs = np.diff(rr_intervals)
    return np.sqrt(np.mean(successive_diffs ** 2))


def compute_frequency_hr(signal_data, fs):
    """使用Welch方法计算心率"""
    if len(signal_data) < 128:
        return None, 0
    filtered = bandpass_filter(signal_data, 0.7, 4, fs, order=3)
    if len(filtered) < 64:
        return None, 0
    try:
        f, psd = signal.welch(filtered, fs=fs, nperseg=min(256, len(filtered)//2))
        valid_range = (f >= 0.7) & (f <= 4)
        if not np.any(valid_range):
            return None, 0
        f_valid = f[valid_range]
        psd_valid = psd[valid_range]
        if len(f_valid) == 0:
            return None, 0
        peak_idx = np.argmax(psd_valid)
        hr_freq = f_valid[peak_idx]
        hr = hr_freq * 60
        if 45 <= hr <= 180:
            return hr, psd_valid[peak_idx]
    except:
        pass
    return None, 0


def normalize_signal(signal_data):
    """归一化信号"""
    if np.std(signal_data) == 0:
        return signal_data
    return (signal_data - np.mean(signal_data)) / np.std(signal_data)

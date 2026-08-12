import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
def get_channel_inds(waveform, common_channel_locations):
    return np.where((waveform.get_channel_locations()[:, None] == common_channel_locations).all(axis=2).any(axis=1))[0]

def fit_ellipse(points: np.ndarray, scale_factor=2):
    """
       使用PCA拟合椭圆区域

       参数:
           points: (N, 2) numpy数组，椭圆区域点云
           confidence_level: 置信水平 (95%表示包含95%的点)
           scale_factor: 手动指定缩放因子，为None时自动计算

       返回:
           椭圆参数字典
    """

    # 1. 计算点云的均值（椭圆中心）
    center = np.mean(points, axis=0)

    # 2. 中心化数据
    centered = points - center

    # 3. 计算协方差矩阵
    cov_matrix = np.cov(centered.T)

    # 4. PCA：计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 5. 排序特征值（从大到小）和对应的特征向量
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 6. 计算标准差（特征值的平方根）
    std_devs = np.sqrt(eigenvalues)

    # 8. 计算椭圆轴长
    axes = std_devs * scale_factor

    # 9. 计算旋转角度（从x轴到第一主成分的夹角）
    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # 10. 确保长轴对应第一主成分
    assert(axes[0] < axes[1], "debug here!")
    # if axes[0] < axes[1]:
    #     axes = axes[::-1]
    #     angle += np.pi / 2

    # 规范化角度到 [0, π)
    angle = angle % np.pi

    return {
        'center': center,
        'axes': axes,
        'angle': angle,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'std_devs': std_devs
    }

def good_or_noise(isi_violation, firing_rate, FWHM, isi_violation_threshold = 10,
    firing_rate_threshold = 0.1, FWHM_lower_threshold = 0.15, FWHM_upper_threshold = 0.75):
     if isi_violation < isi_violation_threshold and firing_rate > firing_rate_threshold and \
             FWHM > FWHM_lower_threshold and FWHM < FWHM_upper_threshold:
         return True
     else:
         return False


def _compute_fwhm_basic(single_waveform, sampling_frequency):
    """基础版本计算半峰宽"""
    # 找到峰值点 (假设负向峰)
    single_waveform_abs = np.abs(single_waveform)
    peak_idx = np.argmax(single_waveform_abs)
    peak_val = single_waveform_abs[peak_idx]
    half_height = peak_val / 2.0

    # 寻找左交叉点
    left_idx = peak_idx
    while left_idx > 0 and single_waveform_abs[left_idx] >= half_height:
        left_idx -= 1

    # 寻找右交叉点
    right_idx = peak_idx
    while right_idx < len(single_waveform_abs) - 1 and single_waveform_abs[right_idx] >= half_height:
        right_idx += 1

    # 计算半峰宽
    half_width_samples = right_idx - left_idx
    return (half_width_samples / sampling_frequency) * 1000.0


def remove_extreme_outliers_iqr(data_series, multiplier=3):
    if isinstance(data_series, list):
        data_series = pd.Series(data_series)
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outlier_mask = (data_series < lower_bound) | (data_series > upper_bound)
    outliers = data_series[outlier_mask]
    outlier_indices = outliers.index.tolist()
    cleaned_data = data_series[~outlier_mask]
    return cleaned_data





import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

import numpy as np
import torchaudio
import random

from sympy.geometry.entity import scale

# from old.TRAIN_NO_EMA1 import permutation_invariant_loss



# 固定随机种子以确保结果可复现
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设置随机种子
set_random_seed(42)

import torch
import torch.nn as nn
import numpy as np
from torch.fft import fft, ifft, fftfreq


def compute_adaptive_scales(fs, duration, freq_low=80, freq_high=2000, num_scales=50):
    """
    自适应计算小波尺度
    Args:
        fs: 采样率
        duration: 语音时长（秒）
        freq_low: 最低频率（Hz）
        freq_high: 最高频率（Hz）
        num_scales: 尺度数量
    Returns:
        scales: 自适应计算的尺度数组
    """
    # 根据时长调整频率分辨率
    # 较长的语音可以使用更精细的频率分辨率
    base_num_scales = num_scales
    if duration > 6.0:  # 长语音使用更多尺度
        num_scales = int(base_num_scales * 1.2)
    elif duration < 2.0:  # 短语音使用较少尺度
        num_scales = int(base_num_scales * 0.8)

    # 计算频率范围
    freqs = np.linspace(freq_low, freq_high, num_scales)

    # 尺度计算：尺度与频率成反比
    scales = (5 / (2 * np.pi)) * fs / freqs

    return scales


class VectorizedMorletCWT(torch.nn.Module):
    def __init__(self, scales, c=343.0, samplerate=8000):
        super().__init__()
        self.scales = torch.tensor(scales, dtype=torch.float32)
        self.c = c
        self.samplerate = samplerate
        self.w0 = 5.0

        # 预计算小波滤波器组
        self.register_buffer('scales_buffer', self.scales)

    def update_scales(self, new_scales):
        """动态更新尺度"""
        self.scales_buffer = torch.tensor(new_scales, dtype=torch.float32, device=self.scales_buffer.device)

    def get_physical_params(self):
        """获取与尺度关联的物理参数（频率、波数、空间步长）"""
        scales = self.scales_buffer
        center_freqs = 5.0 * self.samplerate / (2 * np.pi * scales)  # 中心频率
        k = 2 * np.pi * center_freqs / self.c  # 波数
        dx = self.c / (4 * center_freqs)  # 空间步长（与频率成反比）
        return {
            "scales": scales,
            "center_freqs": center_freqs,
            "k": k,
            "dx": dx
        }

    def forward(self, x):
        """向量化Morlet小波变换 - 支持动态尺度"""
        batch_size, length = x.shape
        device = x.device

        # 补零到2的幂次方
        new_length = 2 ** int(np.ceil(np.log2(length)))
        if new_length != length:
            x_padded = torch.nn.functional.pad(x, (0, new_length - length))
        else:
            x_padded = x

        # FFT输入信号
        x_fft = fft(x_padded)  # [batch, new_length]

        # 获取频率轴
        freqs = fftfreq(new_length, d=1.0 / self.samplerate).to(device)  # [new_length]

        scales = self.scales_buffer.to(device)
        n_scales = len(scales)

        # 向量化计算所有尺度的小波
        freqs_expanded = freqs.unsqueeze(0)  # [1, new_length]
        scales_expanded = scales.unsqueeze(1)  # [n_scales, 1]

        # 计算omega
        omega = 2 * np.pi * freqs_expanded * scales_expanded  # [n_scales, new_length]

        # 创建掩码
        positive_mask = (freqs_expanded >= 0).expand(n_scales, new_length)

        # 向量化计算小波FFT
        wavelet_fft = torch.zeros(n_scales, new_length, dtype=torch.complex64, device=device)

        # 高斯包络
        gaussian = torch.exp(-0.5 * (omega - self.w0) ** 2)

        # 复数小波构造
        wavelet_fft[positive_mask] = (
                torch.pi ** (-0.25) *
                torch.sqrt(2 * np.pi * scales_expanded.expand_as(omega)[positive_mask]) *
                gaussian[positive_mask] *
                torch.exp(1j * self.w0 * (omega[positive_mask] - self.w0))
        )

        # L1归一化
        norm_factors = torch.sqrt(scales) * (np.pi ** 0.25) * torch.sqrt(torch.tensor(2.0, device=device))
        wavelet_fft = wavelet_fft / norm_factors.unsqueeze(1)  # [n_scales, new_length]

        # 频域卷积 - 向量化
        x_fft_expanded = x_fft.unsqueeze(1)  # [batch, 1, new_length]
        wavelet_fft_expanded = wavelet_fft.unsqueeze(0)  # [1, n_scales, new_length]

        coeff_fft = x_fft_expanded * wavelet_fft_expanded  # [batch, n_scales, new_length]

        # 逆FFT
        coeff = ifft(coeff_fft)  # [batch, n_scales, new_length]

        # 只取原始长度部分
        if new_length != length:
            coeff = coeff[:, :, :length]

        return coeff  # [batch, n_scales, length]


class OptimizedMorletCWTHelmholtzLoss(nn.Module):
    def __init__(self, fs=8000, duration=2.0, freq_low=80, freq_high=4000, num_scales=50, c=343.0):
        super().__init__()

        # 自适应计算尺度
        scales = compute_adaptive_scales(fs, duration, freq_low, freq_high, num_scales)

        self.cwt = VectorizedMorletCWT(scales, c, fs)
        self.c, self.fs, self.duration = c, fs, duration

        # 预计算中心频率和有效掩码
        scales_tensor = torch.tensor(scales)
        center_freqs = 5.0 * fs / (2 * np.pi * scales_tensor)
        self.valid_mask = (center_freqs >= freq_low) & (center_freqs <= freq_high)
        self.valid_center_freqs = center_freqs[self.valid_mask]
        self.valid_scales = scales_tensor[self.valid_mask]

        # 预计算波数k
        self.valid_k = 2 * np.pi * self.valid_center_freqs / c

        # 预计算空间步长
        self.valid_dx = c / fs * self.valid_scales
        self.valid_dx_squared = self.valid_dx ** 2

        # 从CWT获取物理参数（替代原有独立计算）
        phys_params = self.cwt.get_physical_params()

        # 注册为buffer确保设备一致性
        self.register_buffer('valid_mask_buffer', torch.tensor(self.valid_mask))
        self.register_buffer('valid_center_freqs_buffer', self.valid_center_freqs)
        self.register_buffer('valid_k_buffer', self.valid_k)
        self.register_buffer('valid_dx_squared_buffer', self.valid_dx_squared)
        self.register_buffer('valid_scales_buffer', self.valid_scales)  # 新增：注册有效尺度缓冲区
        print(f"自适应尺度计算完成: fs={fs}Hz, duration={duration}s, 使用{len(scales)}个尺度")


    def forward(self, waveform):
        B, T = waveform.shape

        # 动态调整尺度（保持不变）
        actual_duration = T / self.fs
        if abs(actual_duration - self.duration) > 0.1:
            scales = compute_adaptive_scales(self.fs, actual_duration)
            self.cwt.scales_buffer = torch.tensor(scales, device=waveform.device)
            self.duration = actual_duration

        # 计算小波系数（核心：保留系数供外部使用）
        coeffs = self.cwt(waveform)  # [B, n_scales, T] 复数
        valid_coeffs = coeffs[:, self.valid_mask_buffer, :]  # [B, n_valid, T]

        # ... 原有亥姆霍兹残差计算（保持不变）...
        # 1. 能量mask
        energy = valid_coeffs.abs().mean(dim=(0, 2))
        mask = energy > energy.median()
        # 2. 二阶差分（实部+虚部）
        d2u_real = torch.zeros_like(valid_coeffs.real)
        d2u_imag = torch.zeros_like(valid_coeffs.imag)
        d2u_real[:, :, 1:-1] = (valid_coeffs.real[:, :, 2:] - 2 * valid_coeffs.real[:, :, 1:-1] + valid_coeffs.real[:, :, :-2])
        d2u_imag[:, :, 1:-1] = (valid_coeffs.imag[:, :, 2:] - 2 * valid_coeffs.imag[:, :, 1:-1] + valid_coeffs.imag[:, :, :-2])
        # 3. dx计算（基于尺度）
        dx = self.c / (4 * self.valid_center_freqs_buffer)
        dx2 = dx.view(1, -1, 1) ** 2
        # 4. 残差计算
        residual_real = d2u_real[:, :, 1:-1] / dx2 + (2 * np.pi * self.valid_k_buffer).view(1, -1, 1) ** 2 * valid_coeffs.real[:, :, 1:-1]
        residual_imag = d2u_imag[:, :, 1:-1] / dx2 + (2 * np.pi * self.valid_k_buffer).view(1, -1, 1) ** 2 * valid_coeffs.imag[:, :, 1:-1]
        # 5. 损失计算
        loss_real = (residual_real.abs().square().mean(dim=(0, 2)) * mask.float()).sum()
        loss_imag = (residual_imag.abs().square().mean(dim=(0, 2)) * mask.float()).sum()
        freq_loss = (loss_real + loss_imag) / mask.sum().clamp(min=1)

        # 返回：频域损失 + 有效小波系数 + 对应尺度（供时域约束使用）
        return freq_loss, valid_coeffs, self.valid_scales_buffer[self.valid_mask_buffer]



class HybridALFIntegrator(nn.Module):
    def __init__(self, f_theta, alpha=0.2, delta_ratio=0.5, max_steps=5):
        super().__init__()
        self.f_theta = f_theta
        self.alpha = alpha
        self.delta_ratio = delta_ratio
        self.max_steps = max_steps

    def alf_step(self, z_in, v_in, t_in, dt,mix):
        # 使用你的中点法 + LFM 校正
        t_half = t_in + dt / 2
        z_half = z_in + v_in * (dt / 2)
        v_half = self.f_theta(t_half, z_half,mix)

        # 局部流匹配校正
        delta = dt * self.delta_ratio
        z_future = z_in + delta * v_in
        t_future = t_in + delta
        v_local = self.f_theta(t_future, z_future,mix)
        v_half = v_half + self.alpha * (v_local - v_half)

        z_out = z_in + (v_in + v_half) * (dt / 2)
        return z_out, v_half

    def forward(self, z0, t0, T,mix):
        # 使用我的多步策略
        z_cur, t_cur = z0, t0
        dt_total = T - t0
        dt_step = dt_total / self.max_steps

        for _ in range(self.max_steps):
            if t_cur >= T:
                break
            dt = min(dt_step, T - t_cur)
            v_cur = self.f_theta(t_cur, z_cur,mix)
            z_cur, v_cur = self.alf_step(z_cur, v_cur, t_cur, dt,mix)
            t_cur += dt

        return z_cur


class ALFIntegratorLFMStep(nn.Module):
    def __init__(self, f_theta, alpha=0.1, delta_ratio=0.5):
        super().__init__()
        self.f_theta = f_theta
        self.alpha = alpha          # 修正强度 0~1
        self.delta_ratio = delta_ratio  # δ = delta_ratio * dt

    def alf_step(self, z_in, v_in, t_in, dt,mix):
        t_half = t_in + dt / 2
        z_half = z_in + v_in * (dt / 2)
        v_half = self.f_theta(t_half, z_half,mix)

        # --- 局部流匹配校正 ---
        delta = dt * self.delta_ratio
        z_future = z_in + delta * v_in
        t_future = t_in + delta
        v_local = self.f_theta(t_future, z_future,mix)
        v_half = v_half + self.alpha * (v_local - v_half)   # 关键校正

        z_out = z_in + (v_in + v_half) * (dt / 2)
        return z_out, v_half

    def forward(self, z0, t0, T,mix):
        dt = T - t0
        v0 = self.f_theta(t0, z0,mix)
        z1, v1 = self.alf_step(z0, v0, t0, dt,mix)

        return z1

class FrequencyDomainHelmholtzPINN:
    """改进版频域亥姆霍兹约束的物理损失函数"""

    # 保持原有实现不变
    def __init__(self, sample_rate=8000, n_fft=512, hop_length=256, win_length=512, c=343.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.c = c
        self.dt = hop_length / sample_rate
        self.dx = c * self.dt
        self.frame_mask = torch.ones(1, 1, 1)
        self.freq_mask = torch.ones(1, 1, 1)

    def stft(self, waveform):
        window = torch.hann_window(self.win_length, device=waveform.device)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, self.win_length,
                          window=window, return_complex=True)
        return stft

    def istft(self, stft_matrix):
        window = torch.hann_window(self.win_length, device=stft_matrix.device)
        waveform = torch.istft(stft_matrix, self.n_fft, self.hop_length, self.win_length,
                               window=window)
        return waveform

    def compute_helmholtz_residual(self, U):
        U_real = torch.view_as_real(U)
        B, F, T, _ = U_real.shape

        if self.frame_mask.shape[2] != T:
            self.frame_mask = torch.ones(1, 1, T, device=U.device)
            if T > 2:
                self.frame_mask[:, :, [0, -1]] = 0

        freqs = torch.fft.fftfreq(self.n_fft, 1 / self.sample_rate)[:F].abs()
        freqs = freqs.to(U.device).unsqueeze(0).unsqueeze(-1)
        k_squared = (2 * torch.pi * freqs / self.c) ** 2

        laplacian_U = torch.zeros(B, F, T, 2, device=U.device)
        if T > 2:
            central_diff = (U_real[:, :, 2:] - 2 * U_real[:, :, 1:-1] + U_real[:, :, :-2])
            # 添加数值稳定性修正
            central_diff = torch.clamp(central_diff, -1e4, 1e4)  # 限制差分范围
            laplacian_U[:, :, 1:-1] = central_diff / (self.dx ** 2 + 1e-8)  # 避免除零

        helmholtz_residual = laplacian_U + k_squared.unsqueeze(-1) * U_real

        # 添加残差裁剪防止数值爆炸
        helmholtz_residual = torch.clamp(helmholtz_residual, -1e4, 1e4)

        residual_magnitude = helmholtz_residual[..., 0] ** 2 + helmholtz_residual[..., 1] ** 2

        # 应用频率权重
        weighted_residual = residual_magnitude

        valid_residual = weighted_residual * self.frame_mask
        residual_loss = 1e-7 * valid_residual.sum() / (self.frame_mask.sum() + 1e-8)
        return residual_loss


def central_diff_2nd(x, axis, step=1):
    """修正中心差分：处理单声道语音的维度"""
    if x.ndim < 2:
        raise ValueError("输入需为[batch, time]维度")
    # 中心差分公式：f''(x) ≈ [f(x+step) - 2f(x) + f(x-step)] / step²
    left = x.narrow(axis, 0, x.shape[axis] - 2 * step)
    mid = x.narrow(axis, step, x.shape[axis] - 2 * step)
    right = x.narrow(axis, 2 * step, x.shape[axis] - 2 * step)
    return (left - 2 * mid + right) / (step ** 2)


# 1. 修改时域波动方程损失类，支持从小波系数计算导数
class TimeDomainWaveEquationLoss(nn.Module):
    def __init__(self, c=343.0, spatial_win_step=5, samplerate=8000):
        super().__init__()
        self.c = c
        self.spatial_win_step = spatial_win_step  # 模拟空间步长（基于小波尺度）
        self.samplerate = samplerate

    def forward(self, cwt_coeffs, scales):
        """
        从小波系数计算时域波动方程残差
        Args:
            cwt_coeffs: 小波系数 [batch, n_scales, time]（复数）
            scales: 小波尺度 [n_scales]（与系数对应）
        """
        # 实部和虚部分开计算（小波系数为复数）
        coeffs_real = cwt_coeffs.real
        coeffs_imag = cwt_coeffs.imag

        # 1. 时域二阶导（对时间轴求导，每个尺度独立计算）
        time_deriv_real = central_diff_2nd(coeffs_real, axis=-1, step=1)  # [B, scales, T-2]
        time_deriv_imag = central_diff_2nd(coeffs_imag, axis=-1, step=1)

        # 2. 模拟空间二阶导（基于小波尺度的物理空间步长）
        # 空间步长 dx 与尺度正相关（尺度越大，对应频率越低，空间分辨率越低）
        dx = self.c / (4 * (5.0 * self.samplerate / (2 * np.pi * scales)))  # 从尺度推导dx
        dx = dx.view(1, -1, 1)  # [1, scales, 1]

        # 空间导数（使用与尺度匹配的步长）
        spatial_deriv_real = central_diff_2nd(coeffs_real, axis=-1, step=self.spatial_win_step)  # [B, scales, T-2*step]
        spatial_deriv_imag = central_diff_2nd(coeffs_imag, axis=-1, step=self.spatial_win_step)

        # 3. 对齐时间维度（取最小长度）
        min_len = min(time_deriv_real.shape[-1], spatial_deriv_real.shape[-1])
        time_deriv_real = time_deriv_real[..., :min_len]
        time_deriv_imag = time_deriv_imag[..., :min_len]
        spatial_deriv_real = spatial_deriv_real[..., :min_len]
        spatial_deriv_imag = spatial_deriv_imag[..., :min_len]

        # 4. 波动方程残差（实部+虚部）
        residual_real = time_deriv_real - (self.c **2) * spatial_deriv_real / (dx** 2 + 1e-8)
        residual_imag = time_deriv_imag - (self.c **2) * spatial_deriv_imag / (dx** 2 + 1e-8)

        # 对尺度和时间取平均
        return (torch.mean(residual_real **2) + torch.mean(residual_imag** 2)) / 2

class PhysicsConstrainedODEFunction(nn.Module):
    def __init__(self, base_ode_func, decoder, output_proj, helmholtz_pinn,
                 fs=8000, duration=2.0, constraint_weight=0.1):
        super().__init__()
        self.base_ode_func = base_ode_func
        self.decoder = decoder
        self.output_proj = output_proj
        self.helmholtz_pinn = helmholtz_pinn
        self.log_constraint_weight = nn.Parameter(torch.log(torch.tensor(constraint_weight)))

        # === 新增：物理损失和源一致性约束的可学习权重参数 ===
        # 使用对数空间以确保权重为正数，并通过指数函数转换回实际权重
        self.log_physics_weight = nn.Parameter(torch.log(torch.tensor(1.0)))  # 物理损失的权重，初始为1.0
        # self.log_consistency_weight = nn.Parameter(torch.log(torch.tensor(1.0)))  # 源一致性约束的权重，初始为1.0

        # 使用自适应尺度计算
        self.helmholtz_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=fs,
            duration=duration,
            freq_low=80,
            freq_high=2000,
            num_scales=50
        )
        # 自适应权重
        self.freq_weight = nn.Parameter(torch.tensor(1.0))
        self.time_weight = nn.Parameter(torch.tensor(1.0))

        self.wave_equation = TimeDomainWaveEquationLoss()

    def decode_to_freq(self, h):
        """从隐藏状态h解码到频域特征"""
        # h形状: [batch, time, hidden_dim] 或 [batch, hidden_dim]
        if len(h.shape) == 2:
            # 单个时间步: [batch, hidden_dim] -> [batch, 1, hidden_dim]
            h = h.unsqueeze(1)

        decoded = self.decoder(h)  # 解码
        freq_features = self.output_proj(decoded)  # 频域特征   [batch, time, freq*2]
        # 如果输入是单个时间步，去掉时间维度
        if freq_features.shape[1] == 1:
            freq_features = freq_features.squeeze(1)  # [batch, freq*2]

        return freq_features

    def to_complex_spectrum(self, freq_features):
        """将实部+虚部特征转换为复数频谱"""
        batch_size = freq_features.shape[0]
        freq_dim = self.helmholtz_pinn.n_fft // 2 + 1
        # 重塑为 [batch, freq, 2] 并转换为复数
        # 检查输入形状
        if len(freq_features.shape) == 2:
            # 单个时间步: [batch, freq*2] -> [batch, freq, 2] -> [batch, freq]
            freq_real = freq_features.view(batch_size, freq_dim, 2).contiguous()
            freq_complex = torch.view_as_complex(freq_real)
            # 添加时间维度以匹配亥姆霍兹约束的期望输入
            freq_complex = freq_complex.unsqueeze(-1)  # [batch, freq, 1]
        else:
            # 多个时间步: [batch, time, freq*2] -> [batch, time, freq, 2] -> [batch, freq, time]
            time_steps = freq_features.shape[1]

            freq_real = freq_features.view(batch_size, time_steps, freq_dim, 2).contiguous()
            freq_complex = torch.view_as_complex(freq_real)
            freq_complex = freq_complex.permute(0, 2, 1)  # [batch, freq, time]

        return freq_complex

    def generate_masks(self, freq_features):
        """从频域特征生成两个掩码"""
        batch_size, time_steps, _ = freq_features.shape
        freq_dim = self.helmholtz_pinn.n_fft // 2 + 1

        # 将特征重塑为 [batch, time, freq, 4]
        freq_reshaped = freq_features.view(batch_size, time_steps, freq_dim, 4)

        # 分离实部和虚部
        mask1_real = freq_reshaped[:, :, :, 0]
        mask1_imag = freq_reshaped[:, :, :, 1]
        mask2_real = freq_reshaped[:, :, :, 2]
        mask2_imag = freq_reshaped[:, :, :, 3]

        # 使用sigmoid确保掩码值在[0,1]范围内
        mask1_real = torch.sigmoid(mask1_real)
        mask1_imag = torch.sigmoid(mask1_imag)
        mask2_real = torch.sigmoid(mask2_real)
        mask2_imag = torch.sigmoid(mask2_imag)

        # 组合成复数掩码
        mask1 = torch.complex(mask1_real, mask1_imag)
        mask2 = torch.complex(mask2_real, mask2_imag)



        return mask1, mask2


    def forward(self, t, h,mix):
        base_dynamics = self.base_ode_func(t, h)
        return base_dynamics


class mulmlp1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.GELU(),
            nn.Dropout(0.1),

        )

        self.mlp0 = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim * 2),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=7, padding=3),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.mlp1 = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim * 2),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.Conv1d(hidden_dim* 2, hidden_dim * 2, kernel_size=5, padding=2),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),

        )

        self.mlp2 = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim * 2),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.Linear(hidden_dim * 2, hidden_dim)
        )

        self.mlp3 = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim * 2),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
            Lambda(lambda x: x.transpose(1, 2)),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.Linear(hidden_dim * 2, hidden_dim)
        )



        self.output_proj = nn.Sequential(

            nn.Linear(hidden_dim * 2,  output_dim),
        )

        self.residual_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        identity = x
        x = self.input_proj(x)

        x0 = self.mlp0(x)
        x1 = self.mlp1(x0)
        x2 = self.mlp2(x1)
        x3 = self.mlp3(x2)
        x = x0 + x1 + x2 + x3

        x = self.output_proj(x)+self.residual_proj(identity)
        return x


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class HelmholtzPINNODESeparator(nn.Module):
    def __init__(self, hidden_dim=128, sample_rate=8000, use_adaptive_alf=True,
                 constraint_weight=0.1, default_duration=2.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.sample_rate = sample_rate
        self.use_adaptive_alf = use_adaptive_alf
        self.default_duration = default_duration

        # 物理约束模块
        self.helmholtz_pinn = FrequencyDomainHelmholtzPINN(sample_rate=sample_rate)
        freq_dim = self.helmholtz_pinn.n_fft // 2 + 1  # 频率点数

        # 编码器（增强版：增加深度+残差连接+LayerNorm）
        self.encoder = mulmlp1(freq_dim * 2, hidden_dim, hidden_dim)

        # 解码器（增强版：深度+残差+LayerNorm）
        self.decoder = mulmlp1(hidden_dim, hidden_dim, hidden_dim)

        # 输出投影层（增强版：捕捉高频特征交互）
        self.output_proj = mulmlp1(hidden_dim, hidden_dim, freq_dim * 4)


        # 构建基础ODE函数
        self.base_ode_func = ScoreEstimator()


        self.ode_func = PhysicsConstrainedODEFunction(
            base_ode_func=self.base_ode_func,
            decoder=self.decoder,
            output_proj=self.output_proj,
            helmholtz_pinn=self.helmholtz_pinn,
            fs=sample_rate,
            duration=default_duration,
            constraint_weight=constraint_weight
        )

        # 积分器
        if use_adaptive_alf:
            self.alf_integrator = HybridALFIntegrator(
                f_theta=self.ode_func
            )


        else:
            self.ode_function = ODEFunction(self.ode_func)

    def generate_masks(self, freq_features):
        """从频域特征生成两个掩码"""
        batch_size, time_steps, _ = freq_features.shape
        freq_dim = self.helmholtz_pinn.n_fft // 2 + 1

        # 将特征重塑为 [batch, time, freq, 4]
        freq_reshaped = freq_features.view(batch_size, time_steps, freq_dim, 4)

        # 分离实部和虚部
        mask1_real = freq_reshaped[:, :, :, 0]
        mask1_imag = freq_reshaped[:, :, :, 1]
        mask2_real = freq_reshaped[:, :, :, 2]
        mask2_imag = freq_reshaped[:, :, :, 3]

        # 使用sigmoid确保掩码值在[0,1]范围内
        mask1_real = torch.sigmoid(mask1_real)
        mask1_imag = torch.sigmoid(mask1_imag)
        mask2_real = torch.sigmoid(mask2_real)
        mask2_imag = torch.sigmoid(mask2_imag)

        # 组合成复数掩码
        mask1 = torch.complex(mask1_real, mask1_imag)
        mask2 = torch.complex(mask2_real, mask2_imag)



        return mask1, mask2

    def forward(self, x, t_grid):
        batch_size, seq_len, _ = x.shape

        # 计算实际时长
        actual_duration = seq_len / self.sample_rate

        # 如果时长与默认值差异较大，更新尺度
        if abs(actual_duration - self.default_duration) > 0.1:
            self.ode_func.helmholtz_loss.duration = actual_duration

        # 时域→频域转换
        x_waveform = x.squeeze(-1)
        # === 添加填充以确保ISTFT长度匹配 ===
        orig_len = x_waveform.shape[-1]
        n_fft = self.helmholtz_pinn.n_fft
        hop_length = self.helmholtz_pinn.hop_length
        win_length = self.helmholtz_pinn.win_length

        # 计算确保ISTFT输出长度的最小填充长度
        T_padded = (orig_len - win_length + hop_length - 1) // hop_length + 1  # 向上取整计算帧数
        L_padded = (T_padded - 1) * hop_length + win_length  # 计算填充后的目标长度

        # 填充波形（仅在需要时）
        if L_padded > orig_len:
            pad_amount = L_padded - orig_len
            x_waveform = F.pad(x_waveform, (0, pad_amount))  # 在末尾填充零
        # === 填充结束 ===

        U = self.helmholtz_pinn.stft(x_waveform)
        U_real = torch.view_as_real(U)
        U_flat = U_real.permute(0, 2, 1, 3).flatten(2)  # [batch, time, freq*2]

        # 缓存频域特征供ODE使用（建立时间尺度联系）
        self.ode_func.current_U_flat = U

        # 编码
        h0 = self.encoder(U_flat)  # [batch, time, hidden_dim]

        # ODE演化（带物理约束）- 只使用一个ODE
        if self.use_adaptive_alf:
            t0, T = t_grid[0], t_grid[-1]
            h_trajectory = self.alf_integrator(h0, t0, T,x_waveform)


        # 解码与掩码生成
        decoded = self.decoder(h_trajectory)
        freq_features = self.output_proj(decoded)  # [batch, time, freq*4]

        # 生成两个掩码
        mask1, mask2 = self.generate_masks(freq_features)

        # 应用掩码到原始混合频谱
        sep1_freq = U * mask1.permute(0, 2, 1)  # [batch, freq, time]
        sep2_freq = U * mask2.permute(0, 2, 1)  # [batch, freq, time]

        # 频域→时域转换
        sep1 = self.helmholtz_pinn.istft(sep1_freq)[..., :orig_len]  # 裁剪到原始长度
        sep2 = self.helmholtz_pinn.istft(sep2_freq)[..., :orig_len]  # 裁剪到原始长度

        # 清理缓存
        self.ode_func.current_U_flat = None

        # 调整输出形状
        sep1 = sep1.unsqueeze(1)
        sep2 = sep2.unsqueeze(1)
        sep1 = sep1.requires_grad_(True)
        sep2 = sep2.requires_grad_(True)
        if not self.training:
            sep1=sep1.detach()
            sep2=sep2.detach()

        return sep1, sep2



    @classmethod
    def load_model(cls, path):
        package = torch.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model

    @classmethod
    def load_model_from_package(cls, package):
        model = cls(

        )
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # 保存基础配置参数

            # 保存当前模型状态
            'state_dict': model.state_dict(),

            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }

        # 可选训练状态保存
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss

        return package




class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = avg_out + max_out
        return x * out.view(b, c, 1, 1)


class ConcatChannels(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5x5 = nn.Conv2d(1, 16, (5, 5), padding=2)  # 输入通道4，输出16
        self.conv3x3 = nn.Conv2d(1, 16, (3, 3), padding=1)  # 输入通道4，输出16

    def forward(self, x):
        out5 = self.conv5x5(x)  # [B,16,F,T]
        out3 = self.conv3x3(x)  # [B,16,F,T]
        return torch.cat([out5, out3], dim=1)  # [B,32,F,T]


class ScoreEstimator(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Conv2d(1, 16, 1),  # 空间注意力
            nn.ReLU(),
            ChannelAttention(16),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )

        # [B, 2*num_speakers, freq_bins, time_frames]
        self.net = nn.Sequential(
            ConcatChannels(),
            nn.Conv2d(32, 32, (3, 3), padding=1),  # 保持[H,W]
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, (1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, (5, 5), padding=2)
        )

    def forward(self, t, input):
        attn_input = input.unsqueeze(1)
        spatial_attn = self.attention(attn_input)  # [B*C, 1, F, T]
        attended = attn_input * spatial_attn + spatial_attn

        base_out = self.net(attended).view(attn_input.shape) + attn_input

        return base_out.squeeze(1)


class ODEFunction(nn.Module):
    def __init__(self, ode_func):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, t, h):
        return self.ode_func(t, h)


def si_snr_loss(pred, target):
    """
    计算SI-SNR损失
    pred: 预测信号，形状 [batch, 1, time]
    target: 目标信号，形状 [batch, 1, time]
    """
    # 移除批次和通道维度，保留 [batch, time]
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    # 确保pred和target长度一致，取最小长度
    min_len = min(pred.shape[-1], target.shape[-1])
    if pred.shape[-1] != min_len:
        pred = pred[..., :min_len]
    if target.shape[-1] != min_len:
        target = target[..., :min_len]

    # 移除直流分量
    pred = pred - torch.mean(pred, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)

    # 计算目标信号能量
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)

    # 计算尺度因子
    scale = torch.sum(pred * target, dim=-1, keepdim=True) / (target_energy + 1e-8)

    # 尺度对齐后的目标信号
    target_scaled = scale * target

    # 计算噪声（残差）
    noise = pred - target_scaled

    # 计算SI-SNR (dB)
    snr = 10 * torch.log10(torch.sum(target_scaled ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8)

    # 返回负SNR作为损失（因为我们要最小化损失）
    return -torch.mean(snr)


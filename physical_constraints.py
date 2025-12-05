# physical_constraints.py
import torch
import torch.nn as nn
import numpy as np

from hmhzeasyodeold1111 import OptimizedMorletCWTHelmholtzLoss, TimeDomainWaveEquationLoss


class PhysicsRegularizer(nn.Module):
    """物理约束正则化器 - 真正的正则化实现"""

    def __init__(self, sample_rate=8000, c=343.0, constraint_type="both"):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c
        self.constraint_type = constraint_type  # "helmholtz", "wave", "both"

        # 初始化物理约束模块
        self.helmholtz_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate,
            duration=2.0,
            freq_low=80,
            freq_high=2000,
            num_scales=50
        )
        self.wave_equation_loss = TimeDomainWaveEquationLoss()

        # 自适应正则化权重
        self.helmholtz_weight = nn.Parameter(torch.tensor(0.01))
        self.wave_weight = nn.Parameter(torch.tensor(0.01))

        # 正则化强度控制
        self.regularization_strength = nn.Parameter(torch.tensor(1.0))

    def compute_physics_constraints(self, separated_sources):
        """计算物理约束违反程度"""
        physics_violations = []

        for source in separated_sources:
            source_violation = 0.0
            count = 0

            # 亥姆霍兹约束违反
            if self.constraint_type in ["helmholtz", "both"]:
                freq_loss, cwt_coeffs, scales = self.helmholtz_loss(source)
                helmholtz_violation = torch.exp(self.helmholtz_weight) * freq_loss
                source_violation += helmholtz_violation
                count += 1

            # 波动方程约束违反
            if self.constraint_type in ["wave", "both"]:
                time_loss = self.wave_equation_loss(cwt_coeffs, scales)
                wave_violation = torch.exp(self.wave_weight) * time_loss
                source_violation += wave_violation
                count += 1

            if count > 0:
                physics_violations.append(source_violation / count)

        return torch.stack(physics_violations) if physics_violations else torch.tensor(0.0)

    def forward(self, model_parameters, separated_sources):
        """
        真正的正则化前向传播
        Args:
            model_parameters: 模型参数，用于计算参数正则化
            separated_sources: 分离的语音信号，用于计算物理约束违反
        """
        # 1. 物理约束违反惩罚
        physics_violations = []
        for source in separated_sources:
            # 计算物理约束违反
            freq_loss, cwt_coeffs, scales = self.helmholtz_loss(source.unsqueeze(0))
            time_loss = self.wave_equation_loss(cwt_coeffs, scales)

            physics_violation = freq_loss + time_loss
            physics_violations.append(physics_violation)

        physics_penalty = torch.mean(torch.stack(physics_violations))

        # physics_violations = self.compute_physics_constraints(separated_sources)
        # physics_penalty = torch.mean(physics_violations)

        # 2. 参数正则化（可选，防止物理约束导致参数爆炸）
        param_regularization = 0.0
        for param in model_parameters:
            if param.requires_grad:
                param_regularization += torch.norm(param, p=2) ** 2

        # 3. 组合正则化项
        total_regularization = (
                torch.exp(self.regularization_strength) * physics_penalty +
                0.001 * param_regularization  # 较小的参数正则化
        )

        return total_regularization


class OptimizedAdaptiveBoundaryFunction(nn.Module):
    """
    向量化优化的自适应边界函数
    """

    def __init__(self, sample_rate, hidden_dim=128):
        super().__init__()
        self.sample_rate = sample_rate

        # 物理特征提取 (频率特征、能量分布等)
        self.physical_feature_extractor = nn.Sequential(
            # STFT特征提取
            nn.Conv1d(1, 64, kernel_size=512, stride=256, padding=256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, hidden_dim),
            nn.ReLU()
        )

        # 基底生成网络
        self.baseline_generator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # 物理特征 + 时间
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # 信号特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=256, stride=128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, hidden_dim),
            nn.ReLU()
        )

        # 边界函数生成器
        self.boundary_generator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, signal,mix=None):
        batch_size, seq_len = signal.shape

        # 如果有混合信号，基于混合信号生成基底
        if mix is not None:
            # 提取混合信号的物理特征
            physical_features = self.physical_feature_extractor(mix.unsqueeze(1))
        else:
            # 基于分离信号自身
            physical_features = self.physical_feature_extractor(s_hat.unsqueeze(1))

        # 1. 提取信号的全局特征 [batch_size, hidden_dim]
        signal_features = self.feature_extractor(signal.unsqueeze(1))

        # 2. 生成时间坐标 [batch_size, seq_len, 1]
        t = torch.linspace(0, 1, seq_len, device=signal.device)
        t_batch = t.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)

        # 3. 将信号特征广播到每个时间点 [batch_size, seq_len, hidden_dim]
        features_expanded = signal_features.unsqueeze(1).expand(-1, seq_len, -1)

        # 4. 组合特征和时间 [batch_size, seq_len, hidden_dim + 1]
        features_with_time = torch.cat([features_expanded, t_batch], dim=-1)

        # 5. 【关键】向量化计算
        # 将 3D 张量展平为 2D，以适应 Linear 层
        batch_T, dim = features_with_time.shape[1], features_with_time.shape[2]
        features_flat = features_with_time.view(-1, dim)  # [batch_size * seq_len, hidden_dim + 1]

        # 一次性计算所有时间点的边界值
        g_flat = self.boundary_generator(features_flat)  # [batch_size * seq_len, 1]

        # 重新塑形回原始形状
        g = g_flat.view(batch_size, seq_len)  # [batch_size, seq_len]

        return g


class AdaptivePhysicsRegularizer(nn.Module):
    """自适应物理正则化器 - 更高级的实现"""

    def __init__(self, sample_rate=8000, c=343.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c

        # 物理约束模块
        self.helmholtz_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50
        )
        self.wave_equation_loss = TimeDomainWaveEquationLoss()

        # 自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2, 32),  # 输入: [数据损失, 物理违反程度]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出: [helmholtz_weight, wave_weight]
            nn.Softplus()  # 确保权重为正
        )

        # 历史记录
        self.register_buffer('data_loss_history', torch.zeros(100))
        self.register_buffer('physics_violation_history', torch.zeros(100))
        self.history_index = 0

    def update_history(self, data_loss, physics_violation):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss
        self.physics_violation_history[self.history_index] = physics_violation
        self.history_index = (self.history_index + 1) % 100

    def compute_adaptive_weights(self, current_data_loss, current_physics_violation):
        """计算自适应权重"""
        # 更新历史
        self.update_history(current_data_loss, current_physics_violation)

        # 计算相对违反程度
        avg_data_loss = self.data_loss_history.mean()
        avg_physics_violation = self.physics_violation_history.mean()

        if avg_data_loss > 0 and avg_physics_violation > 0:
            relative_violation = current_physics_violation / avg_physics_violation
            relative_data_loss = current_data_loss / avg_data_loss

            # 获取模型设备
            device = next(self.adaptive_weight_net.parameters()).device

            # 自适应调整权重 - 确保张量在正确的设备上
            input_features = torch.tensor([relative_data_loss, relative_violation],
                                          dtype=torch.float32, device=device)
            weights = self.adaptive_weight_net(input_features)
            helmholtz_weight, wave_weight = weights[0], weights[1]
        else:
            # 获取模型设备
            device = next(self.adaptive_weight_net.parameters()).device
            helmholtz_weight, wave_weight = torch.tensor(0.1, device=device), torch.tensor(0.1, device=device)

        return helmholtz_weight, wave_weight


    # def compute_adaptive_weights(self, current_data_loss, current_physics_violation):
    #     """计算自适应权重"""
    #     # 更新历史
    #     self.update_history(current_data_loss, current_physics_violation)
    #
    #     # 计算相对违反程度
    #     avg_data_loss = self.data_loss_history.mean()
    #     avg_physics_violation = self.physics_violation_history.mean()
    #
    #     if avg_data_loss > 0 and avg_physics_violation > 0:
    #         relative_violation = current_physics_violation / avg_physics_violation
    #         relative_data_loss = current_data_loss / avg_data_loss
    #
    #         # 自适应调整权重
    #         input_features = torch.tensor([relative_data_loss, relative_violation],
    #                                       dtype=torch.float32)
    #         weights = self.adaptive_weight_net(input_features)
    #         helmholtz_weight, wave_weight = weights[0], weights[1]
    #     else:
    #         helmholtz_weight, wave_weight = torch.tensor(0.1), torch.tensor(0.1)
    #
    #     return helmholtz_weight, wave_weight

    def forward(self, model_parameters, separated_sources, current_data_loss):
        """自适应正则化前向传播"""
        physics_violations = []

        for source in separated_sources:
            # 计算物理约束违反
            freq_loss, cwt_coeffs, scales = self.helmholtz_loss(source.unsqueeze(0))
            time_loss = self.wave_equation_loss(cwt_coeffs, scales)

            physics_violation = freq_loss + time_loss
            physics_violations.append(physics_violation)

        avg_physics_violation = torch.mean(torch.stack(physics_violations))

        # 计算自适应权重
        helmholtz_weight, wave_weight = self.compute_adaptive_weights(
            current_data_loss, avg_physics_violation
        )

        # 重新计算加权物理违反
        weighted_physics_violation = 0.0
        for source in separated_sources:
            freq_loss, cwt_coeffs, scales = self.helmholtz_loss(source.unsqueeze(0))
            time_loss = self.wave_equation_loss(cwt_coeffs, scales)
            weighted_physics_violation += helmholtz_weight * freq_loss + wave_weight * time_loss

        weighted_physics_violation /= len(separated_sources)

        return weighted_physics_violation


class SpeechSeparationHardConstraint(nn.Module):
    """
    专门为语音分离设计的硬约束物理正则器
    对每个分离的说话人输出应用物理修正
    """

    def __init__(self, sample_rate=8000, c=343.0, num_sources=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c
        self.num_sources = num_sources

        # 物理约束模块
        self.helmholtz_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50)
        self.wave_equation_loss = TimeDomainWaveEquationLoss()

        # 为每个说话人源创建独立的约束函数
        self.source_constraints = nn.ModuleList([
            SourceSpecificConstraint(sample_rate, source_idx=i)
            for i in range(num_sources)
        ])

        # 自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2 + num_sources, 64),  # 输入: [数据损失, 物理违反, 各源特征]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus()
        )

        # 历史记录
        self.register_buffer('data_loss_history', torch.zeros(100))
        self.register_buffer('physics_violation_history', torch.zeros(100))
        self.history_index = 0

    def forward(self, separated_sources_delta, mixture=None, current_data_loss=None):
        """
        对每个分离的说话人源应用硬约束

        Args:
            separated_sources_delta: list of tensors [源1_delta, 源2_delta, ...]
            mixture: 混合信号，用于参考
            current_data_loss: 当前数据损失
        """
        batch_size = separated_sources_delta[0].shape[0]
        physics_violations = []
        constrained_sources = []

        for i, delta in enumerate(separated_sources_delta):
            # 获取该源的约束函数
            constraint_func = self.source_constraints[i]

            # 应用源特定的硬约束
            constrained_source = constraint_func(delta, mixture)
            constrained_sources.append(constrained_source)

            # 计算物理违反
            physics_violation = self._compute_source_physics_violation(constrained_source, i)
            physics_violations.append(physics_violation)

        avg_physics_violation = torch.mean(torch.stack(physics_violations))

        # 自适应权重
        if current_data_loss is not None:
            # 计算源特征（能量分布）
            # source_features = self._compute_source_features(constrained_sources)
            source_features = self._compute_source_features(separated_sources_delta)

            weights = self.compute_adaptive_weights(current_data_loss, avg_physics_violation, source_features)
            w_helm, w_wave = weights[0], weights[1]
        else:
            device = avg_physics_violation.device
            w_helm, w_wave = torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

        return 0.5 * (w_helm + w_wave) * avg_physics_violation, constrained_sources

    def _compute_source_physics_violation(self, source, source_idx):
        """计算单个源的物理违反程度"""
        # 这里可以根据不同说话人特性调整物理约束
        freq_loss, cwt_coeffs, scales = self.helmholtz_loss(source)
        time_loss = self.wave_equation_loss(cwt_coeffs, scales)

        # 可以根据源特性调整权重
        return freq_loss + time_loss

    def _compute_source_features(self, sources):
        """计算源特征用于自适应权重"""
        features = []
        for source in sources:
            # 计算能量特征
            energy = torch.mean(source ** 2)
            # 计算频谱质心
            spectrum = torch.fft.fft(source, dim=-1)
            freqs = torch.fft.fftfreq(source.shape[-1], 1 / self.sample_rate)
            spectral_centroid = torch.sum(torch.abs(spectrum) * freqs.abs()) / torch.sum(torch.abs(spectrum))
            features.extend([energy, spectral_centroid])

        return torch.stack(features)

    def compute_adaptive_weights(self, current_data_loss, current_physics_violation, source_features):
        """计算自适应权重"""
        self.update_history(current_data_loss, current_physics_violation)

        avg_data = self.data_loss_history.mean()
        avg_phys = self.physics_violation_history.mean()

        device = next(self.adaptive_weight_net.parameters()).device

        if avg_data > 1e-8 and avg_phys > 1e-8:
            rel_data = current_data_loss / avg_data
            rel_phys = current_physics_violation / avg_phys

            # 组合所有特征
            input_features = torch.cat([
                torch.tensor([rel_data, rel_phys], device=device),
                source_features
            ])

            weights = self.adaptive_weight_net(input_features)
            return weights[0], weights[1]
        else:
            return torch.tensor(0.1, device=device), torch.tensor(0.1, device=device)

    def update_history(self, data_loss, physics_violation):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss
        self.physics_violation_history[self.history_index] = physics_violation
        self.history_index = (self.history_index + 1) % len(self.data_loss_history)


class SourceSpecificConstraint(nn.Module):
    """
    源特定约束 - 为每个说话人学习特定的约束函数
    """

    def __init__(self, sample_rate, source_idx, hidden_dim=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.source_idx = source_idx

        # 可学习的边界函数网络
        self.boundary_net = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 输入: 时间位置
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 初始解网络
        self.initial_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入: [时间, 混合信号特征]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # 初始化边界函数为合理的形状
        self._initialize_boundary_function()

    def _initialize_boundary_function(self):
        """初始化边界函数为淡入淡出形状"""
        with torch.no_grad():
            # 设置最后一层的偏置，使得在边界处输出较小
            self.boundary_net[-2].bias.data.fill_(-2.0)

    def forward(self, delta, mixture=None):
        """
        应用源特定约束: s = s0 + g * delta

        Args:
            delta: 网络输出的残差 [batch_size, seq_len]
            mixture: 混合信号 [batch_size, seq_len]
        """
        batch_size, seq_len = delta.shape

        # 生成时间坐标
        t = torch.linspace(0, 1, seq_len, device=delta.device).unsqueeze(0)  # [1, seq_len]

        # 学习边界函数 g(t)
        g = self.boundary_net(t.unsqueeze(-1)).squeeze(-1)  # [1, seq_len]
        g = g.expand(batch_size, -1)  # [batch_size, seq_len]

        # 学习初始解 s0
        if mixture is not None:
            # 使用混合信号信息
            mixture_features = self._extract_mixture_features(mixture)
            initial_input = torch.stack([t.expand(batch_size, -1), mixture_features], dim=-1)
        else:
            # 仅使用时间信息
            initial_input = t.expand(batch_size, -1).unsqueeze(-1)

        s0 = self.initial_net(initial_input).squeeze(-1)  # [batch_size, seq_len]

        # 应用硬约束
        constrained_signal = s0 + g * delta

        return constrained_signal

    def _extract_mixture_features(self, mixture):
        """从混合信号中提取特征"""
        # 简单的能量特征
        window_size = 256
        hop_size = 128

        features = []
        for i in range(0, mixture.shape[-1] - window_size, hop_size):
            window = mixture[..., i:i + window_size]
            energy = torch.mean(window ** 2, dim=-1)
            features.append(energy)

        # 平均所有窗口的特征
        if features:
            avg_features = torch.mean(torch.stack(features, dim=-1), dim=-1)
        else:
            avg_features = torch.mean(mixture ** 2, dim=-1)

        return avg_features.unsqueeze(-1).expand(-1, mixture.shape[-1])


class MultiScaleSpeechConstraint(nn.Module):
    """
    多尺度语音约束 - 在不同时间尺度应用约束
    """

    def __init__(self, sample_rate, scales=[256, 512, 1024]):
        super().__init__()
        self.sample_rate = sample_rate
        self.scales = scales

        # 为每个尺度创建约束
        self.scale_constraints = nn.ModuleList([
            ScaleSpecificConstraint(scale, sample_rate) for scale in scales
        ])

        # 尺度融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(len(scales), 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, delta, mixture=None):
        scale_outputs = []

        for constraint in self.scale_constraints:
            # 调整到特定尺度
            scaled_delta = self._rescale_signal(delta, constraint.scale)
            # 应用约束
            constrained = constraint(scaled_delta, mixture)
            # 计算该尺度的约束强度
            scale_strength = torch.mean(constrained ** 2)
            scale_outputs.append(scale_strength.unsqueeze(-1))

        # 融合多尺度约束
        scale_features = torch.cat(scale_outputs, dim=-1)
        fusion_weights = self.fusion_net(scale_features)

        # 应用融合后的约束
        final_constrained = torch.zeros_like(delta)
        for i, constraint in enumerate(self.scale_constraints):
            scaled_delta = self._rescale_signal(delta, constraint.scale)
            constrained = constraint(scaled_delta, mixture)
            # 上采样回原始尺度
            upsampled = self._rescale_signal(constrained, delta.shape[-1])
            final_constrained += fusion_weights[..., i].unsqueeze(-1) * upsampled

        return final_constrained / len(self.scale_constraints)

    def _rescale_signal(self, signal, target_length):
        """重新调整信号长度"""
        if signal.shape[-1] == target_length:
            return signal

        return F.interpolate(
            signal.unsqueeze(1),
            size=target_length,
            mode='linear',
            align_corners=False
        ).squeeze(1)


class ScaleSpecificConstraint(nn.Module):
    """尺度特定约束"""

    def __init__(self, scale, sample_rate):
        super().__init__()
        self.scale = scale
        self.sample_rate = sample_rate

        self.constraint_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )

    def forward(self, delta, mixture=None):
        seq_len = delta.shape[-1]
        t = torch.linspace(0, 1, seq_len, device=delta.device).unsqueeze(0)

        # 尺度特定的边界函数
        g = self.constraint_net(t.unsqueeze(-1)).squeeze(-1)
        g = g.expand_as(delta)

        return g * delta


# 使用示例
def create_speech_separation_system(num_sources=2, sample_rate=8000):
    """创建完整的语音分离系统"""

    # 创建硬约束正则器
    hard_constraint_regularizer = SpeechSeparationHardConstraint(
        sample_rate=sample_rate,
        num_sources=num_sources
    )

    return hard_constraint_regularizer


# 训练循环示例
def training_step_with_hard_constraints(model, mixture, targets, regularizer):
    """
    使用硬约束的训练步骤
    """
    # 模型前向传播（输出残差delta）
    separated_deltas = model(mixture)  # [source1_delta, source2_delta, ...]

    # 应用硬约束并计算物理正则化损失
    physics_loss, constrained_sources = regularizer(
        separated_deltas, mixture, current_data_loss=None
    )

    # 计算数据损失（使用约束后的信号）
    data_loss = 0.0
    for constrained, target in zip(constrained_sources, targets):
        data_loss += F.mse_loss(constrained, target)
    data_loss /= len(constrained_sources)

    # 总损失
    total_loss = data_loss + 0.1 * physics_loss

    return {
        'total_loss': total_loss,
        'data_loss': data_loss,
        'physics_loss': physics_loss,
        'constrained_sources': constrained_sources
    }


import torch
import torch.nn as nn


# 假设 OptimizedMorletCWTHelmholtzLoss 和 TimeDomainWaveEquationLoss 已经定义

class PhysicsAwarePostProcessing(nn.Module):
    """
    物理感知后处理层 - 带有自适应权重
    """

    def __init__(self, sample_rate=8000, num_sources=2, constraint_type="boundary", history_len=100):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_sources = num_sources
        self.constraint_type = constraint_type

        # 物理约束模块
        self.helmholtz_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50
        )
        self.wave_equation_loss = TimeDomainWaveEquationLoss()

        # 为每个源创建约束函数
        self.constraint_functions = nn.ModuleList([
            SignalConstraintFunction(sample_rate, constraint_type, source_idx=i)
            for i in range(num_sources)
        ])

        # 【新增】自适应权重网络
        # 输入: [相对数据损失, 相对物理损失]
        # 输出: 物理损失的权重 lambda_physics
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()  # 确保权重为正
        )

        # 【新增】历史记录
        self.history_len = history_len
        self.register_buffer('data_loss_history', torch.zeros(history_len))
        self.register_buffer('physics_loss_history', torch.zeros(history_len))
        self.history_index = 0

    def forward(self, separated_sources,mix, current_data_loss=None, compute_loss=True):
        """
        对初步分离的信号应用物理约束

        Args:
            separated_sources: list of tensors, 每个是 [batch_size, seq_len]
            current_data_loss: 当前批次的数据损失 (标量tensor)，用于自适应权重
            compute_loss: 是否计算物理损失
        """
        constrained_sources = []
        physics_losses = []

        for i, s_hat in enumerate(separated_sources):
            # 应用物理约束
            constraint_func = self.constraint_functions[i]
            s_constrained = constraint_func(s_hat.unsqueeze(0),mix)
            constrained_sources.append(s_constrained)

            # 计算物理损失
            if compute_loss:
                physics_loss = self._compute_physics_loss(s_constrained)
                physics_losses.append(physics_loss)

        if compute_loss and physics_losses:
            raw_physics_loss = torch.mean(torch.stack(physics_losses))

            # 【核心】计算自适应权重
            if current_data_loss is not None:
                lambda_physics = self._compute_adaptive_weight(current_data_loss, raw_physics_loss)
                weighted_physics_loss = lambda_physics * raw_physics_loss
            else:
                # 如果没有提供数据损失（如验证阶段），使用一个小的默认权重
                device = raw_physics_loss.device
                lambda_physics = torch.tensor(0.1, device=device)
                weighted_physics_loss = lambda_physics * raw_physics_loss
        else:
            weighted_physics_loss = torch.tensor(0.0, device=separated_sources[0].device)
            raw_physics_loss = weighted_physics_loss
            lambda_physics = torch.tensor(0.0, device=separated_sources[0].device)

        return constrained_sources, weighted_physics_loss, raw_physics_loss, lambda_physics

    def _compute_physics_loss(self, signal):
        """计算单个信号的物理约束违反程度"""
        freq_loss, cwt_coeffs, scales = self.helmholtz_loss(signal)
        time_loss = self.wave_equation_loss(cwt_coeffs, scales)
        return freq_loss + time_loss

    def _update_history(self, data_loss, physics_loss):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss.detach()
        self.physics_loss_history[self.history_index] = physics_loss.detach()
        self.history_index = (self.history_index + 1) % self.history_len

    def _compute_adaptive_weight(self, current_data_loss, current_physics_loss):
        """计算自适应权重"""
        # 更新历史
        self._update_history(current_data_loss, current_physics_loss)

        # 计算历史平均值
        avg_data_loss = self.data_loss_history.mean()
        avg_physics_loss = self.physics_loss_history.mean()

        device = current_data_loss.device

        # 避免除以零
        if avg_data_loss > 1e-8 and avg_physics_loss > 1e-8:
            # 计算相对损失
            relative_data_loss = current_data_loss / avg_data_loss
            relative_physics_loss = current_physics_loss / avg_physics_loss

            # 输入特征
            input_features = torch.tensor([relative_data_loss, relative_physics_loss],
                                          dtype=torch.float32, device=device)

            # 通过网络计算权重
            lambda_physics = self.adaptive_weight_net(input_features).squeeze(0)
        else:
            # 历史数据不足，使用默认权重
            lambda_physics = torch.tensor(0.1, device=device)

        return lambda_physics





class PhysicsDrivenConstraintFunction(nn.Module):
    def __init__(self, sample_rate, source_idx, hidden_dim=64):
        super().__init__()

        # 物理特征提取 (频率特征、能量分布等)
        self.physical_feature_extractor = nn.Sequential(
            # STFT特征提取
            nn.Conv1d(1, 64, kernel_size=512, stride=256, padding=256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Flatten(),
            nn.Linear(64 * 32, hidden_dim),
            nn.ReLU()
        )

        # 基底生成网络
        self.baseline_generator = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # 物理特征 + 时间
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

    def forward(self, s_hat, mixture=None):
        batch_size, seq_len = s_hat.shape

        # 如果有混合信号，基于混合信号生成基底
        if mixture is not None:
            # 提取混合信号的物理特征
            physical_features = self.physical_feature_extractor(mixture.unsqueeze(1))
        else:
            # 基于分离信号自身
            physical_features = self.physical_feature_extractor(s_hat.unsqueeze(1))

        # 生成自适应基底
        t = torch.linspace(0, 1, seq_len, device=s_hat.device)
        s0 = self._generate_adaptive_baseline(physical_features, t, batch_size, seq_len)

        # 应用约束
        g = self.boundary_net(t.unsqueeze(-1)).squeeze(-1).expand(batch_size, -1)
        return s0 + g * s_hat


class SignalConstraintFunction(nn.Module):
    """
    信号约束函数 - 实现 s_constrained = s0 + g * ŝ
    """

    def __init__(self, sample_rate, constraint_type, source_idx, hidden_dim=64):
        super().__init__()
        self.sample_rate = sample_rate
        self.constraint_type = constraint_type
        self.source_idx = source_idx

        # 边界函数网络 g(t)
        self.boundary_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 基底信号网络 s0(t)
        self.baseline_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )

        # 初始化边界函数
        self._initialize_boundary_function()

    def _initialize_boundary_function(self):
        """初始化边界函数为淡入淡出形状"""
        with torch.no_grad():
            # 设置偏置使得在边界处输出较小
            self.boundary_net[-2].bias.data.fill_(-2.0)

    def forward(self, s_hat,mixf):
        """
        应用约束: s_constrained = s0 + g * ŝ

        Args:
            s_hat: 初步分离的信号 [batch_size, seq_len]
        """
        batch_size, seq_len = s_hat.shape
        mixf=mixf[:,:seq_len]
        # 生成时间坐标 [0, 1]
        t = torch.linspace(0, 1, seq_len, device=s_hat.device).unsqueeze(0)  # [1, seq_len]
        t_expanded = t.expand(batch_size, -1)  # [batch_size, seq_len]

        # 计算边界函数 g(t)
        g = self.boundary_net(t.unsqueeze(-1)).squeeze(-1)  # [1, seq_len]
        g = g.expand(batch_size, -1)  # [batch_size, seq_len]

        # 计算基底信号 s0(t)
        s0 = self.baseline_net(mixf.unsqueeze(-1)).squeeze(-1)  # [1, seq_len]
        s0 = s0.expand(batch_size, -1)  # [batch_size, seq_len]

        # 应用硬约束
        s_constrained = s0 + g * s_hat

        return s_constrained


import torch
import torch.nn as nn
import numpy as np
from hmhzeasyodeold1111 import OptimizedMorletCWTHelmholtzLoss, TimeDomainWaveEquationLoss


class ScaleTimeGeometricRegularizer(nn.Module):
    """尺度-时间几何正则化器 - 基于仿射几何的实现"""

    def __init__(self, sample_rate=8000, c=343.0, constraint_type="both"):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c
        self.constraint_type = constraint_type  # "scale_laplacian", "scale_time", "both"

        # 初始化几何约束模块
        self.scale_laplacian_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate,
            duration=2.0,
            freq_low=80,
            freq_high=2000,
            num_scales=50
        )
        self.scale_time_evolution_loss = TimeDomainWaveEquationLoss()

        # 自适应几何正则化权重
        self.scale_laplacian_weight = nn.Parameter(torch.tensor(0.01))
        self.scale_time_evolution_weight = nn.Parameter(torch.tensor(0.01))

        # 几何正则化强度控制
        self.geometric_regularization_strength = nn.Parameter(torch.tensor(1.0))

    def compute_geometric_constraints(self, separated_sources):
        """计算几何结构违反程度"""
        geometric_violations = []

        for source in separated_sources:
            source_violation = 0.0
            count = 0

            # 尺度-拉普拉斯约束违反
            if self.constraint_type in ["scale_laplacian", "both"]:
                scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(source)
                weighted_violation = torch.exp(self.scale_laplacian_weight) * scale_laplacian_violation
                source_violation += weighted_violation
                count += 1

            # 尺度-时间演化约束违反
            if self.constraint_type in ["scale_time", "both"]:
                scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
                weighted_violation = torch.exp(self.scale_time_evolution_weight) * scale_time_violation
                source_violation += weighted_violation
                count += 1

            if count > 0:
                geometric_violations.append(source_violation / count)

        return torch.stack(geometric_violations) if geometric_violations else torch.tensor(0.0)

    def forward(self, model_parameters, separated_sources):
        """几何正则化前向传播"""
        # 1. 几何约束违反惩罚
        geometric_violations = []
        for source in separated_sources:
            scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(source.unsqueeze(0))
            scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
            geometric_violation = scale_laplacian_violation + scale_time_violation
            geometric_violations.append(geometric_violation)

        geometric_penalty = torch.mean(torch.stack(geometric_violations))

        # 2. 参数正则化（防止几何约束导致参数爆炸）
        param_regularization = 0.0
        for param in model_parameters:
            if param.requires_grad:
                param_regularization += torch.norm(param, p=2) ** 2

        # 3. 组合几何正则化项
        total_geometric_regularization = (
            torch.exp(self.geometric_regularization_strength) * geometric_penalty +
            0.001 * param_regularization
        )

        return total_geometric_regularization


class AdaptiveGeometricRegularizer(nn.Module):
    """自适应几何正则化器"""

    def __init__(self, sample_rate=8000, c=343.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c

        # 几何约束模块
        self.scale_laplacian_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50
        )
        self.scale_time_evolution_loss = TimeDomainWaveEquationLoss()

        # 自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2, 32),  # 输入: [数据损失, 几何违反程度]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 输出: [scale_laplacian_weight, scale_time_evolution_weight]
            nn.Softplus()
        )

        # 历史记录
        self.register_buffer('data_loss_history', torch.zeros(100))
        self.register_buffer('geometric_violation_history', torch.zeros(100))
        self.history_index = 0

    def update_history(self, data_loss, geometric_violation):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss
        self.geometric_violation_history[self.history_index] = geometric_violation
        self.history_index = (self.history_index + 1) % 100

    def compute_adaptive_weights(self, current_data_loss, current_geometric_violation):
        """计算自适应几何权重"""
        self.update_history(current_data_loss, current_geometric_violation)
        avg_data_loss = self.data_loss_history.mean()
        avg_geometric_violation = self.geometric_violation_history.mean()

        device = next(self.adaptive_weight_net.parameters()).device

        if avg_data_loss > 0 and avg_geometric_violation > 0:
            relative_violation = current_geometric_violation / avg_geometric_violation
            relative_data_loss = current_data_loss / avg_data_loss
            input_features = torch.tensor([relative_data_loss, relative_violation],
                                          dtype=torch.float32, device=device)
            weights = self.adaptive_weight_net(input_features)
            scale_laplacian_weight, scale_time_evolution_weight = weights[0], weights[1]
        else:
            device = next(self.adaptive_weight_net.parameters()).device
            scale_laplacian_weight = torch.tensor(0.1, device=device)
            scale_time_evolution_weight = torch.tensor(0.1, device=device)

        return scale_laplacian_weight, scale_time_evolution_weight

    def forward(self, model_parameters, separated_sources, current_data_loss):
        """自适应几何正则化前向传播"""
        geometric_violations = []

        for source in separated_sources:
            scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(source.unsqueeze(0))
            scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
            geometric_violation = scale_laplacian_violation + scale_time_violation
            geometric_violations.append(geometric_violation)

        avg_geometric_violation = torch.mean(torch.stack(geometric_violations))

        # 计算自适应几何权重
        scale_laplacian_weight, scale_time_evolution_weight = self.compute_adaptive_weights(
            current_data_loss, avg_geometric_violation
        )

        # 重新计算加权几何违反
        weighted_geometric_violation = 0.0
        for source in separated_sources:
            scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(source.unsqueeze(0))
            scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
            weighted_geometric_violation += scale_laplacian_weight * scale_laplacian_violation + \
                                           scale_time_evolution_weight * scale_time_violation

        weighted_geometric_violation /= len(separated_sources)

        return weighted_geometric_violation


class ScaleTimeHardConstraint(nn.Module):
    """尺度-时间硬约束 - 为每个分离源应用几何修正"""

    def __init__(self, sample_rate=8000, c=343.0, num_sources=2):
        super().__init__()
        self.sample_rate = sample_rate
        self.c = c
        self.num_sources = num_sources

        # 几何约束模块
        self.scale_laplacian_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50
        )
        self.scale_time_evolution_loss = TimeDomainWaveEquationLoss()

        # 为每个源创建独立的几何约束函数
        self.source_constraints = nn.ModuleList([
            SourceSpecificGeometricConstraint(sample_rate, source_idx=i)
            for i in range(num_sources)
        ])

        # 自适应几何权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2 + num_sources, 64),  # 输入: [数据损失, 几何违反, 各源特征]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softplus()
        )

        # 历史记录
        self.register_buffer('data_loss_history', torch.zeros(100))
        self.register_buffer('geometric_violation_history', torch.zeros(100))
        self.history_index = 0

    def forward(self, separated_sources_delta, mixture=None, current_data_loss=None):
        """对每个分离源应用硬几何约束"""
        batch_size = separated_sources_delta[0].shape[0]
        geometric_violations = []
        constrained_sources = []

        for i, delta in enumerate(separated_sources_delta):
            constraint_func = self.source_constraints[i]
            constrained_source = constraint_func(delta, mixture)
            constrained_sources.append(constrained_source)

            geometric_violation = self._compute_source_geometric_violation(constrained_source, i)
            geometric_violations.append(geometric_violation)

        avg_geometric_violation = torch.mean(torch.stack(geometric_violations))

        # 自适应几何权重
        if current_data_loss is not None:
            source_features = self._compute_source_features(separated_sources_delta)
            weights = self.compute_adaptive_weights(current_data_loss, avg_geometric_violation, source_features)
            w_scale_laplacian, w_scale_time = weights[0], weights[1]
        else:
            device = avg_geometric_violation.device
            w_scale_laplacian, w_scale_time = torch.tensor(1.0, device=device), torch.tensor(1.0, device=device)

        return 0.5 * (w_scale_laplacian + w_scale_time) * avg_geometric_violation, constrained_sources

    def _compute_source_geometric_violation(self, source, source_idx):
        """计算单个源的几何违反程度"""
        scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(source)
        scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
        return scale_laplacian_violation + scale_time_violation

    def _compute_source_features(self, sources):
        """计算源特征用于自适应权重"""
        features = []
        for source in sources:
            energy = torch.mean(source ** 2)
            spectrum = torch.fft.fft(source, dim=-1)
            freqs = torch.fft.fftfreq(source.shape[-1], 1 / self.sample_rate)
            spectral_centroid = torch.sum(torch.abs(spectrum) * freqs.abs()) / torch.sum(torch.abs(spectrum))
            features.extend([energy, spectral_centroid])
        return torch.stack(features)

    def compute_adaptive_weights(self, current_data_loss, current_geometric_violation, source_features):
        """计算自适应几何权重"""
        self.update_history(current_data_loss, current_geometric_violation)
        avg_data = self.data_loss_history.mean()
        avg_geo = self.geometric_violation_history.mean()

        device = next(self.adaptive_weight_net.parameters()).device

        if avg_data > 1e-8 and avg_geo > 1e-8:
            rel_data = current_data_loss / avg_data
            rel_geo = current_geometric_violation / avg_geo
            input_features = torch.cat([
                torch.tensor([rel_data, rel_geo], device=device),
                source_features
            ])
            weights = self.adaptive_weight_net(input_features)
            return weights[0], weights[1]
        else:
            return torch.tensor(0.1, device=device), torch.tensor(0.1, device=device)

    def update_history(self, data_loss, geometric_violation):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss
        self.geometric_violation_history[self.history_index] = geometric_violation
        self.history_index = (self.history_index + 1) % len(self.data_loss_history)


class SourceSpecificGeometricConstraint(nn.Module):
    """源特定几何约束"""

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

        # 初始解网络（基底信号）
        self.initial_net = nn.Sequential(
            nn.Linear(2, hidden_dim),  # 输入: [时间, 混合信号特征]
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._initialize_boundary_function()

    def _initialize_boundary_function(self):
        """初始化边界函数为淡入淡出形状"""
        with torch.no_grad():
            self.boundary_net[-2].bias.data.fill_(-2.0)

    def forward(self, delta, mixture=None):
        """应用源特定硬约束: s = s0 + g * delta"""
        batch_size, seq_len = delta.shape

        # 生成时间坐标
        t = torch.linspace(0, 1, seq_len, device=delta.device).unsqueeze(0)  # [1, seq_len]

        # 学习边界函数 g(t)
        g = self.boundary_net(t.unsqueeze(-1)).squeeze(-1)  # [1, seq_len]
        g = g.expand(batch_size, -1)  # [batch_size, seq_len]

        # 学习初始解 s0(t)
        if mixture is not None:
            mixture_features = self._extract_mixture_features(mixture)
            initial_input = torch.stack([t.expand(batch_size, -1), mixture_features], dim=-1)
        else:
            initial_input = t.expand(batch_size, -1).unsqueeze(-1)

        s0 = self.initial_net(initial_input).squeeze(-1)  # [batch_size, seq_len]

        # 应用硬约束
        constrained_signal = s0 + g * delta
        return constrained_signal

    def _extract_mixture_features(self, mixture):
        """从混合信号中提取特征"""
        window_size = 256
        hop_size = 128
        features = []
        for i in range(0, mixture.shape[-1] - window_size, hop_size):
            window = mixture[..., i:i + window_size]
            energy = torch.mean(window ** 2, dim=-1)
            features.append(energy)
        if features:
            avg_features = torch.mean(torch.stack(features, dim=-1), dim=-1)
        else:
            avg_features = torch.mean(mixture ** 2, dim=-1)
        return avg_features.unsqueeze(-1).expand(-1, mixture.shape[-1])


class MultiScaleGeometricConstraint(nn.Module):
    """多尺度几何约束"""

    def __init__(self, sample_rate, scales=[256, 512, 1024]):
        super().__init__()
        self.sample_rate = sample_rate
        self.scales = scales

        # 为每个尺度创建约束
        self.scale_constraints = nn.ModuleList([
            ScaleSpecificGeometricConstraint(scale, sample_rate) for scale in scales
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


class ScaleSpecificGeometricConstraint(nn.Module):
    """尺度特定几何约束"""

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


class SignalGeometricConstraintFunction(nn.Module):
    """信号几何约束函数 - 实现 s_constrained = s0 + g * ŝ"""

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

        self._initialize_boundary_function()

    def _initialize_boundary_function(self):
        """初始化边界函数为淡入淡出形状"""
        with torch.no_grad():
            self.boundary_net[-2].bias.data.fill_(-2.0)

    def forward(self, s_hat, mixf):
        """应用几何约束: s_constrained = s0 + g * ŝ"""
        batch_size, seq_len = s_hat.shape
        mixf = mixf[:, :seq_len]

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


class GeometricAwarePostProcessing(nn.Module):
    """几何感知后处理层 - 带有自适应权重"""

    def __init__(self, sample_rate=8000, num_sources=2, constraint_type="boundary", history_len=100):
        super().__init__()
        self.sample_rate = sample_rate
        self.num_sources = num_sources
        self.constraint_type = constraint_type

        # 几何约束模块
        self.scale_laplacian_loss = OptimizedMorletCWTHelmholtzLoss(
            fs=sample_rate, duration=2.0, freq_low=80, freq_high=2000, num_scales=50
        )
        self.scale_time_evolution_loss = TimeDomainWaveEquationLoss()

        # 为每个源创建几何约束函数
        self.constraint_functions = nn.ModuleList([
            SignalGeometricConstraintFunction(sample_rate, constraint_type, source_idx=i)
            for i in range(num_sources)
        ])

        # 自适应几何权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(2, 32),  # 输入: [相对数据损失, 相对几何损失]
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Softplus()
        )

        # 历史记录
        self.history_len = history_len
        self.register_buffer('data_loss_history', torch.zeros(history_len))
        self.register_buffer('geometric_loss_history', torch.zeros(history_len))
        self.history_index = 0

    def forward(self, separated_sources, mix, current_data_loss=None, compute_loss=True):
        """对初步分离的信号应用几何约束"""
        constrained_sources = []
        geometric_losses = []

        for i, s_hat in enumerate(separated_sources):
            # 应用几何约束
            constraint_func = self.constraint_functions[i]
            s_constrained = constraint_func(s_hat.unsqueeze(0), mix)
            constrained_sources.append(s_constrained)

            # 计算几何损失
            if compute_loss:
                geometric_loss = self._compute_geometric_loss(s_constrained)
                geometric_losses.append(geometric_loss)

        if compute_loss and geometric_losses:
            raw_geometric_loss = torch.mean(torch.stack(geometric_losses))

            # 计算自适应几何权重
            if current_data_loss is not None:
                lambda_geometric = self._compute_adaptive_weight(current_data_loss, raw_geometric_loss)
                weighted_geometric_loss = lambda_geometric * raw_geometric_loss
            else:
                device = raw_geometric_loss.device
                lambda_geometric = torch.tensor(0.1, device=device)
                weighted_geometric_loss = lambda_geometric * raw_geometric_loss
        else:
            weighted_geometric_loss = torch.tensor(0.0, device=separated_sources[0].device)
            raw_geometric_loss = weighted_geometric_loss
            lambda_geometric = torch.tensor(0.0, device=separated_sources[0].device)

        return constrained_sources, weighted_geometric_loss, raw_geometric_loss, lambda_geometric

    def _compute_geometric_loss(self, signal):
        """计算单个信号的几何约束违反程度"""
        scale_laplacian_violation, cwt_coeffs, scales = self.scale_laplacian_loss(signal)
        scale_time_violation = self.scale_time_evolution_loss(cwt_coeffs, scales)
        return scale_laplacian_violation + scale_time_violation

    def _update_history(self, data_loss, geometric_loss):
        """更新历史记录"""
        self.data_loss_history[self.history_index] = data_loss.detach()
        self.geometric_loss_history[self.history_index] = geometric_loss.detach()
        self.history_index = (self.history_index + 1) % self.history_len

    def _compute_adaptive_weight(self, current_data_loss, current_geometric_loss):
        """计算自适应几何权重"""
        self._update_history(current_data_loss, current_geometric_loss)
        avg_data_loss = self.data_loss_history.mean()
        avg_geometric_loss = self.geometric_loss_history.mean()

        device = current_data_loss.device

        if avg_data_loss > 1e-8 and avg_geometric_loss > 1e-8:
            relative_data_loss = current_data_loss / avg_data_loss
            relative_geometric_loss = current_geometric_loss / avg_geometric_loss
            input_features = torch.tensor([relative_data_loss, relative_geometric_loss],
                                          dtype=torch.float32, device=device)
            lambda_geometric = self.adaptive_weight_net(input_features).squeeze(0)
        else:
            lambda_geometric = torch.tensor(0.1, device=device)

        return lambda_geometric


class GeometricAwareSeparationSystem(nn.Module):
    """几何感知分离系统 - 包装现有分离模型"""

    def __init__(self, base_separation_model, sample_rate=8000, num_sources=2):
        super().__init__()
        self.base_model = base_separation_model
        self.geometric_postprocessor = GeometricAwarePostProcessing(
            sample_rate=sample_rate, num_sources=num_sources
        )

    def forward(self, mixture, compute_geometric_loss=True):
        # 基础模型分离
        separated_sources = self.base_model(mixture)

        if isinstance(separated_sources, tuple):
            separated_sources = list(separated_sources)
        elif separated_sources.dim() == 3:
            separated_sources = [separated_sources[:, i] for i in range(separated_sources.shape[1])]

        # 几何后处理
        constrained_sources, geometric_loss = self.geometric_postprocessor(
            separated_sources, mixture, compute_loss=compute_geometric_loss
        )

        return constrained_sources, geometric_loss, separated_sources


class ConvTasNetGeometricWrapper(nn.Module):
    """Conv-TasNet 几何包装器"""

    def __init__(self, conv_tasnet_model, sample_rate=8000, num_sources=2):
        super().__init__()
        self.conv_tasnet = conv_tasnet_model
        self.geometric_postprocessor = GeometricAwarePostProcessing(
            sample_rate=sample_rate, num_sources=num_sources
        )

    def forward(self, mixture, compute_geometric_loss=True):
        # Conv-TasNet 输出形状: [batch, num_sources, seq_len]
        raw_output = self.conv_tasnet(mixture)
        separated_sources = [raw_output[:, i] for i in range(raw_output.shape[1])]

        # 几何后处理
        constrained_sources, geometric_loss = self.geometric_postprocessor(
            separated_sources, mixture, compute_loss=compute_geometric_loss
        )

        return constrained_sources, geometric_loss, separated_sources



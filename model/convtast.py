

import os
import time

import torch
# from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from physical_constraints import AdaptivePhysicsRegularizer, PhysicsRegularizer, SpeechSeparationHardConstraint, \
    PhysicsAwarePostProcessing


# from Datasets import Datasets
# from pytorch_lightning.core.lightning import LightningModule


class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = torch.mean(x, (1, 2), keepdim=True)
        var = torch.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / \
                torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def select_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D_Block(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, dilation=1, norm_type='gLN'):
        super(Conv1D_Block, self).__init__()
        self.conv1x1 = nn.Conv1d(in_channels, out_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = select_norm(norm_type, out_channels)
        if norm_type == 'gLN':
            self.padding = (dilation*(kernel_size-1))//2
        else:
            self.padding = dilation*(kernel_size-1)
        self.dwconv = nn.Conv1d(out_channels, out_channels, kernel_size,
                                1, dilation=dilation, padding=self.padding, groups=out_channels, bias=True)
        self.prelu2 = nn.PReLU()
        self.norm2 = select_norm(norm_type, out_channels)
        self.sconv = nn.Conv1d(out_channels, in_channels, 1, bias=True)
        self.norm_type = norm_type

    def forward(self, x):
        w = self.conv1x1(x)
        w = self.norm1(self.prelu1(w))
        w = self.dwconv(w)
        if self.norm_type == 'cLN':
            w = w[:, :, :-self.padding]
        w = self.norm2(self.prelu2(w))
        w = self.sconv(w)
        x = x + w
        return x


class TCN(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8):
        super(TCN, self).__init__()
        seq = []
        for i in range(X):
            seq.append(Conv1D_Block(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, norm_type=norm_type, dilation=2**i))
        self.tcn = nn.Sequential(*seq)

    def forward(self, x):
        return self.tcn(x)


class Separation(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, kernel_size=3, norm_type='gLN', X=8, R=3):
        super(Separation, self).__init__()
        s = [TCN(in_channels=in_channels, out_channels=out_channels,
                 kernel_size=kernel_size, norm_type=norm_type, X=X) for i in range(R)]
        self.sep = nn.Sequential(*s)

    def forward(self, x):
        return self.sep(x)


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, bottleneck=128, kernel_size=16, norm_type='gLN'):
        super(Encoder, self).__init__()
        self.encoder = nn.Conv1d(
            in_channels, out_channels, kernel_size, kernel_size//2, padding=0)
        self.norm = select_norm(norm_type, out_channels)
        self.conv1x1 = nn.Conv1d(out_channels, bottleneck, 1)

    def forward(self, x):
        if x.dim() >= 3:
            raise RuntimeError(
                "{} accept 1/2D tensor as input, but got {:d}".format(
                    self.__name__, x.dim()))
        # when inference, only one utt
        if x.dim() == 1:
            x = torch.unsqueeze(x, 0)
        if x.dim() == 2:
            x = torch.unsqueeze(x, 1)
        x = self.encoder(x)
        w = self.norm(x)
        w = self.conv1x1(w)
        return x, w


class Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=1, kernel_size=16):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, kernel_size//2, padding=0, bias=True)

    def forward(self, x):
        x = self.decoder(x)
        return torch.squeeze(x)


class ConvTasNet(nn.Module):
    def __init__(self,
                 N=512,
                 L=16,
                 B=128,
                 H=512,
                 P=3,
                 X=8,
                 R=3,
                 norm="gLN",
                 num_spks=2,
                 activate="relu",
                 causal=False
                 ):
        super(ConvTasNet, self).__init__()
        # -----------------------model-----------------------
        self.encoder = Encoder(1, N, B, L, norm)
        self.separation = Separation(B, H, P, norm, X, R)
        self.decoder = Decoder(H, 1, L)
        self.mask = nn.Conv1d(B, H*num_spks, 1, 1)
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        if activate not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(activate))
        self.non_linear = supported_nonlinear[activate]
        self.num_spks = num_spks

    def forward(self, x):
        x, w = self.encoder(x)
        w = self.separation(w)
        m = self.mask(w)
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0))
        d = [x*m[i] for i in range(self.num_spks)]
        s = [self.decoder(d[i]) for i in range(self.num_spks)]
        return s

def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


# model.py
class ConvTasNetWithPhysicsRegularization(nn.Module):
    def __init__(self,
                 N=512, L=16, B=128, H=512, P=3, X=8, R=3,
                 norm="gLN", num_spks=2, activate="relu", causal=False,
                 use_physics_regularization=True,
                 regularization_type="new"):  # "basic" or "adaptive"#!!!!!!1
        super(ConvTasNetWithPhysicsRegularization, self).__init__()

        # 原有的ConvTasNet结构
        self.encoder = Encoder(1, N, B, L, norm)
        self.separation = Separation(B, H, P, norm, X, R)
        self.decoder = Decoder(H, 1, L)
        self.mask = nn.Conv1d(B, H * num_spks, 1, 1)

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax
        }
        self.non_linear = supported_nonlinear[activate]
        self.num_spks = num_spks

        # 物理正则化
        self.use_physics_regularization = use_physics_regularization
        self.regularization_type = regularization_type

        if self.use_physics_regularization:
            if regularization_type == "adaptive":
                self.physics_regularizer = AdaptivePhysicsRegularizer(sample_rate=8000)
            elif regularization_type == "new":
                self.physics_regularizer = PhysicsAwarePostProcessing(sample_rate=8000, num_sources=2)

            else:
                self.physics_regularizer = PhysicsRegularizer(sample_rate=8000)

            # 正则化强度
            self.regularization_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """标准前向传播 - 不包含正则化计算"""
        x_encoded, w = self.encoder(x)
        w = self.separation(w)
        m = self.mask(w)
        m = torch.chunk(m, chunks=self.num_spks, dim=1)
        m = self.non_linear(torch.stack(m, dim=0))
        d = [x_encoded * m[i] for i in range(self.num_spks)]
        separated_sources = [self.decoder(d[i]) for i in range(self.num_spks)]

        return separated_sources

    def compute_regularization_loss(self, separated_sources, mix,data_loss=None):
        """计算正则化损失 - 单独的方法"""
        if not self.use_physics_regularization:
            return torch.tensor(0.0)

        # 获取模型参数
        model_params = list(self.parameters())

        if self.regularization_type == "adaptive" and data_loss is not None:
            regularization_loss = self.physics_regularizer(
                model_params, separated_sources, data_loss
            )
        elif self.regularization_type == "new" and data_loss is not None:
            regularization_loss, constrained_sources = self.physics_regularizer(
                separated_sources, mix, data_loss
            )
        else:
            regularization_loss = self.physics_regularizer(
                model_params, separated_sources
            )
        regularization_strength = torch.sigmoid(self.regularization_lambda) * 1e-4
        return regularization_strength * regularization_loss,constrained_sources
        # 应用正则化强度
        # return torch.exp(self.regularization_lambda) * regularization_loss

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

if __name__ == "__main__":
    from thop import profile
    model=ConvTasNet()
    input_tensor = torch.randn(1, 16000)
    input_tensor = input_tensor.cuda()
    model = model.cuda()
    out=model(input_tensor)
    print(out.shape)
    # macs, params = profile(model, inputs=(input_tensor,))
    # gmacs = macs / (10 ** 9)
    # print("GMACs:", gmacs)
    # gflops = macs * 2 / 1e9  # 将GMAC转换为GFLOP
    # print(f"GFLOPs: {gflops:.2f}")

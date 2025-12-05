from itertools import permutations
import torch
from torch import nn
import torch.nn.functional as F
EPS = 1e-8


class MixerMSE(nn.Module):

    def __init__(self):

        super(MixerMSE, self).__init__()

        self.criterion1 = nn.MSELoss()

        self.criterion2 = nn.MSELoss()

    def forward(self, x, target):

        loss = self.criterion1(x[0, 0, :], target[0, 0, :]) + self.criterion2(x[0, 1, :], target[0, 1, :])

        return loss


def cal_loss_no(source, estimate_source, source_lengths):
    """
        Args:
            来源：[B， C， T]，B 为批量大小，C 为说话人数量，T 为每批长度
            estimate_source：[B、C、T]
            source_lengths： [B]
    """
    # estimate_source=estimate_source.permute(0,2,1)
    max_snr, perms, max_snr_idx = cal_si_snr(source, estimate_source, source_lengths)

    loss = 0 - torch.mean(max_snr)

    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)

    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr(source, estimate_source, source_lengths):
    """
       使用 PIT 训练计算 SI-SNR。

参数：
            source： [B， C， T]， B 是批量大小
            estimate_source：[B、C、T]
            source_lengths：[B]，每个项目都在 [0， T] 之间
    """
    # print(source.size())
    # print(estimate_source.size())
    # estimate_source=estimate_source.permute(0,2,1)



    assert source.size() == estimate_source.size()
    B, C, T = source.size()  # get all parameters
    # mask padding position along T
    mask = get_mask(source, source_lengths)
    estimate_source *= mask

    # Step 1. Zero-mean norm
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # mask padding position along T
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # reshape to use broadcast
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # print("s_target.type()", s_target.type(), s_estimate.type())
    # s_target = <s', s>s / ||s||^2
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # print("pair_wise_dot.type()", pair_wise_dot.type(), "s_target_energy.type()", s_target_energy.type())
    # print("pair_wise_proj.type()", pair_wise_proj.type())
    # e_noise = s' - s_target
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # print(e_noise.type())
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS) # [B, C, C]
    # print("pair_wise_si_snr",pair_wise_si_snr.type())

    # Get max_snr of each utterance
    # permutations, [C!, C]
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # one-hot, [C!, C, C]
    index = torch.unsqueeze(perms, 2)
    # print(index.type())
    # 如果不加.type(torch.float),perms-one-hot为long，在执行torch.einsum时会报错
    perms_one_hot = torch.unsqueeze(perms, dim=0).type(torch.float)
    # print("perms_one_hot", perms_one_hot.type())
    # [B, C!] <- [B, C, C] einsum [C!, C, C], SI-SNR sum of each permutation
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    # print("snr_set.type()",snr_set.type())
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # max_snr = torch.gather(snr_set, 1, max_snr_idx.view(-1, 1))  # [B, 1]
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C

    return max_snr, perms, max_snr_idx


# def auxiliary_loss(new_module_output, pretrain_output):
#     # 在通道维度计算（每个通道独立计算）
#     cosine_sim = F.cosine_similarity(new_module_output, pretrain_output.detach(), dim=2)  # [B, C]
#     cosine_sim = F.relu(cosine_sim)  # 保持之前的负值截断
#     # 能量计算使用tanh
#     output_energy = torch.mean(torch.abs(new_module_output))
#     safe_energy = torch.clamp(output_energy, min=1e-6)
#     scaled_energy = torch.tanh(safe_energy)  # 使用tanh将能量压缩到[0,1)范围
#
#
#     return (
#             10 *  cosine_sim.mean() +
#             10 *  (1 -scaled_energy)
#     )

def cal_loss_pit(source, estimate_source, source_lengths):
    """
        Args:
            来源：[B， C， T]，B 为批量大小，C 为说话人数量，T 为每批长度
            estimate_source：[B、C、T]
            source_lengths： [B]
    """
    # estimate_source = estimate_source.permute(0, 2, 1)
    max_snr, perms, max_snr_idx = cal_si_snr_with_pit(source, estimate_source, source_lengths)

    loss = 0 - torch.mean(max_snr)

    reorder_estimate_source = reorder_source(estimate_source, perms, max_snr_idx)

    return loss, max_snr, estimate_source, reorder_estimate_source


def cal_si_snr_with_pit(source, estimate_source, source_lengths):
    """
        Calculate SI-SNR with PIT training.

        Args:
            source: [B, C, T], B is batch size
            estimate_source: [B, C, T]
            source_lengths: [B], each item is between [0, T]
    """
    # 检查输入的源数据和估计源数据的大小是否相等

    assert source.size() == estimate_source.size()
    # 获取输入数据的维度信息
    B, C, T = source.size()  # get all parameters
    # 调用 get_mask 函数生成掩码，用于标记填充位置
    # print(source.size())
    # print(source_lengths)
    mask = get_mask(source, source_lengths)
    # 将估计源数据乘以掩码，以处理填充位置
    estimate_source *= mask

    # Step 1. Zero-mean norm
    # 将源长度转换为浮点数并重塑为 [B, 1, 1] 的形状
    num_samples = source_lengths.view(-1, 1, 1).float()  # [B, 1, 1]
    # 计算源数据的均值
    mean_target = torch.sum(source, dim=2, keepdim=True) / num_samples
    # 计算估计源数据的均值
    mean_estimate = torch.sum(estimate_source, dim=2, keepdim=True) / num_samples
    # 源数据减去均值得到零均值的源数据
    zero_mean_target = source - mean_target
    # 估计源数据减去均值得到零均值的估计源数据
    zero_mean_estimate = estimate_source - mean_estimate
    # 对零均值的源数据和估计源数据应用掩码，处理填充位置
    zero_mean_target *= mask
    zero_mean_estimate *= mask

    # Step 2. SI-SNR with PIT
    # 将零均值的源数据重塑为 [B, 1, C, T] 的形状，以便使用广播
    s_target = torch.unsqueeze(zero_mean_target, dim=1)  # [B, 1, C, T]
    # 将零均值的估计源数据重塑为 [B, C, 1, T] 的形状，以便使用广播
    s_estimate = torch.unsqueeze(zero_mean_estimate, dim=2)  # [B, C, 1, T]
    # 计算 s_estimate 和 s_target 的点积
    pair_wise_dot = torch.sum(s_estimate * s_target, dim=3, keepdim=True)  # [B, C, C, 1]
    # 计算 s_target 的能量
    s_target_energy = torch.sum(s_target ** 2, dim=3, keepdim=True) + EPS  # [B, 1, C, 1]
    # 计算 s_estimate 在 s_target 上的投影
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, C, T]
    # 计算噪声 e_noise
    e_noise = s_estimate - pair_wise_proj  # [B, C, C, T]
    # 计算 SI-SNR
    pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=3) / (torch.sum(e_noise ** 2, dim=3) + EPS)
    # 将 SI-SNR 转换为分贝形式
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS) # [B, C, C]


    # Get max_snr of each utterance
    # 生成源数据维度 C 的排列组合
    perms = source.new_tensor(list(permutations(range(C))), dtype=torch.long)
    # 将排列组合扩展为 one-hot 编码
    index = torch.unsqueeze(perms, 2)
    # 创建 one-hot 编码矩阵，用于后续计算
    perms_one_hot = source.new_zeros((*perms.size(), C)).scatter_(2, index, 1).type(torch.float)
    # 使用爱因斯坦求和约定计算每个排列组合的 SI-SNR 总和
    snr_set = torch.einsum('bij,pij->bp', [pair_wise_si_snr, perms_one_hot])
    # 找到每个样本的最大 SI-SNR 的索引
    max_snr_idx = torch.argmax(snr_set, dim=1)  # [B]
    # 计算最大的 SI-SNR 并除以 C
    max_snr, _ = torch.max(snr_set, dim=1, keepdim=True)
    max_snr /= C
    return max_snr, perms, max_snr_idx


def reorder_source(source, perms, max_snr_idx):
    """
        Args:
            source: [B, C, T]
            perms: [C!, C], 排列
            max_snr_idx: [B], 每个项目介于 [0, C!)
        Returns:
            reorder_source: [B, C, T]
    """
    # 获取输入数据的维度信息
    B, C, *_ = source.size()
    # 选择具有最大 SI-SNR 的排列
    max_snr_perm = torch.index_select(perms, dim=0, index=max_snr_idx)
    # 创建一个与源数据形状相同的零张量
    reorder_source = torch.zeros_like(source)
    # 根据最大 SI-SNR 的排列对源数据进行重新排序
    for b in range(B):
        for c in range(C):
            reorder_source[b, c] = source[b, max_snr_perm[b][c]]
    return reorder_source


def get_mask(source, source_lengths):
    """
        Args:
            source: [B, C, T]
            source_lengths: [B]
        Returns:
            mask: [B, 1, T]
    """
    # 获取输入数据的维度信息
    B, _, T = source.size()
    # print(B)
    # print(T)
    # 创建一个全 1 的张量作为初始掩码
    mask = source.new_ones((B, 1, T))
    # 将超出源长度的部分置为 0，生成最终的掩码
    for i in range(B):
        mask[i, :, source_lengths[i]:] = 0
    return mask



if __name__ == "__main__":
    torch.manual_seed(123)
    B, C, T = 1, 2, 32000
    # fake data
    source = torch.randint(4, (B, C, T))
    estimate_source = torch.randint(4, (B, C, T))
    source[0, :, -3:] = 0
    estimate_source[0, :, -3:] = 0
    source_lengths = torch.FloatTensor([T, T - 1]).type(torch.int)
    print('source', source)
    print('estimate_source', estimate_source)
    print('source_lengths', source_lengths)

    loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_no(source, estimate_source, source_lengths)
    print('loss', loss)
    print('max_snr', max_snr)
    print('reorder_estimate_source', reorder_estimate_source)

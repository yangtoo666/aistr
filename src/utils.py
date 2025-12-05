import math
import torch


def overlap_and_add(signal, frame_step):
    """
       从成帧表示中重建信号。

添加形状为
        '[...， frames， frame_length]'，用 'frame_step' 抵消后续帧。
        生成的张量的形状为 '[...， output_size]'，其中

output_size = （帧数 - 1） * frame_step + frame_length

参数：
            signal：一个 [...， frames， frame_length] 张量。所有维度都可以是未知的，并且 rank 必须至少为 2。
            frame_step：表示重叠偏移量的整数。必须小于或等于 frame_length。

返回：
            形状为 [...， output_size] 的 Tensor，包含 signal 最内层两个维度的重叠添加帧。
            output_size = （帧数 - 1） * frame_step + frame_length

基于 https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


def remove_pad(inputs, inputs_lengths):
    """
        Args:
            inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
            inputs_lengths: torch.Tensor, [B]
        Returns:
            results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.dim()
    if dim == 3:
        C = inputs.size(1)
    for input, length in zip(inputs, inputs_lengths):
        if dim == 3: # [B, C, T]
            results.append(input[:,:length].view(C, -1).cpu().numpy())
        elif dim == 2:  # [B, T]
            results.append(input[:length].view(-1).cpu().numpy())
    return results


def check_parameters(net):
    """
        Returns module parameters. Mb
    """
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


if __name__ == '__main__':
    torch.manual_seed(123)
    M, C, K, N = 2, 2, 3, 4
    frame_step = 2
    signal = torch.randint(5, (M, C, K, N))
    result = overlap_and_add(signal, frame_step)
    print(signal)
    print(result)

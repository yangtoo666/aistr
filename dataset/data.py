"""
    Logic:
    (1)  AudioDataLoader 从 AudioDataset 生成一个 mini_batch，这个 mini_batch 的
        大小是 AudioDataLoader 的 batch_size。现在，我们总是将 AudioDataLoader 的批处
        理大小设置为 1。我们真正关心的小批量大小是在 AudioDataset 的 __init__(…)中设置的。
        实际上，我们在 AudioDataset 中生成一个 mini_batch 的信息。

    (2) 在 AudioDataLoader 从 AudioDataset 获得一个 mini_batch 之后，AudioDataLoader
        调用它的 collate_fn(batch) 来处理这个 mini_batch。

    Input:
        Mixture WJS0 tr, cv and tt path

    Output:
        One batch at a time.
        Each input's shape is B x T
        Each target's shape is B x C x T
"""

import json
import math
import os
import numpy as np
import torch
import torch.utils.data as data
import librosa
from dataset.preprocess  import preprocess_one_dir

import os
import json
import math
from torch.utils.data import Dataset


# class AudioDataset(Dataset):
#
#     def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_max_len=8.0):
#         super(AudioDataset, self).__init__()
#
#         try:
#             mix_json = os.path.join(json_dir, 'mix.json')
#             s1_json = os.path.join(json_dir, 's1.json')
#             s2_json = os.path.join(json_dir, 's2.json')
#
#             with open(mix_json, 'r') as f:
#                 mix_list = json.load(f)
#
#             with open(s1_json, 'r') as f:
#                 s1_list = json.load(f)
#
#             with open(s2_json, 'r') as f:
#                 s2_list = json.load(f)
#         except FileNotFoundError:
#             print("One or more JSON files not found.")
#             raise
#         except json.JSONDecodeError:
#             print("Error decoding JSON files.")
#             raise
#
#         sorted_mix_list = self.sort_by_sample_num_desc(mix_list)
#         sorted_s1_list = self.sort_by_sample_num_desc(s1_list)
#         sorted_s2_list = self.sort_by_sample_num_desc(s2_list)
#
#         if segment >= 0.0:
#             self.mini_batch = self.process_with_segment(sorted_mix_list, sorted_s1_list, sorted_s2_list,
#                                                         batch_size, sample_rate, segment)
#         else:
#             self.mini_batch = self.process_without_segment(sorted_mix_list, sorted_s1_list, sorted_s2_list,
#                                                            batch_size, sample_rate, cv_max_len)
#
#     def sort_by_sample_num_desc(self, wav_list):
#         return sorted(wav_list, key=lambda info: int(info[1]), reverse=True)
#
#     def process_with_segment(self, sorted_mix_list, sorted_s1_list, sorted_s2_list,
#                              batch_size, sample_rate, segment):
#         segment_len = int(segment * sample_rate)
#         drop_utt = 0
#         drop_len = 0
#
#         for _, sample in sorted_mix_list:
#             if sample < segment_len:
#                 drop_utt += 1
#                 drop_len += sample
#
#         print("Drop {} utterance({:.2f} h) which is short than {} samples".format(drop_utt,
#                                                                                   drop_len / sample_rate / 3600,
#                                                                                   segment_len))
#
#         mini_batch = []
#         start = 0
#
#         while True:
#             num_segments = 0
#             end = start
#             part_mix, part_s1, part_s2 = [], [], []
#
#             while num_segments < batch_size and end < len(sorted_mix_list):
#                 utterance_len = int(sorted_mix_list[end][1])
#
#                 if utterance_len >= segment_len:
#                     # 截断处理
#                     print(sorted_mix_list[end][0], 2)
#                     mix_info = (sorted_mix_list[end][0], segment_len)
#                     s1_info = (sorted_s1_list[end][0], segment_len)
#                     s2_info = (sorted_s2_list[end][0], segment_len)
#
#                     num_segments += 1
#
#                     if num_segments > batch_size:
#                         if start == end:
#                             end += 1
#                         break
#
#                     part_mix.append(mix_info)
#                     part_s1.append(s1_info)
#                     part_s2.append(s2_info)
#                 end += 1
#
#             if len(part_mix) > 0:
#                 mini_batch.append([part_mix, part_s1, part_s2, sample_rate, segment_len])
#
#             if end == len(sorted_mix_list):
#                 break
#
#             start = end
#         #
#         return mini_batch
#
#     def process_without_segment(self, sorted_mix_list, sorted_s1_list, sorted_s2_list,
#                                 batch_size, sample_rate, cv_max_len):
#         mini_batch = []
#         start = 0
#
#         while True:
#             end = min(len(sorted_mix_list), start + batch_size)
#
#             if int(sorted_mix_list[start][1]) > cv_max_len * sample_rate:
#                 start = end
#                 continue
#
#             mini_batch.append([sorted_mix_list[start:end],
#                                sorted_s1_list[start:end],
#                                sorted_s2_list[start:end],
#                                sample_rate,
#                                -1])
#
#             if end == len(sorted_mix_list):
#                 break
#
#             start = end
#
#         return mini_batch
#
#     def __getitem__(self, index):
#         return self.mini_batch[index]
#
#     def __len__(self):
#         return len(self.mini_batch)
#
class AudioDataset(data.Dataset):

    def __init__(self, json_dir, batch_size, sample_rate=8000, segment=4.0, cv_max_len=8.0):

        """
          参数：
                json_dir：包含 mix.json、s1.json 和 s2.json 目录
                segment：音频片段的持续时间，设置为 -1 时，使用完整音频

          xxx_list 是一个列表，每个项目都是一个元组（wav_file、#samples）
        """

        super(AudioDataset, self).__init__()

        # 拼接 json 文件地址
        mix_json = os.path.join(json_dir, 'mix.json')
        s1_json = os.path.join(json_dir, 's1.json')
        s2_json = os.path.join(json_dir, 's2.json')

        # 读取 json 文件（json 文件 => 数据地址 + 数据长度）
        with open(mix_json, 'r') as f:
            mix_list = json.load(f)

        with open(s1_json, 'r') as f:
            s1_list = json.load(f)

        with open(s2_json, 'r') as f:
            s2_list = json.load(f)

        # 按照数据长度排序
        def sort(wav_list):
            return sorted(wav_list, key=lambda info: int(info[1]), reverse=True)

        # 按照长度降序排列
        sorted_mix_list = sort(mix_list)
        sorted_s1_list = sort(s1_list)
        sorted_s2_list = sort(s2_list)

        # 只读取长度为 4 秒的语音
        if segment >= 0.0:
            segment_len = int(segment * sample_rate)  # 4s * 8000/s = 32000 samples

            drop_utt = 0  # 语音数量
            drop_len = 0  # 语音点数

            # 统计小于 4 秒的语音
            for _, sample in sorted_mix_list:
                if sample < segment_len:
                    drop_utt += 1
                    drop_len += sample

            print("Drop {} utterance({:.2f} h) which is short than {} samples".format(drop_utt,
                                                                                      drop_len/sample_rate/36000,
                                                                                      segment_len))

            mini_batch = []
            start = 0

            while True:
                num_segments = 0
                end = start
                part_mix, part_s1, part_s2 = [], [], []

                while num_segments < batch_size and end < len(sorted_mix_list):
                    # 内层循环的条件是
                    # num_segments < batch_size（当前批次内语音片段数量小于设定的批次大小）和
                    # end < len(sorted_mix_list)（还没遍历完整个语音列表）同时满足.
                    # 只要这两个条件有一个不满足，循环就会结束。
                    # 这句话意味着，如果语音列表不能被batch_size整除的话，多余的部分会被抛弃

                    utterance_len = int(sorted_mix_list[end][1])  # 当前语音数据长度

                    if utterance_len >= segment_len:  # 判断语音是否大于 4 秒

                        num_segments += math.ceil(utterance_len/segment_len)  # 向上取整

                        # 大于 4 秒丢弃
                        if num_segments > batch_size:
                            if start == end:
                                end += 1
                            break

                        part_mix.append(sorted_mix_list[end])
                        part_s1.append(sorted_s1_list[end])
                        part_s2.append(sorted_s2_list[end])

                    end += 1

                if len(part_mix) > 0:
                    mini_batch.append([part_mix, part_s1, part_s2, sample_rate, segment_len])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch
        # 读取所有数据
        else:
            mini_batch = []
            start = 0

            while True:
                # 所有语句长度和 start+batch_size 比较大小
                end = min(len(sorted_mix_list), start+batch_size)

                # 跳过较长音频避免内存不足问题
                if int(sorted_mix_list[start][1]) > cv_max_len * sample_rate:
                    start = end
                    continue

                mini_batch.append([sorted_mix_list[start:end],
                                  sorted_s1_list[start:end],
                                  sorted_s2_list[start:end],
                                  sample_rate,
                                  segment])

                if end == len(sorted_mix_list):
                    break

                start = end

            self.mini_batch = mini_batch

    def __getitem__(self, index):
        return self.mini_batch[index]

    def __len__(self):
        return len(self.mini_batch)


class AudioDataLoader(data.DataLoader):
    """
        NOTE: 这里只使用 batch_size = 1，所以 drop_last = True 在这里没有意义
    """

    def __init__(self, *args, **kwargs):

        super(AudioDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn


def _collate_fn(batch):
    """

    """

    def load_mixtures_and_sources1(batch):
        """

        """
        lens0 = []
        mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
        """通过 zip 函数将 mix_infos、s1_infos 和 s2_infos 这三个列表（或可迭代对象）对应元素进行组合，遍历这个组合后的迭代器，
        从中提取出每个元素（看起来是某种包含音频路径信息的元组等）中的第一个元素，分别赋值给 mix_path、s1_path 和 s2_path。"""

        for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
            mix_path = mix_info[1]

            s1_path = s1_info[1]
            s2_path = s2_info[1]
            # x=x+1
            assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]

            # 读取语音数据

            lens0.append(mix_path )
        # print(mixtures[1].shape)
        mixlen=max(lens0)
        return  mixlen,lens0

    mixtures_list = []

    sources_list = []
    mixtures_pad=[]
    sources_pad=[]
    lens1=0
    for onebatch in batch:
        lens1,lens0 =load_mixtures_and_sources1(onebatch)
        lens0 = torch.tensor(lens0)  # 转换为张量
        # print(lens0)
        mixtures, sources = load_mixtures_and_sources(onebatch)
        # print(mixtures)
        # print(mixtures[1].shape)
        pad_value = lens1
        mixtures_pad0 = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 补零，保证长度一样
        sources_pad = pad_list([torch.from_numpy(s).float() for s in sources], pad_value)  # 补零，保证长度一样
        sources_pad0 = sources_pad.permute((0, 2, 1)).contiguous()  # N x T x C -> N x C x T

        # print(mixtures_pad0.shape)
        # mixtures_pad0 = mixtures_pad0.unsqueeze(0)
        # sources_pad0 = sources_pad0.unsqueeze(0)
        mixtures_list.append(mixtures_pad0)

        sources_list.append(sources_pad0)
        # print(mixtures_list[1].shape)
        mixtures_pad = torch.cat(mixtures_list, dim=0)
        sources_pad = torch.cat(sources_list, dim=0)



    # print(mixtures_pad.shape)
    return mixtures_pad, lens0, sources_pad





def load_mixtures_and_sources(batch):
    """
        每个信息包括 wav path 和 wav duration。
        返回：
            mixtures：包含 B 项的列表，每项为 T np.ndarray
            sources：包含 B 项的列表，每项为 T x C np.ndarray
            T 因项目而异。
            这个函数的主要作用是从给定的 batch 数据中加载音频混合数据（mixtures）以及对应的源音频数据（sources），
            并将它们分别整理成特定格式的列表返回。返回的 mixtures 列表中的每个元素是一个 NumPy 数组（
            形状为 T，表示不同长度的时间序列数据，
            这里 T 因具体音频项目而异），sources
            列表中的每个元素是一个二维的 NumPy 数组（形状为 T x C，
            T 同样表示时间序列长度，C 可能表示通道数或者其他特征维度数量，同样会因项目不同而变化）。
    """
    mixtures, sources = [], []
    mix_infos, s1_infos, s2_infos, sample_rate, segment_len = batch
    """通过 zip 函数将 mix_infos、s1_infos 和 s2_infos 这三个列表（或可迭代对象）对应元素进行组合，遍历这个组合后的迭代器，
    从中提取出每个元素（看起来是某种包含音频路径信息的元组等）中的第一个元素，分别赋值给 mix_path、s1_path 和 s2_path。"""
    # print(len(mix_infos))
    for mix_info, s1_info, s2_info in zip(mix_infos, s1_infos, s2_infos):
        mix_path = mix_info[0]

        s1_path = s1_info[0]
        s2_path = s2_info[0]
        # x=x+1
        assert mix_info[1] == s1_info[1] and s1_info[1] == s2_info[1]

        # 读取语音数据
        mix, _ = librosa.load(mix_path, sr=sample_rate)
        print(mix_path)
        s1, _ = librosa.load(s1_path, sr=sample_rate)
        s2, _ = librosa.load(s2_path, sr=sample_rate)

        # 将 s1 与 s2 合并
        s = np.dstack((s1, s2))[0]  # 32000 x 2
        utt_len = mix.shape[-1]  # 32000

        if segment_len >= 0:
            for i in range(0, utt_len-segment_len+1, segment_len):
                mixtures.append(mix[i:i+segment_len])
                sources.append(s[i:i+segment_len])

            if utt_len % segment_len != 0:
                mixtures.append(mix[-segment_len:])
                sources.append(s[-segment_len:])
        else:
            mixtures.append(mix)
            sources.append(s)
    # print(mixtures[1].shape)
    return mixtures, sources


def pad_list(xs, pad_value):
    """pad_list 函数的主要功能是对一个包含张量（通常是表示序列数据的张量，
    比如音频序列、文本序列对应的张量等）的列表进行填充操作，
    使得列表中的所有张量在第一个维度（通常对应序列长度维度）上具有相同的长度。
    填充时使用指定的填充值来补齐不足的部分，最终返回一个填充好的、维度统一的张量"""
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]
    return pad


class EvalDataset(data.Dataset):

    def __init__(self, mix_dir, mix_json, batch_size, sample_rate=8000):
        """
            Args:
                mix_dir: directory including mixture wav files
                mix_json: json file including mixture wav files
        """
        super(EvalDataset, self).__init__()

        assert mix_dir!=None or mix_json!=None

        if mix_dir is not None:
            preprocess_one_dir(mix_dir, mix_dir, 'mix', sample_rate=sample_rate)
            mix_json = os.path.join(mix_dir, 'mix.json')

        with open(mix_json, 'r') as f:
            mix_infos = json.load(f)

        def sort(infos):
            return sorted(infos, key=lambda info: int(info[1]), reverse=True)

        sorted_mix_infos = sort(mix_infos)

        mini_batch = []
        start = 0
        while True:
            end = min(len(sorted_mix_infos), start+batch_size)
            mini_batch.append([sorted_mix_infos[start:end], sample_rate])

            if end == len(sorted_mix_infos):
                break

            start = end

        self.minibatch = mini_batch

    def __getitem__(self, index):
        return self.minibatch[index]

    def __len__(self):
        return len(self.minibatch)


class EvalDataLoader(data.DataLoader):
    """
        NOTE: just use batch_size = 1 here, so drop_last = True makes no sense here.
    """
    def __init__(self, *args, **kwargs):

        super(EvalDataLoader, self).__init__(*args, **kwargs)

        self.collate_fn = _collate_fn_eval


def _collate_fn_eval(num_batch):
    """
        Args:
            num_batch: list, len(batch) = 1. See AudioDataset.__getitem__()
        Returns:
            mixtures_pad: B x T, torch.Tensor
            ilens: B, torch.Tentor
            filenames: a list contain B strings
    """
    assert len(num_batch) == 1

    mixtures, filenames = load_mixtures(num_batch[0])

    ilens = np.array([mix.shape[0] for mix in mixtures])  # 获取输入序列长度的批处理

    pad_value = 0

    mixtures_pad = pad_list([torch.from_numpy(mix).float() for mix in mixtures], pad_value)  # 填充 0

    ilens = torch.from_numpy(ilens)

    return mixtures_pad, ilens, filenames


def load_mixtures(batch):
    """
        Returns:
            mixtures: a list containing B items, each item is T np.ndarray
            filenames: a list containing B strings
            T varies from item to item.
    """
    mixtures, filenames = [], []

    mix_infos, sample_rate = batch

    for mix_info in mix_infos:
        mix_path = mix_info[0]

        mix, _ = librosa.load(mix_path, sr=sample_rate)

        mixtures.append(mix)
        filenames.append(mix_path)

    return mixtures, filenames


if __name__ == "__main__":

    dataset = AudioDataset(json_dir="D:\\ygt\\Libri2Mix_8k\\Li bri2Mix\\wav8k\\json\\tr", batch_size=1, sample_rate=8000, segment=-1, cv_max_len=6)

    loader = AudioDataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False)

    print("mini_batch number: {}".format(len(loader)))

    for i, batch in enumerate(loader):
        # print(batch[3])
        mixtures, lens, sources = batch
        # print(mixtures.shape)
        # print(lens)
        # print(sources.shape)
        if i >= 0:
            break

    # dataset = EvalDataset(mix_dir="C:\\dataset\\min\\tt\\mix",
    #                       mix_json="C:\\dataset\\json\\tt",
    #                       batch_size=1,
    #                       sample_rate=8000)
    #
    # loader = EvalDataLoader(dataset,
    #                         batch_size=1,
    #                         shuffle=True,
    #                         num_workers=0,
    #                         drop_last=False)
    #
    # print("mini_batch number: {}".format(len(loader)))
    #
    # for i, batch in enumerate(loader):
    #     mixtures, lens, filename = batch
    #     if i >= 0:
    #         break
    #

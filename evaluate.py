import os

from model.hmhzeasyodeold1111 import si_snr_loss

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import argparse
from mir_eval.separation import bss_eval_sources
import numpy as np
import torch
from dataset.data import AudioDataLoader, AudioDataset
from src.pit_criterion import cal_loss_no, cal_loss_pit

from src.utils import remove_pad
import json5
import pesq
import numpy as np
from pystoi import stoi
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
# from model.gnnnmf import Gnnnmfatt
# from speechbrain.inference.separation import SepformerSeparation as separator

# def cal_STOI(src_ref, src_est, fs=8000):
#     """
#     计算STOI指标的函数
#
#     参数:
#     src_ref: 参考语音信号，一维的numpy数组表示音频采样数据
#     src_est: 估计的语音信号，一维的numpy数组表示音频采样数据
#     fs: 音频的采样率，默认值为16000Hz
#
#     返回值:
#     stoi_score: 计算得到的STOI分数
#     """
#     if len(src_ref) != len(src_est):
#         raise ValueError("参考语音信号和估计语音信号长度不一致")
#     stoi_score = stoi(src_ref, src_est, fs, extended=False)
#     return stoi_score


def cal_STOI(src_ref, src_est, fs=8000):
    """
       使用PIT计算使pesq最大的排列，并返回最大排列中两个pesq的平均值。

       参数:
       src_ref (np.ndarray): 形状为(2, 63877)的参考语音数组，代表两个说话人的参考语音
       src_est (np.ndarray): 形状为(2, 63877)的估计语音数组，代表分离出的两个说话人的语音

       返回:
       float: 最大排列中两个pesq的平均值
       """
    num_speakers = src_ref.shape[0]
    permutations = [(0, 1), (1, 0)]
    max_estoi_sum = float('-inf')

    for perm in permutations:
        estoi_vals = []
        for s in range(num_speakers):
            ref = src_ref[s].astype(np.float64)  # 确保数据类型正确
            est = src_est[perm[s]].astype(np.float64)
            # 使用extended=True启用ESTOI计算
            estoi_val = stoi(ref, est, fs, extended=True)
            estoi_vals.append(estoi_val)
        current_sum = sum(estoi_vals)
        if current_sum > max_estoi_sum:
            max_estoi_sum = current_sum

    return max_estoi_sum / num_speakers


def cal_PESQ(src_ref, src_est, fs=8000):
    """
       使用PIT计算使pesq最大的排列，并返回最大排列中两个pesq的平均值。

       参数:
       src_ref (np.ndarray): 形状为(2, 63877)的参考语音数组，代表两个说话人的参考语音
       src_est (np.ndarray): 形状为(2, 63877)的估计语音数组，代表分离出的两个说话人的语音

       返回:
       float: 最大排列中两个pesq的平均值
       """
    num_speakers = src_ref.shape[0]  # 获取说话人数量，这里为2
    permutations = [(0, 1), (1, 0)]  # 两个说话人的所有排列情况
    max_pesq_sum = float('-inf')
    for perm in permutations:
        pesq_vals = []
        for s in range(num_speakers):
            ref = src_ref[s]  # 获取对应说话人的参考语音
            est = src_est[perm[s]]  # 根据排列获取对应说话人的估计语音
            # 假设采样率是16000Hz，根据实际调整，这里计算pesq值
            pesq_val = pesq.pesq(8000,ref, est,'nb' )
            pesq_vals.append(pesq_val)
        pesq_sum = sum(pesq_vals)
        max_pesq_sum = max(max_pesq_sum, pesq_sum)
    return max_pesq_sum / num_speakers

def cal_SDRi(src_ref, src_est, mix):

    """
       计算 Source-to-Distortion Ratio 改进 （SDRi）。

注意：bss_eval_sources 非常非常慢。

参数：
            src_ref：numpy.ndarray， [C， T]
            src_est：numpy.ndarray， [C， T]，按最佳 PIT 排列重新排序
            混合： numpy.ndarray， [T]

返回：
            average_SDRi
    """

    src_anchor = np.stack([mix, mix], axis=0)
    assert len(src_ref)==len(src_est)
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0] - sdr0[0]) + (sdr[1] - sdr0[1])) / 2

    return avg_SDRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):

    """
        Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)

        Args:
            ref_sig: numpy.ndarray, [T]
            out_sig: numpy.ndarray, [T]
        Returns:
            SISNR
    """

    assert len(ref_sig) == len(out_sig)

    ref_sig = ref_sig - np.mean(ref_sig)

    out_sig = out_sig - np.mean(out_sig)

    ref_energy = np.sum(ref_sig ** 2) + eps

    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy

    noise = out_sig - proj

    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)

    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)

    return sisnr


def cal_SISNRi(src_ref, src_est, mix):

    """
        Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)

        Args:
            src_ref: numpy.ndarray, [C, T]
            src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
            mix: numpy.ndarray, [T]
        Returns:
            average_SISNRi
    """

    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr11 = cal_SISNR(src_ref[0], src_est[1])
    sisnr1 = max(sisnr1, sisnr11)
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr22 = cal_SISNR(src_ref[1], src_est[0])
    sisnr2 = max(sisnr2, sisnr22)
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)

    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2

    return avg_SISNRi


def main(config):
    assert torch.cuda.is_available(), "CUDA设备不可用，请检查GPU配置"
    torch.backends.cudnn.benchmark = True  # 启用CUDA加速
    print("CUDA可用，已启用加速")
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0
    total_PESQ = 0
    total_STOI=0

    from model.convtast import ConvTasNetWithPhysicsRegularization as helmholtz_model
    model = helmholtz_model.load_model(config["model_path"])
    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")

    model.eval()  # 将模型设置为验证模式

    if torch.cuda.is_available():
        model.cuda()

    # 加载数据
    dataset = AudioDataset(config["evaluate_dataset"]["data_dir"], config["evaluate_dataset"]["batch_size"],
                           sample_rate=config["evaluate_dataset"]["sample_rate"],
                           segment=config["evaluate_dataset"]["segment"])

    data_loader = AudioDataLoader(dataset, batch_size=1, num_workers=2)

    # 不计算梯度
    with torch.no_grad():

        for i, (data) in enumerate(data_loader):



            # 添加设备统一转换
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            padded_mixture, mixture_lengths, padded_source = [d.to(device) for d in data]

            separated_sources = model(padded_mixture)
            compute_physics_loss = False
            separated_sources, _, _, _ = model.physics_regularizer(
                separated_sources, padded_mixture, False, compute_physics_loss
            )



            sep1, sep2 = separated_sources[0].unsqueeze(0), separated_sources[1].unsqueeze(0)

            # 计算SI-SNR损失
            loss, max_snr, estimate_source, reorder_estimate_source = cal_loss_pit(padded_source,  # mix
                                                                                   torch.stack([sep1, sep2], dim=1).squeeze(2),  # [s1, s2]
                                                                                   mixture_lengths)  # length


            # Remove padding and flat
            mixture = remove_pad(padded_mixture, mixture_lengths)
            source = remove_pad(padded_source, mixture_lengths)

            # NOTE: use reorder estimate source
            estimate_source = remove_pad(reorder_estimate_source, mixture_lengths)

            # for each utterance
            for mix, src_ref, src_est in zip(mixture, source, estimate_source):
                print("Utt", total_cnt + 1, ": ")

                # Compute SDRi
                if config["cal_sdr"]:
                    # (2, 32000) (2, 32000) (32000,)
                    avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                    print("    SDRi = {0:.2f}, ".format(avg_SDRi))
                    total_SDRi += avg_SDRi
                    # 将 (SDRi值, SI-SNRi值, 当前语音样本相关信息) 存入sdr_records
                    # sdr_records.append((avg_SDRi, cal_SISNRi(src_ref, src_est, mix), (mix, src_ref, src_est)))
                avg_PESQ = cal_PESQ(src_ref, src_est, 8000)
                print("    PESQ = {0:.2f}".format(avg_PESQ))
                total_PESQ += avg_PESQ
                # cal_STOI
                avg_STOI = cal_STOI(src_ref, src_est, 8000)
                print("    RSTOI = {0:.2f}".format(avg_STOI))
                total_STOI += avg_STOI
                # Compute SI-SNRi
                avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
                print("    SI-SNRi = {0:.2f}".format(avg_SISNRi))
                total_SISNRi += avg_SISNRi

                total_cnt += 1

    if config["cal_sdr"]:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi/total_cnt))
    print("Average PESQ improvement: {0:.2f}".format(total_PESQ / total_cnt))
    print("Average ESTOI improvement: {0:.2f}".format(total_STOI / total_cnt))
    print("Average SI_SNR improvement: {0:.2f}".format(total_SISNRi/total_cnt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Speech Separation Performance")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/test/evaluate.json5",
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)


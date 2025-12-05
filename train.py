import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# from model.sdeoptmodel import VPSDE_Model
from model.hmhzeasyodeold1111 import HelmholtzPINNODESeparator as helmholtz_model


import argparse
import torch
from dataset.data import AudioDataLoader, AudioDataset
# from dataset.data5spk import AudioDataLoader, AudioDataset
from src.trainer import Trainer

# from model.gnnnmf import Gnnnmfatt
import json5
import numpy as np
from adamp import AdamP, SGDP


def main(config):
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # 数据
    tr_dataset = AudioDataset(json_dir=config["train_dataset"]["train_dir"],
                              batch_size=config["train_dataset"]["batch_size"],
                              sample_rate=config["train_dataset"]["sample_rate"],
                              segment=config["train_dataset"]["segment"])  # 语音时长

    cv_dataset = AudioDataset(json_dir=config["validation_dataset"]["validation_dir"],
                              batch_size=config["validation_dataset"]["batch_size"],
                              sample_rate=config["validation_dataset"]["sample_rate"],
                              segment=config["validation_dataset"]["segment"],
                              cv_max_len=config["validation_dataset"]["cv_max_len"])

    tr_loader = AudioDataLoader(tr_dataset,
                                batch_size=config["train_loader"]["batch_size"],
                                shuffle=config["train_loader"]["shuffle"],
                                num_workers=config["train_loader"]["num_workers"])

    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size=config["validation_loader"]["batch_size"],
                                shuffle=config["validation_loader"]["shuffle"],
                                num_workers=config["validation_loader"]["num_workers"])

    data = {"tr_loader": tr_loader, "cv_loader": cv_loader}

    # 模型
    if config["model"]["type"] == "Gnnnmfatt":
        model = Gnnnmfatt(out_channels=config["model"]["Gnnnmfatt"]["out_channels"],
                          in_channels=config["model"]["Gnnnmfatt"]["in_channels"],
                          graphdim=config["model"]["Gnnnmfatt"]["graphdim"],
                          upsampling_depth=config["model"]["Gnnnmfatt"]["upsampling_depth"],
                          enc_kernel_size=config["model"]["Gnnnmfatt"]["enc_kernel_size"],
                          num_sources=config["model"]["Gnnnmfatt"]["num_sources"],
                          sample_rate=config["model"]["Gnnnmfatt"]["sample_rate"],
                          numblocks=config["model"]["Gnnnmfatt"]["numblocks"],
                          )
        # model = Gnnnmfdiffwave()
    elif config["model"]["type"] == "gnnode":
        model = VPSDE_Model(num_steps=config["model"]["gnnode"]["num_steps"],
                            num_speakers=config["model"]["gnnode"]["num_speakers"],
                          )
    elif config["model"]["type"] == "pinn":
        model = helmholtz_model(config["model"]["pinn"]["hidden_dim"],
                                config["model"]["pinn"]["sample_rate"],
                                config["model"]["pinn"]["use_adaptive_alf"],
                                config["model"]["pinn"]["constraint_weight"],
                                )
    elif config["model"]["type"] == "pinnconvtast":
        from model.convtast import ConvTasNetWithPhysicsRegularization
        model = ConvTasNetWithPhysicsRegularization(
                                )

    else:
        print("No loaded model!")



    if torch.cuda.is_available():
        print("Using GPU")
        # model = torch.nn.DataParallel(model)#多GPU训练
        model.cuda()

    if config["optimizer"]["type"] == "sgd":
        optimize = torch.optim.SGD(
            params=model.parameters(),
            lr=config["optimizer"]["sgd"]["lr"],
            momentum=config["optimizer"]["sgd"]["momentum"],
            weight_decay=config["optimizer"]["sgd"]["l2"])
    elif config["optimizer"]["type"] == "adam":
        optimize = torch.optim.Adam(
            params=model.parameters(),
            lr=config["optimizer"]["adam"]["lr"],
            betas=(config["optimizer"]["adam"]["beta1"], config["optimizer"]["adam"]["beta2"]))
    elif config["optimizer"]["type"] == "sgdp":
        optimize = SGDP(
            params=model.parameters(),
            lr=config["optimizer"]["sgdp"]["lr"],
            weight_decay=config["optimizer"]["sgdp"]["weight_decay"],
            momentum=config["optimizer"]["sgdp"]["momentum"],
            nesterov=config["optimizer"]["sgdp"]["nesterov"],
        )
    elif config["optimizer"]["type"] == "adamp":
        optimize = AdamP(
            params=model.parameters(),
            lr=config["optimizer"]["adamp"]["lr"],
            betas=(config["optimizer"]["adamp"]["beta1"], config["optimizer"]["adamp"]["beta2"]),
            weight_decay=config["optimizer"]["adamp"]["weight_decay"],
        )
        # 建议修改配置部分（原代码中存在配置引用错误）
    elif config["optimizer"]["type"] == "nadam":
        optimize = torch.optim.NAdam(
            params=model.parameters(),
            lr=config["optimizer"]["nadam"]["lr"],  # 原错误引用 adamp 应改为 nadam
            betas=(
                config["optimizer"]["nadam"]["beta1"],  # 添加 beta1 配置项
                config["optimizer"]["nadam"]["beta2"]  # 添加 beta2 配置项
            ),
            weight_decay=config["optimizer"]["nadam"]["weight_decay"],
            momentum_decay=config["optimizer"]["nadam"]["momentum_decay"],  # 新增特有参数
            decoupled_weight_decay=True  # 开启解耦权重衰减
        )
    elif config["optimizer"]["type"] == "adamw":
        optimize = torch.optim.AdamW(
            params=model.parameters(),
            lr=config["optimizer"]["adamw"]["lr"],
            betas=(config["optimizer"]["adamw"]["beta1"],
                   config["optimizer"]["adamw"]["beta2"]),
            weight_decay=config["optimizer"]["adamw"]["weight_decay"]
        )
    else:
        print("Not support optimizer")
        return

    trainer = Trainer(data, model, optimize, config)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Speech Separation")

    parser.add_argument("-C",
                        "--configuration",
                        default="./config/train/train.json5",#这个参数的类型被指定为字符串（type=str），
                        # 默认值是"./config/train/train.json5"，它的作用是指定一个配置文件（这里提示是*.json类型的文件）的路径，
                        # 这个配置文件可能包含了语音分离模型训练相关的各种参数，比如模型结构参数、训练超参数、数据路径等信息
                        type=str,
                        help="Configuration (*.json).")

    args = parser.parse_args()

    configuration = json5.load(open(args.configuration))

    main(configuration)

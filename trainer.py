import os
import time
import numpy as np
import torch
from src.pit_criterion import cal_loss_pit, cal_loss_no, MixerMSE
from torch.utils.tensorboard import SummaryWriter
import gc
import torch.nn.functional as F

class Trainer(object):
    def __init__(self, data, model, optimizer, config):
        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # 训练配置
        self.use_cuda = config["train"]["use_cuda"]
        self.epochs = config["train"]["epochs"]
        self.half_lr = config["train"]["half_lr"]
        self.early_stop = config["train"]["early_stop"]
        self.max_norm = config["train"]["max_norm"]

        # 模型保存与加载
        self.save_folder = config["save_load"]["save_folder"]
        self.checkpoint = config["save_load"]["checkpoint"]
        self.continue_from = config["save_load"]["continue_from"]
        self.model_path = config["save_load"]["model_path"]

        # 日志
        self.print_freq = config["logging"]["print_freq"]

        # 损失记录
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_improve = 0

        # 可视化
        self.write = SummaryWriter("./logs")

        self._reset()
        self.MixerMSE = MixerMSE()

    def _reset(self):
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.save_folder + self.continue_from)
            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module
            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])
            self.start_epoch = int(package.get('epoch', 1))
            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0

    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            print("Train Start...")
            self.model.train()
            start_time = time.time()
            # 使用几何正则化训练
            tr_loss = self.geometric_regularized_train_one_epoch(epoch)
            gc.collect()
            torch.cuda.empty_cache()
            self.write.add_scalar("train loss", tr_loss, epoch+1)
            end_time = time.time()
            run_time = end_time - start_time
            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | Train Loss {2:.3f}'.format(epoch+1, run_time, tr_loss))
            print('-' * 85)

            if self.checkpoint:
                file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                if self.continue_from == "":
                    if isinstance(self.model, torch.nn.DataParallel):
                        self.model = self.model.module
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print('Saving checkpoint model to %s' % file_path)

            print('Cross validation Start...')
            self.model.eval()
            start_time = time.time()
            val_loss = self._run_one_epoch1(epoch, cross_valid=True)
            self.write.add_scalar("validation loss", val_loss, epoch+1)
            end_time = time.time()
            run_time = end_time - start_time
            print('-' * 85)
            print('End of Epoch {0} | Time {1:.2f}s | Valid Loss {2:.3f}'.format(epoch+1, run_time, val_loss))
            print('-' * 85)

            # 学习率调整
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_improve += 1
                    if self.val_no_improve >= 3:
                        self.halving = True
                    if self.val_no_improve >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_improve = 0

            if self.halving:
                optime_state = self.optimizer.state_dict()
                optime_state['param_groups'][0]['lr'] = optime_state['param_groups'][0]['lr']/2.0
                self.optimizer.load_state_dict(optime_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(lr=optime_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss
            self.tr_loss[epoch] = tr_loss
            self.cv_loss[epoch] = val_loss

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.model.serialize(self.model, self.optimizer, epoch + 1,
                                                tr_loss=self.tr_loss, cv_loss=self.cv_loss), file_path)
                print("Find better validated model, saving to %s" % file_path)

    def geometric_regularized_train_one_epoch(self, epoch, cross_valid=False):
        """使用几何正则化的训练方法"""
        start_time = time.time()
        total_loss = 0
        total_data_loss = 0
        total_geometric_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data

            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            # 模型前向传播
            separated_sources = self.model(padded_mixture)

            # 计算数据损失
            target1 = padded_source[:, 0, :]
            target2 = padded_source[:, 1, :]

            sep1, sep2 = separated_sources[0].unsqueeze(0), separated_sources[1].unsqueeze(0)

            loss1 = self.si_snr_loss(sep1, target1) + self.si_snr_loss(sep2, target2)
            loss2 = self.si_snr_loss(sep1, target2) + self.si_snr_loss(sep2, target1)
            current_data_loss = min(loss1, loss2)

            compute_geometric_loss = True

            # 几何正则化后处理
            constrained_sources, weighted_geometric_loss, raw_geometric_loss, lambda_geometric = self.model.geometric_regularizer(
                separated_sources, padded_mixture, current_data_loss, compute_geometric_loss
            )

            sep1, sep2 = constrained_sources[0].unsqueeze(0), constrained_sources[1].unsqueeze(0)

            loss1 = self.si_snr_loss(sep1, target1) + self.si_snr_loss(sep2, target2)
            loss2 = self.si_snr_loss(sep1, target2) + self.si_snr_loss(sep2, target1)
            data_loss = min(loss1, loss2)

            # 总损失 = 数据损失 + 几何正则化损失
            total_batch_loss = data_loss + 0.1 * weighted_geometric_loss
            geometric_loss = weighted_geometric_loss

            if not cross_valid:
                self.optimizer.zero_grad(set_to_none=True)
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += total_batch_loss.item()
            total_data_loss += data_loss.item()
            total_geometric_loss += geometric_loss.item()

            end_time = time.time()
            run_time = end_time - start_time

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Total Loss {2:.3f} | Data Loss {3:.3f} | Geo Loss {4:.3f} | {5:.1f} s/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1), data_loss.item(),
                    geometric_loss.item(), run_time / (i + 1)), flush=True)

            # 清理缓存
            del sep1, sep2, target1, target2, loss1, loss2, data_loss, geometric_loss, total_batch_loss
            torch.cuda.empty_cache()

        # 记录几何正则化损失
        avg_geometric_loss = total_geometric_loss / (i + 1)
        self.write.add_scalar("geometric_regularization_loss", avg_geometric_loss, epoch + 1)

        return total_loss / (i + 1)

    def si_snr_loss(self, pred, target):
        """SI-SNR损失函数"""
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        min_len = min(pred.shape[-1], target.shape[-1])
        if pred.shape[-1] != min_len:
            pred = pred[..., :min_len]
        if target.shape[-1] != min_len:
            target = target[..., :min_len]

        pred = pred - torch.mean(pred, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

        target_energy = torch.sum(target ** 2, dim=-1, keepdim=True)
        scale = torch.sum(pred * target, dim=-1, keepdim=True) / (target_energy + 1e-8)
        target_scaled = scale * target

        noise = pred - target_scaled
        snr = 10 * torch.log10(torch.sum(target_scaled ** 2, dim=-1) /
                               (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8)

        return -torch.mean(snr)

    def _run_one_epoch1(self, epoch, cross_valid=False):
        """验证阶段运行"""
        start_time = time.time()
        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source = data

            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            self.model.eval()
            with torch.no_grad():
                separated_sources = self.model(padded_mixture)
                compute_geometric_loss = False
                # 几何正则化后处理（验证时不计算损失）
                constrained_sources, _, _, _ = self.model.geometric_regularizer(
                    separated_sources, False, compute_geometric_loss
                )

                sep1, sep2 = constrained_sources[0].unsqueeze(0), constrained_sources[1].unsqueeze(0)

            # 计算SI-SNR损失
            target1 = padded_source[:, 0:1, :]
            target2 = padded_source[:, 1:2, :]

            data_loss1 = self.si_snr_loss(sep1, target1) + self.si_snr_loss(sep2, target2)
            data_loss2 = self.si_snr_loss(sep1, target2) + self.si_snr_loss(sep2, target1)
            loss = min(data_loss1, data_loss2)

            total_loss += loss.item()

            end_time = time.time()
            run_time = end_time - start_time

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | Current Loss {3:.6f} | {4:.1f} s/batch'.format(
                    epoch + 1, i + 1, total_loss / (i + 1), loss.item(), run_time / (i + 1)), flush=True)

        return total_loss / (i + 1)   
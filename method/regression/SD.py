import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from scipy.stats import spearmanr
from Regression2.method.model.federal_learning.ours import concoder
from tqdm import tqdm
from Regression2.dataloader.IS import patch2loader_IS_emo_sequence
from Regression2.method.regression.IG.utils import save_mat, restore_pred_label

# 训练类 包含初始化 训练和验证方法
class Trainer(object):
    def __init__(self, args):
        self.args = args
        # 初始化网络模型
        self.model = concoder()  # concoder model

        # 设定损失函数
        self.criterion = nn.MSELoss()  # MSEloss

        # 设定优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=160, gamma=0.1)

        # 检测cuda是否可用
        if args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()

    # 训练
    def train(self, epoch, train_loader):
        # 性能指标 精度和损失
        train_loss = 0.0

        self.model.train()
        tbar = tqdm(train_loader)

        for i, sample in enumerate(tbar):
            x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye, target = sample[0], sample[1], sample[2], sample[3], \
                sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10]

            if self.args.cuda:
                x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye, target = x_pf.cuda(), x_f.cuda(), x_lt.cuda(), x_c.cuda(), x_rt.cuda(), \
                    x_lp.cuda(), x_p.cuda(), x_rp.cuda(), x_o.cuda(), x_eye.cuda(), target.cuda()

            # 获取模型输出和 VAE 损失
            fusion_out, vae_loss = self.model(x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye)

            # 计算 MSE 和 VAE 损失的加权和
            mse_loss = self.criterion(fusion_out, target)
            total_loss = mse_loss + vae_loss  # 这里可以根据需求调整 VAE 损失的权重

            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 优化步骤
            self.optimizer.step()

            train_loss += total_loss.item()
            tbar.set_description(f'Epoch:{epoch}, Train loss: {train_loss / (i + 1):.3f}')

        self.scheduler.step()
        net_res = f'Train: total loss: {train_loss:.6f}'
        tqdm.write(net_res)
        return train_loss

    # 验证
    def test(self, test_loader):
        test_loss = 0.0
        rmse_loss = 0.0
        total_pred = []
        total_mean_pred = []

        self.model.eval()
        tbar = tqdm(test_loader, desc='\r')

        for i, sample in enumerate(tbar):
            x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye, target = sample[0], sample[1], sample[2], sample[3], \
                sample[4], sample[5], sample[6], sample[7], sample[8], sample[9], sample[10]

            if self.args.cuda:
                x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye, target = x_pf.cuda(), x_f.cuda(), x_lt.cuda(), x_c.cuda(), x_rt.cuda(), \
                    x_lp.cuda(), x_p.cuda(), x_rp.cuda(), x_o.cuda(), x_eye.cuda(), target.cuda()

            # 获取模型输出和 VAE 损失
            fusion_out, vae_loss = self.model(x_pf, x_f, x_lt, x_c, x_rt, x_lp, x_p, x_rp, x_o, x_eye)

            mean_output = torch.mean(fusion_out, dim=1)
            mean_target = torch.mean(target, dim=1).squeeze()

            total_pred.append(fusion_out.detach().cpu().numpy())
            total_mean_pred.append(mean_output.detach().cpu().numpy())

            # 计算总损失
            mse_loss = self.criterion(fusion_out, target)
            total_loss = mse_loss + vae_loss

            rmse_loss += torch.sqrt(mse_loss).item()
            test_loss += total_loss.item()
            tbar.set_description(f'test loss: {test_loss / (i + 1):.3f}')

        net_res = f'Test: total loss: {test_loss:.6f}'
        tqdm.write(net_res)
        return test_loss, rmse_loss, total_pred, total_mean_pred



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    category = ['positive', 'neutral', 'negative']
    for type in category:
        for sub in [s for s in range(1, 15) if s != 6 and s != 7]:
            for session in range(1, 4):
                parser = argparse.ArgumentParser()
                parser.add_argument("--batch_size", type=int, default=64, help="batch size used in SGD")
                parser.add_argument("--steps_per_epoch", type=int, default=64, help="the number of batches per epoch")
                parser.add_argument("--epochs", type=int, default=1, help="the number of epochs")
                parser.add_argument("--random_seed", type=int, default=42, help="the random seed number")
                parser.add_argument('--workers', type=int, default=4, metavar='N', help='dataloader threads')
                parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
                parser.add_argument("--sub", type=int, default=sub, help="the number of epochs")
                parser.add_argument("--session", type=int, default=session, help="the random seed number")
                # parser.add_argument('--trial', type=int, default=trial, help='the trial to conduct experiment')
                parser.add_argument('--sequence_len', type=int, default=2, help='length of an EEG sequence')
                parser.add_argument('--overlap_rate', type=float, default=0.5,
                                    help='overlap rate between 2 continuous sequences')
                parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
                parser.add_argument('--category', type=str, default=type, help='emotion category')
                args = parser.parse_args()

                args.cuda = not args.no_cuda and torch.cuda.is_available()

                cate = None
                if args.category == 'positive':
                    if args.session == 1:
                        cate = [1, 6, 9, 10, 14]
                    elif args.session == 2:
                        cate = [1, 6, 9, 10, 14]
                    else:
                        cate = [1, 6, 9, 10, 14]
                elif args.category == 'neutral':
                    if args.session == 1:
                        cate = [2, 5, 8, 11, 13]
                    elif args.session == 2:
                        cate = [2, 5, 8, 11, 13]
                    else:
                        cate = [2, 5, 8, 11, 13]
                else:
                    if args.session == 1:
                        cate = [3, 4, 7, 12, 15]
                    elif args.session == 2:
                        cate = [3, 4, 7, 12, 15]
                    else:
                        cate = [3, 4, 7, 12, 15]

                setup_seed(args.random_seed)

                # 载入de loader
                train_loader, test_loader_list, test_int_label_list = patch2loader_IS_emo_sequence(args)
                trainer1 = Trainer(args)

                print("----------------session {}------------------".format(args.session))
                # 训练
                for epoch in range(0, args.epochs):
                    loss = trainer1.train(epoch, train_loader)

                # 测试
                for i in range(0, 5):
                    print("-------------sub {} trial {}-------------".format(sub, cate[i]))

                    test_loss, rmse_loss, pred, mean_pred = trainer1.test(test_loader_list[i])
                    # test_loss, rmse_loss = trainer1.test(test_loader_list[i])
                    mean_pred = np.hstack(mean_pred)
                    pred = np.vstack(pred)

                    test_label = test_int_label_list[i]
                    mean_test_int_label = np.mean(test_label, axis=1).squeeze()

                    test_label = restore_pred_label(test_label)
                    pred_label = restore_pred_label(pred)

                    spearmanr_value, p_value = spearmanr(test_label, pred_label)

                    threshold = (np.max(test_label) - np.min(test_label)) / 2
                    length = len(pred_label)

                    location_rate = 0.0

                    for j in range(0, length):
                        if pred_label[j] > threshold and test_label[j] > threshold:
                            location_rate += 1
                        elif pred_label[j] < threshold and test_label[j] < threshold:
                            location_rate += 1
                        elif pred_label[j] == test_label[j]:
                            location_rate += 1
                    location_rate = location_rate / length


                    sort_segment = np.argsort(mean_pred, kind='quicksort')
                    top_k = sort_segment[0:10]
                    pred_dict = {'pred': pred_label, 'mean_pred': mean_pred, 'true_label': test_label,
                                 'threshold': threshold, 'location_rate': location_rate,
                                 'mse': test_loss, 'rmse': rmse_loss,
                                 'spearmanr_value': spearmanr_value, 'p_value': p_value,
                                 'top_k': top_k}

                    dict_path = r"C:\Users\kk\Desktop\SCI\regresssion\result\\"

                    if not os.path.exists(dict_path):
                        os.mkdir(dict_path)
                    save_mat(pred_dict, args, cate[i], sub, dict_path)

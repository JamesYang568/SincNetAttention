# -*- coding: utf-8 -*-
# speaker_id.py
# Mirco Ravanelli
# Mila - University of Montreal

# Description:
# This code performs a speaker_id experiments with SincNet.

# How to run it:
# python speaker_id.py --cfg=cfg/SincNet_TIMIT.cfg

import os
# import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models import MLP, flip
from dnn_models import SincNet as CNN  # 注意这里CNN是指的SincNet
from poolings import DoubleMHA
from data_io import ReadList, read_conf, str_to_bool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdditiveMarginSoftmax(nn.Module):
    # AMSoftmax
    def __init__(self, margin=0.35, s=30):
        super().__init__()

        self.m = margin  #
        self.s = s
        self.epsilon = 0.000000000001
        print('AMSoftmax m = ' + str(margin))

    def forward(self, predicted, target):
        # ------------ AM Softmax ------------ #
        predicted = predicted / (predicted.norm(p=2, dim=0) + self.epsilon)
        indexes = range(predicted.size(0))
        cos_theta_y = predicted[indexes, target]
        cos_theta_y_m = cos_theta_y - self.m
        exp_s = np.e ** (self.s * cos_theta_y_m)

        sum_cos_theta_j = (np.e ** (predicted * self.s)).sum(dim=1) - (np.e ** (predicted[indexes, target] * self.s))

        log = -torch.log(exp_s / (exp_s + sum_cos_theta_j + self.epsilon)).mean()

        return log


# 获取数据的函数
def create_batches_rnd(batch_size, data_folder, wav_lst, N_snt, wlen, lab_dict, fact_amp):
    # 一批大小，数据文件夹，文件名单，名单长度，宽度，字典，振幅
    # Initialization of the minibatch (batch_size,[0=>x_t,1=>x_t+N,1=>random_samp])  随机获取128个片段
    sig_batch = np.zeros([batch_size, wlen])  # 转换成128*wlen的矩阵

    lab_batch = np.zeros(batch_size)

    snt_id_arr = np.random.randint(N_snt, size=batch_size)  # 获取小于N_snt的128个向量

    rand_amp_arr = np.random.uniform(1.0 - fact_amp, 1 + fact_amp, batch_size)  # 获取在振幅两侧的随机采样

    for i in range(batch_size):

        # select a random sentence from the list
        # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst[snt_id_arr[i]])
        # signal=signal.astype(float)/32768

        [signal, fs] = sf.read(data_folder + wav_lst[snt_id_arr[i]])

        # accesing to a random chunk
        snt_len = signal.shape[0]
        snt_beg = np.random.randint(snt_len - wlen - 1)  # randint(0, snt_len-2*wlen-1)
        snt_end = snt_beg + wlen

        channels = len(signal.shape)
        if channels == 2:
            print('WARNING: stereo to mono: ' + data_folder + wav_lst[snt_id_arr[i]])
            signal = signal[:, 0]

        sig_batch[i, :] = signal[snt_beg:snt_end] * rand_amp_arr[i]  # 片段的内容
        lab_batch[i] = lab_dict[wav_lst[snt_id_arr[i]]]  # 得到随机选择的片段信息dict

    inp = Variable(torch.from_numpy(sig_batch).float().cuda().contiguous())
    lab = Variable(torch.from_numpy(lab_batch).float().cuda().contiguous())

    return inp, lab


# Reading cfg file 程序的入口函数，读取config用于神经网络的建立
options = read_conf()

# 下面是超参数的赋值
# [data]
tr_lst = options.tr_lst
te_lst = options.te_lst
pt_file = options.pt_file
class_dict_file = options.lab_dict
data_folder = options.data_folder + '/'
output_folder = options.output_folder

# [windowing]
fs = int(options.fs)
cw_len = int(options.cw_len)
cw_shift = int(options.cw_shift)

# [cnn]
cnn_N_filt = list(map(int, options.cnn_N_filt.split(',')))
cnn_len_filt = list(map(int, options.cnn_len_filt.split(',')))
cnn_max_pool_len = list(map(int, options.cnn_max_pool_len.split(',')))
cnn_use_laynorm_inp = str_to_bool(options.cnn_use_laynorm_inp)
cnn_use_batchnorm_inp = str_to_bool(options.cnn_use_batchnorm_inp)
cnn_use_laynorm = list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
cnn_use_batchnorm = list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
cnn_act = list(map(str, options.cnn_act.split(',')))
cnn_drop = list(map(float, options.cnn_drop.split(',')))

# [dnn]
fc_lay = list(map(int, options.fc_lay.split(',')))
fc_drop = list(map(float, options.fc_drop.split(',')))
fc_use_laynorm_inp = str_to_bool(options.fc_use_laynorm_inp)
fc_use_batchnorm_inp = str_to_bool(options.fc_use_batchnorm_inp)
fc_use_batchnorm = list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
fc_use_laynorm = list(map(str_to_bool, options.fc_use_laynorm.split(',')))
fc_act = list(map(str, options.fc_act.split(',')))

# [class]
class_lay = list(map(int, options.class_lay.split(',')))
class_drop = list(map(float, options.class_drop.split(',')))
class_use_laynorm_inp = str_to_bool(options.class_use_laynorm_inp)
class_use_batchnorm_inp = str_to_bool(options.class_use_batchnorm_inp)
class_use_batchnorm = list(map(str_to_bool, options.class_use_batchnorm.split(',')))
class_use_laynorm = list(map(str_to_bool, options.class_use_laynorm.split(',')))
class_act = list(map(str, options.class_act.split(',')))

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# training list
wav_lst_tr = ReadList(tr_lst)
snt_tr = len(wav_lst_tr)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# loss function
# cost = nn.NLLLoss()
cost = AdditiveMarginSoftmax()

# Converting context and shift in samples  加窗
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Batch_dev
Batch_dev = 128

# Feature extractor CNN(SincNet)
CNN_arch = {'input_dim': wlen,
            'fs': fs,
            'cnn_N_filt': cnn_N_filt,
            'cnn_len_filt': cnn_len_filt,
            'cnn_max_pool_len': cnn_max_pool_len,
            'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
            'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
            'cnn_use_laynorm': cnn_use_laynorm,
            'cnn_use_batchnorm': cnn_use_batchnorm,
            'cnn_act': cnn_act,
            'cnn_drop': cnn_drop,
            }

CNN_net = CNN(CNN_arch)
CNN_net.to(device)

# 建立注意力机制
# print(CNN_net.out_dim) 6420
AttentionModule = DoubleMHA(CNN_net.out_dim, 20)  # 8 16 32的头数
# 修改这里保证pooling中assert self.encoder_size % heads_number == 0  # d_model 可以通过

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()

# print(CNN_net.out_dim)  6420
DNN1_arch = {'input_dim': 321,
             'fc_lay': fc_lay,
             'fc_drop': fc_drop,
             'fc_use_batchnorm': fc_use_batchnorm,
             'fc_use_laynorm': fc_use_laynorm,
             'fc_use_laynorm_inp': fc_use_laynorm_inp,
             'fc_use_batchnorm_inp': fc_use_batchnorm_inp,
             'fc_act': fc_act,
             }

DNN1_net = MLP(DNN1_arch)  # 三次循环
DNN1_net.to(device)
# print(fc_lay[-1]) 2048

DNN2_arch = {'input_dim': fc_lay[-1],
             'fc_lay': class_lay,
             'fc_drop': class_drop,
             'fc_use_batchnorm': class_use_batchnorm,
             'fc_use_laynorm': class_use_laynorm,
             'fc_use_laynorm_inp': class_use_laynorm_inp,
             'fc_use_batchnorm_inp': class_use_batchnorm_inp,
             'fc_act': class_act,  # 注意这里使用的就是softmax最后一层
             }

DNN2_net = MLP(DNN2_arch)  # 1次循环
DNN2_net.to(device)

# 批处理
# 如果有做好的模型则使用做好的模型（的训练好的参数）
if pt_file != 'none':
    checkpoint_load = torch.load(pt_file)
    CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
    DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
    DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])

# 配置优化器 全部是用的RMSProp
optimizer_CNN = optim.RMSprop(CNN_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_ATT = optim.RMSprop(AttentionModule.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN1 = optim.RMSprop(DNN1_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)
optimizer_DNN2 = optim.RMSprop(DNN2_net.parameters(), lr=lr, alpha=0.95, eps=1e-8)

# 开始进行训练，1500次
for epoch in range(N_epochs):

    test_flag = 0
    CNN_net.train()  # 这里因为里面包含了的dropout和batchnorm，因此需要指明是在训练还是在验证，在训练时则要.train,在验证时要.eval
    AttentionModule.train()
    DNN1_net.train()  # 注意要在自动forward之前
    DNN2_net.train()

    loss_sum = 0
    err_sum = 0

    for i in range(N_batches):  # 处理一批
        [inp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_tr, snt_tr, wlen, lab_dict, 0.2)

        # 进行训练
        output = CNN_net(inp)
        # under deprecate
        # print(output.shape) [128, 6420]  128条片段，每个是6420的
        # o1, o3 = output.split([1, 1], dim=1)  # 切割列
        # temp = torch.full((128, 1), fill_value=20,dtype = torch.int)  全为20的向量
        # output = torch.cat((output,temp), dim=1) 这个是在现有维度上进行拼接
        # 扩大数据，使符合条件
        output = output.unsqueeze(dim=0)  # 得到【1,128,6420】
        output = output.repeat(20, 1, 1)  # repeat重复对应的位置多少遍（乘以多少），如果是1则乘以1，不变。  这样得到[20,128,6420]
        output = output.permute(1, 0, 2)  # permute交换维度，新的维度就是从左向右的，而对应位置的数字则是原来这个维度的位置

        output, alignment = AttentionModule(output)  # output shape [128, 321]
        pout = DNN2_net(DNN1_net(output))  # DNN1_net(output).shape = [128, 2048]
        # pout.shape = [128, 462]
        # os.system("pause")

        pred = torch.max(pout, dim=1)[1]
        # 在分类问题中，通常需要使用max()函数对softmax函数的输出值进行操作，求出预测值索引 。
        # 将输出最大值在向量中的索引以及最大值是谁。这里（计算准确率）不需要最大值是谁，只需要直到其索引，因此[1]
        loss = cost(pout, lab.long())
        err = torch.mean((pred != lab.long()).float())
        # 优化

        optimizer_CNN.zero_grad()
        optimizer_ATT.zero_grad()
        optimizer_DNN1.zero_grad()
        optimizer_DNN2.zero_grad()

        loss.backward()
        optimizer_CNN.step()
        optimizer_ATT.step()
        optimizer_DNN1.step()
        optimizer_DNN2.step()

        loss_sum = loss_sum + loss.detach()
        err_sum = err_sum + err.detach()
        # if i%10 == 0:  just for test :)
        #     print('loss sum:' + str(loss_sum.item()))
        #     print('err sum:' + str(err_sum.item()))

    loss_tot = loss_sum / N_batches
    err_tot = err_sum / N_batches

    # Full Validation  new
    if epoch % N_eval_epoch == 0:  # 每8个写入到文件一下
        # os.system("pause")
        CNN_net.eval()  # 说明是在验证
        AttentionModule.eval()
        DNN1_net.eval()
        DNN2_net.eval()
        test_flag = 1
        loss_sum = 0
        err_sum = 0
        err_sum_snt = 0

        with torch.no_grad():  # 由于是三个不同的模型拼装所以no_grad上下文不再自动进行梯度下降
            for i in range(snt_te):  # 列表长度

                # [fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
                # signal=signal.astype(float)/32768

                [signal, fs] = sf.read(data_folder + wav_lst_te[i])

                signal = torch.from_numpy(signal).float().cuda().contiguous()
                lab_batch = lab_dict[wav_lst_te[i]]

                # split signals into chunks
                beg_samp = 0
                end_samp = wlen

                N_fr = int((signal.shape[0] - wlen) / (wshift))

                sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()
                lab = Variable((torch.zeros(N_fr + 1) + lab_batch).cuda().contiguous().long())
                pout = Variable(torch.zeros(N_fr + 1, class_lay[-1]).float().cuda().contiguous())
                count_fr = 0
                count_fr_tot = 0
                while end_samp < signal.shape[0]:
                    sig_arr[count_fr, :] = signal[beg_samp:end_samp]
                    beg_samp = beg_samp + wshift
                    end_samp = beg_samp + wlen
                    count_fr = count_fr + 1
                    count_fr_tot = count_fr_tot + 1
                    if count_fr == Batch_dev:
                        inp = Variable(sig_arr)
                        # norm the size
                        temp = CNN_net(inp)
                        temp = temp.unsqueeze(dim=0)
                        temp = temp.repeat(20, 1, 1)
                        temp = temp.permute(1, 0, 2)
                        pout[count_fr_tot - Batch_dev:count_fr_tot, :] = DNN2_net(DNN1_net(AttentionModule(temp)[0]))
                        count_fr = 0
                        sig_arr = torch.zeros([Batch_dev, wlen]).float().cuda().contiguous()

                if count_fr > 0:
                    inp = Variable(sig_arr[0:count_fr])
                    temp = CNN_net(inp)
                    temp = temp.unsqueeze(dim=0)
                    temp = temp.repeat(20, 1, 1)
                    temp = temp.permute(1, 0, 2)
                    pout[count_fr_tot - count_fr:count_fr_tot, :] = DNN2_net(DNN1_net(AttentionModule(temp)[0]))

                pred = torch.max(pout, dim=1)[1]
                loss = cost(pout, lab.long())
                err = torch.mean((pred != lab.long()).float())

                [val, best_class] = torch.max(torch.sum(pout, dim=0), 0)
                err_sum_snt = err_sum_snt + (best_class != lab[0]).float()

                loss_sum = loss_sum + loss.detach()
                err_sum = err_sum + err.detach()

            err_tot_dev_snt = err_sum_snt / snt_te
            loss_tot_dev = loss_sum / snt_te
            err_tot_dev = err_sum / snt_te

        print("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f" % (
            epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        with open(output_folder + "/res.res", "a") as res_file:
            res_file.write("epoch %i, loss_tr=%f err_tr=%f loss_te=%f err_te=%f err_te_snt=%f\n" % (
                epoch, loss_tot, err_tot, loss_tot_dev, err_tot_dev, err_tot_dev_snt))

        checkpoint = {'CNN_model_par': CNN_net.state_dict(),
                      'Attention_model_par': AttentionModule.state_dict(),
                      'DNN1_model_par': DNN1_net.state_dict(),
                      'DNN2_model_par': DNN2_net.state_dict(),
                      }
        torch.save(checkpoint, output_folder + '/model_raw.pkl')

    else:
        print("epoch %i, loss_tr=%f err_tr=%f" % (epoch, loss_tot, err_tot))

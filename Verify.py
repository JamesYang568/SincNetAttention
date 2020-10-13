import os
import torch
import numpy as np
from sklearn.metrics import make_scorer, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from dnn_models import SincNet as CNN
from dnn_models import MLP, flip
from data_io import ReadList, read_conf, str_to_bool
from speaker_id import create_batches_rnd

# python Verify.py --cfg=cfg/SincNet_TIMIT_Vr.cfg

# TODO 注意DATAFOlder的变化
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

# [optimization]
lr = float(options.lr)
batch_size = int(options.batch_size)
N_epochs = int(options.N_epochs)
N_batches = int(options.N_batches)
N_eval_epoch = int(options.N_eval_epoch)
seed = int(options.seed)

# test list
wav_lst_te = ReadList(te_lst)
snt_te = len(wav_lst_te)

# Converting context and shift in samples
wlen = int(fs * cw_len / 1000.00)
wshift = int(fs * cw_shift / 1000.00)

# Folder creation
try:
    os.stat(output_folder)
except:
    os.mkdir(output_folder)

# setting seed
torch.manual_seed(seed)
np.random.seed(seed)

# Loading label dictionary
lab_dict = np.load(class_dict_file, allow_pickle=True).item()

# 读取模型
checkpoint = torch.load('exp/SincNet_TIMIT/model_raw')

CNN_net = CNN({})
CNN_net.load_state_dict(checkpoint['CNN_model_par'])
DNN1_net = MLP({})
DNN1_net.load_state_dict(checkpoint['DNN1_model_par'])
DNN2_net = MLP({})
DNN2_net.load_state_dict(checkpoint['DNN2_model_par'])
eer = 0
for i in range(N_batches):
    # 将test信号预处理batch  TODO data—folder是test数据集
    [inp, lab] = create_batches_rnd(batch_size, data_folder, wav_lst_te, snt_te, wlen, lab_dict, 0.2)

    pout = DNN2_net(DNN1_net(CNN_net(inp)))

    pred = torch.max(pout, dim=1)[1]  # 寻找最大那个就是预测的谁
    fpr, tpr, thresholds = roc_curve(lab, pred, pos_label=1)
    eer += brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # count = 0
    # # FAR 被判定为正样本，但事实上是负样本/总负样本
    # # FRR 被判定为负样本，但事实上是正样本/总正样本
    #

    err = torch.mean((pred != lab.long()).float())
    print(err)

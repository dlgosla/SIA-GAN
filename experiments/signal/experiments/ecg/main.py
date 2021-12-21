

import os
import torch
import torch.nn as nn
from options import Options

from data import load_data
import torch.backends.cudnn as cudnn
import numpy as np
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# from dcgan import DCGAN as myModel

device = torch.device("cuda" if
torch.cuda.is_available() else "cpu")

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
cudnn.benchmark = False
cudnn.deterministic = True
np.random.seed(random_seed)
random.seed(random_seed)


opt = Options().parse()
print(opt)
dataloader=load_data(opt)
print("load data success!!!")



if opt.model == "beatgan":
    from model import BeatGAN as MyModel

else:
    raise Exception("no this model :{}".format(opt.model))


model=MyModel(opt,dataloader,device)

if not opt.istest:
    print("################  Train  ##################")
    model.train()
else:
    print("################  Eval  ##################")
    model.load()
    model.test_type()
#    model.test_time()
    # model.plotTestFig()
    # print("threshold:{}\tf1-score:{}\tauc:{}".format( th, f1, auc))    

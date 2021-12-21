import os
import torch
from options import Options
import numpy as np
from data import load_data    
    
opt = Options().parse()
dataloader=load_data(opt)
device = torch.device("cuda:0" if
torch.cuda.is_available() else "cpu")

from model import BeatGAN as MyModel
model=MyModel(opt,dataloader,device)

save_dir = '/root/volume/B2IGAN/beatgan_base/token128_200epoch/experiments/ecg/200epoch/beatgan/ecg/model/'
# model.G.load_state_dict(torch.load(os.path.join(save_dir, model.model+"_folder_"+str(model.opt.folder) + '_G.pkl')))

ckpt = torch.load(os.path.join(save_dir, model.model+"_folder_"+str(model.opt.folder) + '_G.pkl'))
for key in list(ckpt.keys()):
    print(key)

#state_dict_filt = {k.replace('encoder1', 'signal_encoder'): v for k, v in ckpt.items() if 'encoder1' in k}
state_dict_filt = {k: v for k, v in ckpt.items() if 'transformer_encoder' in k}

print('---------------------------')
for k, v in state_dict_filt.items():
    print(k)
torch.save(state_dict_filt, 'signal_transformer.pkl')

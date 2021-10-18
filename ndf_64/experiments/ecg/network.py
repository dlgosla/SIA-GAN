import os,pickle
import numpy as np
import torch
import torch.nn as nn

from plotUtil import plot_dist,save_pair_fig,save_plot_sample,print_network,save_plot_pair_sample,loss_plot,auc_plot, test_auc_plot
#from network_util import TransformerEncoder, TransformerEncoderLayer

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)

class Generator_Transformer(nn.Module):
    def __init__(self, ninp=128, nhead=8, nhid=512, dropout=0.0, nlayers=3):
        super(Generator_Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.linear1 = nn.Linear(1,ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, input): #[bs,50,1] -> [bs,50,1]
        input = input.squeeze(3) #[bs,50,1,1]->[bs,50,1]

        #- linear [bs,50,1] -> [bs,50,D]
        li = self.linear1(input)
        
        #- transformer [bs,50,D] -> [bs,50,D]
        tf = li.permute(2,0,1) #[D,bs,50]=[token,bs,dim]
        tf = self.transformer_encoder(tf)
        tf = tf.permute(1,2,0)

        #- parse cls token [bs,50,D] -> [bs,50,1]
        #cls_token = tf[:,:,-1].unsqueeze(2).unsqueeze(3)
        #print(cls_token.shape, tf[:,:,-1].unsqueeze(2).shape)
        
        cls_token = tf[:,:,0].unsqueeze(2)
                                
        return cls_token


class Multimodal_Transformer(nn.Module):
    def __init__(self, bs, ntoken=128, ninp=50, nhead=5, nhid=512, dropout=0.0, nlayers=3):
        super(Multimodal_Transformer, self).__init__()
        from torch.nn import TransformerEncoder , TransformerEncoderLayer

        self.ntoken = ntoken
        self.ninp = ninp
        #self.linear1 = nn.Linear(100,50)
        #self.linear2 = nn.Linear(1, 64)
        #self.linear3 = nn.Linear(64, 128)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.transformer_encoder1 = TransformerEncoder(encoder_layers, nlayers)
        #self.transformer_encoder2 = TransformerEncoder(encoder_layers, nlayers)
        #self.transformer_encoder3 = TransformerEncoder(encoder_layers, nlayers)
        
        
    def forward(self, input_s, input_f, input_sf): #[bs,50,1] , [bs,50,1,1]

        input_s = input_s.permute(2,0,1)
        input_f = input_f.permute(2,0,1)
        input_sf = input_sf.permute(2,0,1)
        
        tf = self.transformer_encoder(input_sf)
        #tf = self.transformer_encoder1(src_Q=input_sf, src_K=input_sf, src_V=input_sf) #output = V
        #tf = self.transformer_encoder2(src_Q=input_s, src_K=tf, src_V=tf)
        #tf = self.transformer_encoder3(src_Q=input_s, src_K=tf, src_V=tf)
        
        tf = tf.permute(1,2,0) #[bs,50,128]

        #- parse cls token [bs,50,128] -> [bs,50,1]
        cls_token1 = tf[:,:,0].unsqueeze(2)

        return cls_token1


class Signal_Encoder(nn.Module):
    def __init__(self, ngpu,opt,out_z):
        super(Signal_Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(opt.nc,opt.ndfs,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(opt.ndfs, opt.ndfs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(opt.ndfs * 2, opt.ndfs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(opt.ndfs * 4, opt.ndfs * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(opt.ndfs * 8, opt.ndfs * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(opt.ndfs * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
        


class Signal_Decoder(nn.Module):
    def __init__(self, ngpu,opt):
        super(Signal_Decoder, self).__init__()
        self.ngpu = ngpu
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.nz,opt.ngfs*16,10,1,0,bias=False),
            nn.BatchNorm1d(opt.ngfs*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(opt.ngfs * 16, opt.ngfs * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(opt.ngfs * 8, opt.ngfs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(opt.ngfs * 4, opt.ngfs*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(opt.ngfs * 2, opt.ngfs , 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs ),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(opt.ngfs , opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output
        
class Frequency_1D_Encoder(nn.Module):
    def __init__(self, ngpu,opt,out_z):
        super(Frequency_1D_Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 320
            nn.Conv1d(opt.nc,opt.ndfs,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 160
            nn.Conv1d(opt.ndfs, opt.ndfs * 2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 80
            nn.Conv1d(opt.ndfs * 2, opt.ndfs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 40
            nn.Conv1d(opt.ndfs * 4, opt.ndfs * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 20
            nn.Conv1d(opt.ndfs * 8, opt.ndfs * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ndfs * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 10

            nn.Conv1d(opt.ndfs * 16, out_z, 10, 1, 0, bias=False),
            # state size. (nz) x 1
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class Frequency_2D_Encoder(nn.Module):
    def __init__(self, ngpu,opt,out_z):
        super(Frequency_2D_Encoder, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(opt.nc,opt.ndf,4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 8, opt.ndf * 16, 4, 1, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(opt.ndf * 16, out_z, 7, 1, 0, bias=False),
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output


##

class Frequency_1D_Decoder(nn.Module):
    def __init__(self, ngpu,opt):
        super(Frequency_1D_Decoder, self).__init__()
        self.ngpu = ngpu
        self.main=nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose1d(opt.nz,opt.ngfs*16,10,1,0,bias=False),
            nn.BatchNorm1d(opt.ngfs*16),
            nn.ReLU(True),
            # state size. (ngf*16) x10
            nn.ConvTranspose1d(opt.ngfs * 16, opt.ngfs * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 20
            nn.ConvTranspose1d(opt.ngfs * 8, opt.ngfs * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs * 4),
            nn.ReLU(True),
            # state size. (ngf*2) x 40
            nn.ConvTranspose1d(opt.ngfs * 4, opt.ngfs*2, 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs*2),
            nn.ReLU(True),
            # state size. (ngf) x 80
            nn.ConvTranspose1d(opt.ngfs * 2, opt.ngfs , 4, 2, 1, bias=False),
            nn.BatchNorm1d(opt.ngfs ),
            nn.ReLU(True),
            # state size. (ngf) x 160
            nn.ConvTranspose1d(opt.ngfs , opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 320
        )
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output

class Frequency_2D_Decoder(nn.Module):
    def __init__(self, ngpu,opt):
        super(Frequency_2D_Decoder, self).__init__()
        self.ngpu = ngpu
        self.main=nn.Sequential(
            nn.ConvTranspose2d(opt.nz,opt.ngf*16,7,1,0,bias=False),
            nn.BatchNorm2d(opt.ngf*16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 16, opt.ngf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 2, opt.ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf ),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf , opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()

        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class AD_MODEL(object):
    def __init__(self,opt,dataloader,device):
        self.G=None
        self.D=None

        self.opt=opt
        self.niter=opt.niter
        self.dataset=opt.dataset
        self.model = opt.model
        self.outf=opt.outf


    def  train(self):
        raise NotImplementedError

    def visualize_results(self, epoch,samples,is_train=True):
        if is_train:
            sub_folder="train"
        else:
            sub_folder="test"

        save_dir=os.path.join(self.outf,self.model,self.dataset,sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_sample(samples, epoch, self.dataset, num_epochs=self.niter,
                         impath=os.path.join(save_dir,'epoch%03d' % epoch + '.png'))


    def visualize_pair_results(self,epoch,samples1,samples2,is_train=True,sample_type='s'):
        if is_train:        
            #sub_folder="train"
            if sample_type == 's':
                sub_folder = "train/sig"
            else:
                sub_folder = 'train/freq'                                                  
        else:                
            #sub_folder="test"
            if sample_type == 's':
                sub_folder = "test/sig"
            else:
                sub_folder = 'test/freq'            

        save_dir=os.path.join(self.outf,self.model,self.dataset,sub_folder)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_plot_pair_sample(samples1, samples2, epoch, self.dataset, num_epochs=self.niter, impath=os.path.join(save_dir,'epoch%03d' % epoch + '.png'))

    def save(self,train_hist):
        save_dir = os.path.join(self.outf, self.model, self.dataset,"model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, self.model + '_history.pkl'), 'wb') as f:
            pickle.dump(train_hist, f)

    def save_weight_GD_S(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G_signal.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D_signal.pkl'))


    def save_weight_GD_F(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset, "model")



    def load(self):
        save_dir = os.path.join(self.outf, self.model, self.dataset,"model")
        
        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G_signal.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D_signal.pkl')))
        
        #self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_G.pkl')))
        #self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model+"_folder_"+str(self.opt.folder) + '_D.pkl')))


    def save_loss(self,train_hist):
        loss_plot(train_hist, os.path.join(self.outf, self.model, self.dataset), self.model)


    def save_auc(self, train_hist):
        auc_plot(train_hist, os.path.join(self.outf, self.model, self.dataset), self.model)

    def save_test_auc(self, train_hist):
        test_auc_plot(train_hist, os.path.join(self.outf, self.model, self.dataset), self.model)



    def saveTestPair(self,pair,save_dir):
        '''
        :param pair: list of (input,output)
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        for idx,p in enumerate(pair):
            input=p[0]
            output=p[1]
            save_pair_fig(input,output,os.path.join(save_dir,str(idx)+".png"))




    def analysisRes(self,N_res,A_res,min_score,max_score,threshold,save_dir):
        '''
        :param N_res: list of normal score
        :param A_res:  dict{ "S": list of S score, "V":...}
        :param min_score:
        :param max_score:
        :return:
        '''
        print("############   Analysis   #############")
        print("############   Threshold:{}   #############".format(threshold))
        all_abnormal_score=[]
        all_normal_score=np.array([])
        for a_type in A_res:
            a_score=A_res[a_type]
            print("*********  Type:{}  *************".format(a_type))
            normal_score=normal(N_res, min_score, max_score)
            abnormal_score=normal(a_score, min_score, max_score)
            all_abnormal_score=np.concatenate((all_abnormal_score,np.array(abnormal_score)))
            all_normal_score=normal_score
            plot_dist(normal_score,abnormal_score , str(self.opt.folder)+"_"+"N", a_type,
                      save_dir)

            TP=np.count_nonzero(abnormal_score >= threshold)
            FP=np.count_nonzero(normal_score >= threshold)
            TN=np.count_nonzero(normal_score < threshold)
            FN=np.count_nonzero(abnormal_score<threshold)
            print("TP:{}".format(TP))
            print("FP:{}".format(FP))
            print("TN:{}".format(TN))
            print("FN:{}".format(FN))
            print("Accuracy:{}".format((TP + TN) * 1.0 / (TP + TN + FP + FN)))
            print("Precision/ppv:{}".format(TP * 1.0 / (TP + FP)))
            print("sensitivity/Recall:{}".format(TP * 1.0 / (TP + FN)))
            print("specificity:{}".format(TN * 1.0 / (TN + FP)))
            print("F1:{}".format(2.0 * TP / (2 * TP + FP + FN)))

        # all_abnormal_score=np.reshape(np.array(all_abnormal_score),(-1))
        # print(all_abnormal_score.shape)
        plot_dist(all_normal_score, all_abnormal_score, str(self.opt.folder)+"_"+"N", "A",
                 save_dir)






def normal(array,min_val,max_val):
    return (array-min_val)/(max_val-min_val)


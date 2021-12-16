import torch

import time,os,sys
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from network import Multimodal_Transformer, Signal_Encoder, Signal_Decoder, Frequency_Encoder, Frequency_Decoder, AD_MODEL, weights_init, print_network
from network_util import ArcMarginProduct

dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))

from metric import evaluate



class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        
        #- signal
        model_s = Signal_Encoder(opt.ngpu,opt,1)
        layers_s = list(model_s.main.children())

        self.features_s = nn.Sequential(*layers_s[:-1])
        self.classifier_s = nn.Sequential(layers_s[-1])
        self.classifier_s.add_module('Sigmoid', nn.Sigmoid())

        #- frequency
        model_f = Frequency_Encoder(opt.ngpu,opt,1)
        layers_f = list(model_f.main.children())

        self.features_f = nn.Sequential(*layers_f[:-1])
        self.classifier_f = nn.Sequential(layers_f[-1])
        self.classifier_f.add_module('Sigmoid', nn.Sigmoid())


    def forward(self, x_sig):
        #- signal
        features_signal = self.features_s(x_sig)
        features_signal = features_signal
        classifier_signal = self.classifier_s(features_signal)
        classifier_signal = classifier_signal.view(-1, 1).squeeze(1)
        #- frequency
        # features_freq = self.features_f(x_freq)
        # features_freq = features_freq
        # classifier_freq = self.classifier_f(features_freq)
        # classifier_freq = classifier_freq.view(-1, 1).squeeze(1)

        return classifier_signal, features_signal #, classifier_freq, features_freq


class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.signal_encoder = Signal_Encoder(opt.ngpu,opt,opt.nz)
        self.freq_encoder = Frequency_Encoder(opt.ngpu,opt,opt.nz)
        
        #self.metric_fc = ArcMarginProduct(512, 2, s=30, m=0.5)
        self.arcface = ArcMarginProduct(50, 50)
        
        self.opt = opt
        self.tf = Multimodal_Transformer(bs=opt.batchsize, ntoken=128, ninp=50, nhead=5, nhid=512, dropout=0.0, nlayers=3)
        self.label = torch.empty(size=(opt.batchsize,), dtype=torch.float32, device="cuda:0")
        
        self.signal_decoder = Signal_Decoder(opt.ngpu,opt)
        self.freq_decoder = Frequency_Decoder(opt.ngpu,opt)


    def forward(self, x_sig, x_freq):
        latent_signal = self.signal_encoder(x_sig) # unimodal encoder [bs,50,1]
        latent_freq = self.freq_encoder(x_freq) #[bs,50,1,1]
        
        #label = 0
        #label = label.to(device).long()
        #print(label.shape, "label.shape")
        #print(label, "label")
        
        
        cls_signal, cls_freq = self.tf(latent_signal, latent_freq) # multimodal transformer
        #label = torch.empty(size=(self.opt.batchsize, 1), device="cuda:0")
        #label.data.fill_(0)
        
        cls_signal = cls_signal.squeeze()
        
        arc_cls_signal = []
        for cls in cls_signal:
          label = torch.empty(size=(1, 1), device="cuda:0")
          label.data.fill_(0)
          cls = cls.reshape(1,50)
          #print(cls.shape, cls)
          arc = self.arcface(cls, label)
          arc = arc.reshape(50)
          #print(arc.shape)
          
          arc_cls_signal.append(arc.tolist())
  
        
        cls_freq = cls_freq.squeeze()
        arc_cls_freq = []
        for cls in cls_freq:
          label = torch.empty(size=(1, 1), device="cuda:0")
          label.data.fill_(1)
          cls = cls.reshape(1,50)
          #print(cls.shape, cls)
          arc = self.arcface(cls, label)
          arc = arc.reshape(50)
          #print(arc.shape)
          
          arc_cls_freq.append(arc.tolist())
        
        
        arc_cls_signal = torch.Tensor(arc_cls_signal).unsqueeze(2).to("cuda:0")
        
        #arc_cls_siganl = arc_cls_signal.unsqueeze(1)
        
        
        arc_cls_freq = torch.Tensor(arc_cls_freq).unsqueeze(2).to("cuda:0")
        #arc_cls_freq = arc_cls_signal.unsqueeze(2)
        
        #print(arc_cls_signal.shape, "signal")
        #print(arc_cls_freq.shape, "freq")
        
        #print("cls", cls_signal.shape, cls_signal)
        #arc_cls_signal = self.arcface(cls_signal, label)
        #print(arc_cls_signal.shape, arc_cls_signal, "arc")
        
        
        '''
        arc_cls_signal = []
        for cls in cls_signal:
            label = torch.empty(size=(1, 1), device="cuda:0")
            label.data.fill_(0)
            self.arcface(cls, label)
            print(self.arcface(cls, label).shape)
            arc_cls_signal.append(self.arcface(cls, label))
        
            
        arc_cls_signal = torch.Tensor(arc_cls_signal)
        print(arc_cls_signal.shape, "arc signal")
        '''
            
        #print(latent)            
            
        
        #label = np.int32(1)
        
        '''
        print(label, "label",label.shape )
        cls_signal = self.arcface(cls_signal.squeeze(), label)
        
        label = torch.empty(size=(self.opt.batchsize,1), device="cuda:0")
        label.data.fill_(1)
        
        print(label, "label",label.shape )
        cls_freq = self.arcface(cls_freq.squeeze(), label)
        '''
        
        '''
        self.label.data.resize_(self.opt.batchsize).fill_(0)
        
        print("signal", cls_signal.shape)
        cls_signal = self.arcface(cls_signal, self.label )
        print("signal", cls_signal.shape)
        
        print("freq", cls_freq.shape)
        cls_freq = self.arcface(cls_freq, torch.FloatTensor(1).to('cuda'))
        print("freq", cls_freq.shape)
        '''

        gen_signal1 = self.signal_decoder(arc_cls_signal) #unimodal decoder
        gen_signal2 = self.signal_decoder(arc_cls_freq)
        gen_signal = (gen_signal1 + gen_signal2) / 2.0
        # gen_freq = self.freq_decoder(cls_freq.unsqueeze(3))

        return gen_signal, latent_signal, latent_freq


class BeatGAN(AD_MODEL):


    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)
        if not self.opt.istest:
            print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        if not self.opt.istest:
            print_network(self.D)
      
        #metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)


        # #- load pretrained frequency encoder model
        # pretrained_model_f = '/root/vol1/b2igan/multimodal_network/frequency_freeze/freq_to_signal/experiments/ecg/freq_encoder.pkl'
        # self.G.load_state_dict(torch.load(pretrained_model_f), strict=False)

        
#        for k in self.G.state_dict():
#            print(k)
        #- freeze frequency encoder model
#        for name, param in self.G.named_parameters():
#            if 'freq_encoder' in name:
#                param.requires_grad = False
                # print(name)



        self.bce_criterion = nn.BCELoss().cuda()
        self.mse_criterion=nn.MSELoss().cuda()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        self.input_s = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.input_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt_s    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.gt_f    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input_s = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize),dtype=torch.float32, device=self.device)
        self.fixed_input_f = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize),dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label= 0


        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['D_loss_s'] = []
        self.train_hist['D_loss_f'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['G_loss_s'] = []
        self.train_hist['G_loss_f'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.train_hist['auc_s']=[]
        self.train_hist['auc_f']=[]

        print("Train model.")
        start_time = time.time()
        best_auc=0
        best_auc_epoch=0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch+=1

               #- train, validation
                self.train_epoch()
                auc_s,th_s,f1_s=self.validate()
                self.train_hist["auc_s"].append(auc_s)
#                self.train_hist["auc_f"].append(auc_f)
                self.save(self.train_hist)
                self.save_loss(self.train_hist)
                self.save_auc(self.train_hist)

                #- test
                # if epoch%10==0:
                #     test_auc = self.train_test_type()
                #     self.test_hist["auc"].append(test_auc)
                #     self.save_test_auc(self.test_hist)

                if auc_s > best_auc: #val auc
                    best_auc = auc_s
                    best_auc_epoch=self.cur_epoch
                    self.save_weight_GD()


                f.write("[{}] auc_s:{:.4f}\t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc_s,best_auc,best_auc_epoch))

                print("[{}] auc_s:{:.4f} th_s:{:.4f} f1_s:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc_s,th_s,f1_s,best_auc,best_auc_epoch))
                # print("[{}] auc_f:{:.4f} th_f:{:.4f} f1_f:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc_f,th_f,f1_f,best_auc,best_auc_epoch))


        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))


    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0

        err_d, err_g, err_d_s, err_d_f, err_g_s, err_g_f = 0., 0., 0., 0., 0., 0.
        num_batch = len(self.dataloader["train"])

        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()
            err_d += errors["err_d"]
            err_d_s += errors["err_d_s"]
            # err_d_f += errors["err_d_f"]
            err_g += errors["err_g"]
            err_g_s += errors["err_g_s"]
            err_g_f += errors["err_g_f"]


            if (epoch_iter  % self.opt.print_freq) == 0:
                print("Epoch: [%d] [%4d/%4d] D_loss(s): %.6f, G_loss(s): %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d_s"], errors["err_g_s"]))
                       
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        self.train_hist['D_loss'].append(err_d/num_batch)
        self.train_hist['D_loss_s'].append(err_d_s/num_batch)
        # self.train_hist['D_loss_f'].append(err_d_f/num_batch)
        self.train_hist['G_loss'].append(err_g/num_batch)
        self.train_hist['G_loss_s'].append(err_g_s/num_batch)
        self.train_hist['G_loss_f'].append(err_g_f/num_batch)
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        with torch.no_grad():
            real_input,fake_output = self.get_generated_x()

            self.visualize_pair_results(self.cur_epoch,
                                        real_input,
                                        fake_output,
                                        is_train=True)


    def set_input(self, input):
        self.input_s.resize_(input[0][0].size()).copy_(input[0][0])
        self.gt_s.resize_(input[0][1].size()).copy_(input[0][1])

        self.input_f.resize_(input[1][0].size()).copy_(input[1][0])
        self.gt_f.resize_(input[1][1].size()).copy_(input[1][1])


        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            self.fixed_input_s.resize_(input[0][0].size()).copy_(input[0][0])
            self.fixed_input_f.resize_(input[1][0].size()).copy_(input[1][0])


    def optimize(self):
        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        # if self.err_d.item() < 5e-6:
        #     self.reinitialize_netd()


    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_d_real_s, self.feat_real_s = self.D(self.input_s) #, self.input_f)
        
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake_s,_, _ = self.G(self.input_s, self.input_f)
        self.out_d_fake_s, self.feat_fake_s = self.D(self.fake_s) #, self.fake_f)
        # --

        # for name, param in self.G.named_parameters():
        #     if 'freq_encoder.main.0.weight' in name:
        #         # param.requires_grad = False
        #         print(name)
        #         print(param)

        self.err_d_real_s = self.bce_criterion(self.out_d_real_s, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.float32).cuda())
        self.err_d_fake_s = self.bce_criterion(self.out_d_fake_s, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.float32).cuda())

        # self.err_d_real_f = self.bce_criterion(self.out_d_real_f, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.float32).cuda())
        # self.err_d_fake_f = self.bce_criterion(self.out_d_fake_f, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.float32).cuda())

        self.err_d_s = self.err_d_real_s + self.err_d_fake_s
        # self.err_d_f = self.err_d_real_f + self.err_d_fake_f
        self.err_d = self.err_d_s # + self.err_d_f

        self.err_d.backward()
        self.optimizerD.step()


    # def reinitialize_netd(self):
    #     """ Initialize the weights of netD
    #     """
    #     self.D.apply(weights_init)
    #     print('Reloading d net')


    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake_s, self.latent_s, self.latent_f = self.G(self.input_s, self.input_f)
        self.out_d_fake_s, self.feat_fake_s = self.D(self.fake_s) #, self.fake_f)
        self.out_d_real_s, self.feat_real_s = self.D(self.input_s) #, self.input_f)

        # self.err_g_dist = torch.max(1-torch.dist(self.latent_s,self.latent_f,2), torch.zeros(1).cuda())
        

        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv_s = self.mse_criterion(self.feat_fake_s, self.feat_real_s)  # loss for feature matching
        self.err_g_rec_s = self.mse_criterion(self.fake_s, self.input_s)  # constrain x' to look like x

        # self.err_g_adv_f = self.mse_criterion(self.feat_fake_f, self.feat_real_f)  # loss for feature matching
        # self.err_g_rec_f = self.mse_criterion(self.fake_f, self.input_f)  # constrain x' to look like x

        self.err_g_s =  self.err_g_rec_s + self.err_g_adv_s # + self.err_g_dist#* self.opt.w_adv
        # self.err_g_f =  self.err_g_rec_f + self.err_g_adv_f
        self.err_g = self.err_g_s # + self.err_g_f
        self.err_g.backward()
        self.optimizerG.step()


    ##
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_s': self.err_d_s.item(),
                    # 'err_d_f': self.err_d_f.item(),
                    'err_g_s': self.err_g_s.item(),
                    'err_g_f': self.err_g_s.item(),
                  }


        return errors



    def get_generated_x(self):
        fake, _, _ = self.G(self.fixed_input_s, self.fixed_input_f)    
        return  self.fixed_input_s.cpu().data.numpy(), fake.cpu().data.numpy()


    def train_test_type(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False)
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)
        over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
        over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
        min_score,max_score=np.min(over_all),np.max(over_all)
        A_res={
            "S":y_pred_S,
            "V":y_pred_V,
            "F":y_pred_F,
            "Q":y_pred_Q
        }
        self.analysisRes(y_pred_N,A_res,min_score,max_score,res_th,save_dir)

        aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#################################")
        print("########## Test Result ##########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))
        return aucroc


    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_s, y_pred_s=self.predict(self.dataloader["val"])
        rocprc_s,rocauc_s,best_th_s,best_f1_s=evaluate(y_s, y_pred_s)
#        rocprc_f,rocauc_f,best_th_f,best_f1_f=evaluate(y_f, y_pred_f)
        return rocauc_s,best_th_s,best_f1_s #, rocauc_f,best_th_f,best_f1_f


    def predict(self,dataloader_,scale=True):
        with torch.no_grad():

            self.an_scores_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            # self.an_scores_f = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            # self.gt_labels_f = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            # self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
            #                             device=self.device)


            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake_s, _, _ = self.G(self.input_s, self.input_f)

                error_s = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake_s.view(self.fake_s.shape[0], -1)), 2),
                    dim=1)
                self.an_scores_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = error_s.reshape(error_s.size(0))
                self.gt_labels_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = self.gt_s.reshape(error_s.size(0))


                # error_f = torch.mean(
                #     torch.pow((self.input_f.view(self.input_f.shape[0], -1) - self.fake_f.view(self.fake_f.shape[0], -1)), 2),
                #     dim=1)
                # self.an_scores_f[i*self.opt.batchsize : i*self.opt.batchsize+error_f.size(0)] = error_f.reshape(error_f.size(0))
                # self.gt_labels_f[i*self.opt.batchsize : i*self.opt.batchsize+error_f.size(0)] = self.gt_f.reshape(error_f.size(0))


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores_s = (self.an_scores_s - torch.min(self.an_scores_s)) / (torch.max(self.an_scores_s) - torch.min(self.an_scores_s))
                # self.an_scores_f = (self.an_scores_f - torch.min(self.an_scores_f)) / (torch.max(self.an_scores_f) - torch.min(self.an_scores_f))

            y_s=self.gt_labels_s.cpu().numpy()
            y_pred_s=self.an_scores_s.cpu().numpy()

            # y_f=self.gt_labels_f.cpu().numpy()
            # y_pred_f=self.an_scores_f.cpu().numpy()

            return y_s, y_pred_s #, y_f, y_pred_f

    def predict_for_right(self,dataloader_,min_score,max_score,threshold,save_dir):
        '''

        :param dataloader:
        :param min_score:
        :param max_score:
        :param threshold:
        :param save_dir:
        :return:
        '''
        assert  save_dir is not None
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.G.eval()
        self.D.eval()
        with torch.no_grad():
            # Create big error tensor for the test set.
            test_pair=[]
            self.an_scores_s = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake_s, _, _= self.G(self.input_s, self.input_f)

                error_s = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake_s.view(self.fake_s.shape[0], -1)), 2),
                    dim=1)
                # self.an_scores_s[i*self.opt.batchsize : i*self.opt.batchsize+error_s.size(0)] = error_s.reshape(error_s.size(0))

                # error_f = torch.mean(
                #     torch.pow((self.input_f.view(self.input_f.shape[0], -1) - self.fake_f.view(self.fake_f.shape[0], -1)), 2),
                #     dim=1)
                # self.an_scores_f[i*self.opt.batchsize : i*self.opt.batchsize+error_f.size(0)] = error_f.reshape(error_f.size(0))

                # Save test images.
                batch_input = self.input_s.cpu().numpy()
                batch_output = self.fake_s.cpu().numpy()
                ano_score=error_s.cpu().numpy()
                assert batch_output.shape[0]==batch_input.shape[0]==ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair)>=100:
                        break
                    normal_score=(ano_score[idx]-min_score)/(max_score-min_score)

                    if normal_score>=threshold:
                        test_pair.append((batch_input[idx],batch_output[idx]))

            # print(len(test_pair))
            #- plot test images
            self.saveTestPair(test_pair,save_dir)


    def test_type_signal(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False)
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)

        over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
        over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
        min_score,max_score=np.min(over_all),np.max(over_all)
        A_res={
            "S":y_pred_S,
            "V":y_pred_V,
            "F":y_pred_F,
            "Q":y_pred_Q
        }
        self.analysisRes(y_pred_N,A_res,min_score,max_score,res_th,save_dir)


        #save fig for Interpretable
        self.predict_for_right(self.dataloader["test_N"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "N"))
        self.predict_for_right(self.dataloader["test_S"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "S"))
        self.predict_for_right(self.dataloader["test_V"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "V"))
        self.predict_for_right(self.dataloader["test_F"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "F"))
        self.predict_for_right(self.dataloader["test_Q"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "Q"))
        aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
        print("#############################")
        print("########  Result  ###########")
        print("ap:{}".format(aucprc))
        print("auc:{}".format(aucroc))
        print("best th:{} --> best f1:{}".format(best_th,best_f1))


        with open(os.path.join(save_dir,"res-record.txt"),'w') as f:
            f.write("auc_prc:{}\n".format(aucprc))
            f.write("auc_roc:{}\n".format(aucroc))
            f.write("best th:{} --> best f1:{}".format(best_th, best_f1))

    # def test_type_freq(self):
    #     self.G.eval()
    #     self.D.eval()
    #     res_th=self.opt.threshold
    #     save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

    #     if not os.path.exists(save_dir):
    #         os.makedirs(save_dir)

    #     _, _, y_N, y_pred_N = self.predict(self.dataloader["test_N"],scale=False)
    #     _, _, y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False)
    #     _, _, y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False)
    #     _, _, y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False)
    #     _, _, y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False)

    #     over_all=np.concatenate([y_pred_N,y_pred_S,y_pred_V,y_pred_F,y_pred_Q])
    #     over_all_gt=np.concatenate([y_N,y_S,y_V,y_F,y_Q])
    #     min_score,max_score=np.min(over_all),np.max(over_all)
    #     A_res={
    #         "S":y_pred_S,
    #         "V":y_pred_V,
    #         "F":y_pred_F,
    #         "Q":y_pred_Q
    #     }
    #     self.analysisRes(y_pred_N,A_res,min_score,max_score,res_th,save_dir)


    #     #save fig for Interpretable
    #     self.predict_for_right(self.dataloader["test_N"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "N"))
    #     self.predict_for_right(self.dataloader["test_S"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "S"))
    #     self.predict_for_right(self.dataloader["test_V"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "V"))
    #     self.predict_for_right(self.dataloader["test_F"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "F"))
    #     self.predict_for_right(self.dataloader["test_Q"], min_score,max_score,res_th,save_dir=os.path.join(save_dir, "Q"))
    #     aucprc,aucroc,best_th,best_f1=evaluate(over_all_gt,(over_all-min_score)/(max_score-min_score))
    #     print("#############################")
    #     print("########  Result  ###########")
    #     print("ap:{}".format(aucprc))
    #     print("auc:{}".format(aucroc))
    #     print("best th:{} --> best f1:{}".format(best_th,best_f1))


    #     with open(os.path.join(save_dir,"res-record.txt"),'w') as f:
    #         f.write("auc_prc:{}\n".format(aucprc))
    #         f.write("auc_roc:{}\n".format(aucroc))
    #         f.write("best th:{} --> best f1:{}".format(best_th, best_f1))



    def test_time(self):
        self.G.eval()
        self.D.eval()
        size=self.dataloader["test_N"].dataset.__len__()
        start=time.time()

        for i, (data_x,data_y) in enumerate(self.dataloader["test_N"], 0):
            input_x=data_x
            for j in range(input_x.shape[0]):
                input_x_=input_x[j].view(1,input_x.shape[1],input_x.shape[2]).to(self.device)
                digit, _, gen_x = self.G(input_x_)

                error = torch.mean(
                    torch.pow((digit.view(digit.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end=time.time()
        print((end-start)/size)

import cv2
import time,os,sys
import librosa, librosa.display 

from matplotlib import pyplot as plt

import time,os,sys


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import Generator_Transformer, AD_MODEL,weights_init,print_network,Frequency_1D_Decoder, Frequency_2D_Decoder, Frequency_1D_Encoder, Frequency_2D_Encoder



dirname=os.path.dirname
sys.path.insert(0,dirname(dirname(os.path.abspath(__file__))))

from metric import evaluate


##
class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        #model = Encoder(opt.ngpu,opt,1)
        model = Frequency_1D_Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features




##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder1 = Frequency_2D_Encoder(opt.ngpu,opt,opt.nz)
        self.tf = Generator_Transformer(ninp=50, nhead=5, nhid=512, dropout=0.0, nlayers=3)
        self.decoder = Frequency_1D_Decoder(opt.ngpu,opt)

    def forward(self, x):
        latent_i = self.encoder1(x)
        tf = self.tf(latent_i) # [bs,50,1] -> [bs,50,1]
        gen_x = self.decoder(tf)
        return gen_x, latent_i


class BeatGAN(AD_MODEL):


    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__(opt, dataloader, device)
        self.dataloader = dataloader
        self.device = device
        self.opt=opt

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator( opt).to(device)
        self.G.apply(weights_init)
        if not self.opt.istest:
            print_network(self.G)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)
        if not self.opt.istest:
            print_network(self.D)


        self.bce_criterion = nn.BCELoss().cuda()
        self.mse_criterion=nn.MSELoss().cuda()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        #self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        #self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        #self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        #self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize),dtype=torch.float32, device=self.device)
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
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []




        print("Train model.")
        start_time = time.time()
        best_auc=0
        best_auc_epoch=0

        with open(os.path.join(self.outf, self.model, self.dataset, "val_info.txt"), "w") as f:
            for epoch in range(self.niter):
                self.cur_epoch+=1
                self.train_epoch()
                auc,th,f1=self.validate()
                if auc > best_auc:
                    best_auc = auc
                    best_auc_epoch=self.cur_epoch
                    self.save_weight_GD()
                f.write("[{}] auc:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc,best_auc,best_auc_epoch ))
                print("[{}] auc:{:.4f} th:{:.4f} f1:{:.4f} \t best_auc:{:.4f} in epoch[{}]\n".format(self.cur_epoch,auc,th,f1,best_auc,best_auc_epoch ))



        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        self.save(self.train_hist)

        self.save_loss(self.train_hist)



    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()

            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (epoch_iter  % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
                #print("err_adv:{}  ,err_rec:{}  ,err_enc:{}".format(errors["err_g_adv"],errors["err_g_rec"],errors["err_g_enc"]))

        #fig = plt.figure(figsize=(5,5)) 
        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        with torch.no_grad():
            real_input,fake_output = self.get_generated_x()
            #print("real_input.shpae", real_input.shape)
            #img = librosa.display.specshow(real_input[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
            #fig.savefig("train.png")

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
        '''  
        #self.input.data.resize_(input[0].size()).copy_(input[0])
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
        #self.gt.data.resize_(input[1].size()).copy_(input[1])
        with torch.no_grad():
            self.gt.resize_(input[1].size()).copy_(input[1])

        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            #self.fixed_input.data.resize_(input[0].size()).copy_(input[0])
            with torch.no_grad():
                self.fixed_input.resize_(input[0].size()).copy_(input[0])
        ''' 

    ##
    def optimize(self):
        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        #if self.err_d.item() < 5e-6:
        #    self.reinitialize_netd()

    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.D(self.input_s)
        
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake, self.latent_i = self.G(self.input_f)
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --

        #to~~~ delete if some problem will happen
        #print('self.out_d_real: ',type(self.out_d_real))
        #print('self.device: ', self.device)
        #print('self.batchsize: ', self.batchsize)
        #print('self.real_label: ',self.real_label)
        #print(self.real_label.device)
        #print('torch.full: ', torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor))
        
        
        #below the code, if .cuda() isn't exist, torch.full.device is cpu
        #print('torch.full is go to cpu?: ', torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.FloatTensor).cuda().device)
        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device).type(torch.float32).cuda())

        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device).type(torch.float32).cuda())


        self.err_d=self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()

    ##
    #def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
    #    self.D.apply(weights_init)
    #    print('Reloading d net')

    ##
    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input_f)
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input_s)


        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv=self.mse_criterion(self.feat_fake,self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(self.fake, self.input_s)  # constrain x' to look like x


        self.err_g =  self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()


    ##
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }


        return errors

        ##

    def get_generated_x(self):
        #print(self.input_f.shape, self.fixed_input_f.shape, self.input_f == self.fixed_input_f)
        fake = self.G(self.fixed_input_f)[0]

        return  self.fixed_input_s.cpu().data.numpy(),fake.cpu().data.numpy()

    ##


    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        results_path = os.path.join("results", "train")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        y_,y_pred=self.predict(self.dataloader["val"],is_train=True)
        rocprc,rocauc,best_th,best_f1=evaluate(y_,y_pred)
        return rocauc,best_th,best_f1

    def predict(self,dataloader_,scale=True, path="/",is_train=False):
        with torch.no_grad():

            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)
                                        
                
            #fig = plt.figure(figsize=(5,5))
            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input_f)
                
                #visualization
                if i==0 and is_train:
                     real_s = self.input_s.cpu().numpy()
                     real_f = self.input_f.cpu().numpy()
                     fake_f = self.fake.cpu().numpy()
                     
                     #--save real signal/real freq
                     fig = plt.figure(figsize=(5,10))
                     plt.subplot(2,1,1)
                     plt.plot(real_s[0][0])
                     
                     plt.subplot(2,1,2)
                     img = librosa.display.specshow(real_f[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
                    
                     fig.savefig("results/train/real_epoch{0}.png".format(self.cur_epoch))
                     #--
                     
                     #--save real signal/fake siganl
                     fig = plt.figure(figsize=(5,10))
                     plt.subplot(2,1,1)
                     plt.plot(real_s[0][0])
                    
                     plt.subplot(2,1,2)
                     plt.plot(fake_f[0][0])
                     fig.savefig("results/train/fake_epoch{0}.png".format(self.cur_epoch))
                     #--

                # error = torch.mean(torch.pow((d_feat.view(self.input.shape[0],-1)-d_gen_feat.view(self.input.shape[0],-1)), 2), dim=1)
                #
                error = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt_f.reshape(error.size(0))
#                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_=self.gt_labels.cpu().numpy()
            y_pred=self.an_scores.cpu().numpy()

            return y_,y_pred


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
            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)

            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input_f)

                error = torch.mean(
                    torch.pow((self.input_s.view(self.input_s.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                
                
                #if i==0 and data_type=='f':
                if i==0:
                     real_s = self.input_s.cpu().numpy()
                     real_f = self.input_f.cpu().numpy()
                     fake_f = self.fake.cpu().numpy()
                     
                     #--save real signal/real freq
                     fig = plt.figure(figsize=(5,10))
                     plt.subplot(2,1,1)
                     plt.plot(real_s[0][0])
                     
                     plt.subplot(2,1,2)
                     img = librosa.display.specshow(real_f[0][0], sr=360, hop_length = 2, y_axis="linear", x_axis="time")
                    
                     fig.savefig("results/test/real_{0}.png".format(save_dir[-1]))
                     #--
                     
                     #--save real signal/fake siganl
                     fig = plt.figure(figsize=(5,10))
                     plt.subplot(2,1,1)
                     plt.plot(real_s[0][0])
                    
                     plt.subplot(2,1,2)
                     plt.plot(fake_f[0][0])
                     fig.savefig("results/test/fake_{0}.png".format(save_dir[-1]))
                     #--
                     

                # # Save test images.

                batch_input = self.input_s.cpu().numpy()
                batch_output = self.fake.cpu().numpy()
                ano_score=error.cpu().numpy()
                assert batch_output.shape[0]==batch_input.shape[0]==ano_score.shape[0]
                for idx in range(batch_input.shape[0]):
                    if len(test_pair)>=100:
                        break
                    normal_score=(ano_score[idx]-min_score)/(max_score-min_score)

                    if normal_score>=threshold:
                        test_pair.append((batch_input[idx],batch_output[idx]))

            # print(len(test_pair))
            self.saveTestPair(test_pair,save_dir)



    def test_type(self):
        self.G.eval()
        self.D.eval()
        res_th=self.opt.threshold
        save_dir = os.path.join(self.outf, self.model, self.dataset, "test", str(self.opt.folder))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        results_path = os.path.join("results", "test")
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        y_N, y_pred_N=self.predict(self.dataloader["test_N"],scale=False, path=os.path.join(save_dir, "N"))
        y_S, y_pred_S = self.predict(self.dataloader["test_S"],scale=False, path=os.path.join(save_dir, "S"))
        y_V, y_pred_V = self.predict(self.dataloader["test_V"],scale=False, path=os.path.join(save_dir, "V"))
        y_F, y_pred_F = self.predict(self.dataloader["test_F"],scale=False,path=os.path.join(save_dir, "F"))
        y_Q, y_pred_Q = self.predict(self.dataloader["test_Q"],scale=False, path=os.path.join(save_dir, "Q"))
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
        # self.predictForRight(self.dataloader["test_N"], save_dir=os.path.join(save_dir, "N"))
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

    def test_time(self):
        self.G.eval()
        self.D.eval()
        size=self.dataloader["test_N"].dataset.__len__()
        start=time.time()

        for i, (data_x,data_y) in enumerate(self.dataloader["test_N"], 0):
            input_x=data_x
            for j in range(input_x.shape[0]):
                input_x_=input_x[j].view(1,input_x.shape[1],input_x.shape[2]).to(self.device)
                gen_x,_ = self.G(input_x_)

                error = torch.mean(
                    torch.pow((input_x_.view(input_x_.shape[0], -1) - gen_x.view(gen_x.shape[0], -1)), 2),
                    dim=1)

        end=time.time()
        print((end-start)/size)

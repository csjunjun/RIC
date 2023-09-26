# Imports necessary libraries and modules
#from itertools import islice
from distutils.log import error
import os
import shutil
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #56160 a = 0.1, 10187 svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@2_correctbest_resumefrom139
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch import utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pickle
from torchvision import datasets, utils
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from random import shuffle
from torchvision.utils import make_grid,save_image
#from IPython.display import Image
from utee import selector
import juncw
from RIC import Net
import PerceptualSimilarity.models
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def getCEloss(labels,outputs):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

    return torch.clamp((i-j), min=-kappa)
    
def attack1_loss(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
    loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
    outputs = target_model(mix_img)
    classloss = torch.mean(getCEloss(labels,outputs))
    
    loss_all =   B*loss_secret  + classloss
    
    return loss_all, loss_secret,classloss,loss_cover


def valmetric(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    outputs = target_model(mix_img)
    pre = torch.argmax(outputs,dim=1)
    acc_num = len(torch.where(pre==labels)[0])    
    psnr, ssim, mse_s,mse_c,mean_pixel_error,lpips_error = [], [], [],[],[],[]
    norm_S =  convert1(S*_std_torch+_mean_torch)
    norm_S_prime = convert1(S_prime)
    norm_mix_image = convert1(mix_img)
    mse_sc,psnr_sc,ssim_sc,lpips_sc = [], [], [],[]
    for i in range(len(S_prime)):
        mse_s.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, multichannel=True))
        mean_pixel_error.append(float(torch.sum(torch.abs(torch.round((S*_std_torch+_mean_torch)[i]*255)-torch.round(S_prime[i]*255)))/(3*imgsize*imgsize)))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], S_prime[i],normalize=True)
        lpips_error.append(float(tmp))
        #mix_image and secret image
        mse_sc.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-mix_img[i])))
        psnr_sc.append(peak_signal_noise_ratio(norm_S[i],norm_mix_image[i],data_range=255))
        ssim_sc.append(structural_similarity(norm_S[i],norm_mix_image[i],win_size=11, data_range=255.0, multichannel=True))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], mix_img[i],normalize=True)
        lpips_sc.append(float(tmp))
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error),np.sum(lpips_error),np.sum(lpips_sc),np.sum(mse_sc),np.sum(psnr_sc),np.sum(ssim_sc)




def load_checkpoint(filepath,net_local=None):
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)

        start_epoch = checkpoint['epoch']
        if net_local is None:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net_local.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))

def getAdvZ(img_size,labels,batch_size,minibs):

    c=10
    k=5
    st = 100
    z0 = torch.rand((minibs,3,img_size,img_size))    
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)  
   
    succ = False
    adv = cwatt(z0,labels)
    del cwatt
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)

    succ = len(torch.where(pre==labels)[0])

    return adv,succ





def plotval(epoch,vallabel,outputname,*args):
    assert len(vallabel)==len(args)
    i = 0
    for data in args:
        plt.clf()
        plt.plot(data)
        plt.xlabel(epoch)
        plt.ylabel(vallabel[i])
        plt.savefig('{}/{}.jpg'.format(outputname,vallabel[i]))  
        i += 1

def plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch):
    plt.clf()
    plt.plot(train_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/lossCurve.png'.format(outputname))  

    plt.clf()
    plt.plot(train_loss_secret_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/secrectlossCurve.png'.format(outputname))  

    plt.clf()
    plt.plot(attloss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/attlossCurve.png'.format(outputname))  

    plt.clf()
    plt.plot(loss_cover_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/coverlossCurve.png'.format(outputname)) 


def save_checkpoint(state, filename):
    checkpointname = filename+'_checkpoint.pth.tar'
    torch.save(state, checkpointname)

def train_model(train_loader, beta, learning_rate):
    

    # Iterate over batches performing forward and backward passes
    train_loss_secret_history = []
    attloss_history = []
    loss_cover_history = []  
 
    outputname = 'svhn_train'
    
    print(outputname)
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    best_val_psnr_secret = -1
    best_val_psnr_sc = 10000
    vallabel =['MSE','PSNR','SSIM','APE','LIPIPS','PSNR_c','SSIM_c','LIPIPS_c','MSE_c']
    l2loss_secret_history = []
    # Iterate over batches performing forward and backward passes
    psnr_secret_history = []
    ssim_secret_history = []
    mean_pixel_error_history = []  
    lpips_error_history = []
    psnr_sc_history =[]
    ssim_sc_history =[]
    lpips_sc_history=[]
    mse_sc_history =[]
    for epoch in range(num_epochs):

        # Train mode

        train_losses = []
        # Train one epoch
        cover_succ = 0
        for idx, train_batch in enumerate(train_loader):
           
            net.train()
            train_secrets, labels  = train_batch
            train_secrets = train_secrets.to(device)
            
            labels = labels.to(device)
            train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets))
            cover_succ+= int(succ)
            print("epo:{} att z succ rate:{}".format(epoch,cover_succ/((idx+1)*batch_size)))

            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

   
            # Forward + Backward + Optimize
          
            optimizer.zero_grad()
            mix_img,recover_secret = net(train_secrets,train_covers) # to be [-1,1]
            
            train_loss, train_loss_secret,attloss,loss_cover = attack1_loss(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
            #show

            if (idx+1)%print_freq == 0 or idx==1:
                toshow = torch.cat((train_secrets[:4]*_std_torch+_mean_torch,train_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
                imgg = make_grid(toshow,nrow=nrow_)
                save_image(imgg,'{}/{}_{}.png'.format(outputname,epoch,idx),normalize=False)
   
                plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch)
                      

            train_loss.backward()
            optimizer.step()
                    
            # Saves training loss
            train_losses.append(float(train_loss.data))
            train_loss_secret_history.append(float(train_loss_secret.data))
            attloss_history.append(float(attloss.data))
            loss_cover_history.append(float(loss_cover.data))
            # Prints mini-batch losses
            print('Training: Batch {0}/{1}. Loss of {2:.4f},secret loss of {3:.4f}, attack1_loss of {4:.4f}, loss_cover {5:.5f}'.format(idx+1, len(train_loader), train_loss.data,  train_loss_secret.data,attloss.data,loss_cover))
    
     
            

        if (epoch==0 or (epoch+1)%20==0):
            modelsavepath ='output/svhn/Epoch_{}_{}_latest'.format(epoch,outputname)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()},modelsavepath)
                
        mean_train_loss = np.mean(train_losses)
    
        # Prints epoch average loss
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
        plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch) 
    
    
    return net,mean_train_loss


def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img


def test():
   
    
    MODELS_PATH = '../Weights/Svhn.pth.tar'
    net.eval()
    load_checkpoint(MODELS_PATH)
    
    total = 0
    acctotal = 0
    l2loss_secrets = 0 
    l2loss_covers = 0 
    psnr_secrets =0 
    ssim_secrets =0 
    cover_succ=0
    mean_pixel_errors =0 
    total_all = 0
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
   
    outputname="svhn_test"     
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    
    for idx, train_batch in enumerate(test_loader):

        train_secrets, labels  = train_batch
        train_secrets = train_secrets.to(device)
        total_all += train_secrets.shape[0]
        if total_all<=1000:
            continue
        if total_all>1500:
            break
        total  += train_secrets.shape[0]

        labels = labels.to(device)

        
        train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets))
       
        # Creates variable from secret and cover images
        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)


     
        mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]

        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
            valmetric(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
        acctotal += int(acc_num)
        l2loss_secrets += float(l2loss_secret)
        l2loss_covers += float(l2loss_cover)
        psnr_secrets += float(psnr_secret)
        ssim_secrets += float(ssim_secret)
        mean_pixel_errors += float(mean_pixel_error)
        lpips_errors += float(lpips_error)
        lpips_scs += float(lpips_sc)
        mse_scs += float(mse_sc)
        psnr_scs += float(psnr_sc)
        ssim_scs += float(ssim_sc)

        print("attack success rate:{}/{}={:.6f}".format(acctotal,total,acctotal/total))
        print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets/total))
        print("avg. l2loss_covers:{:.6f}".format(l2loss_covers/total))
        print("avg. psnr_secrets:{:.6f}".format(psnr_secrets/total))
        print("avg. ssim_secrets:{:.6f}".format(ssim_secrets/total))
        print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors/total))
        print("avg. lpips_errors:{:.6f}".format(lpips_errors/total))
        print("avg. lpips_scs:{:.6f}".format(lpips_scs/total))
        print("avg. mse_scs:{:.6f}".format(mse_scs/total))
        print("avg. psnr_scs:{:.6f}".format(psnr_scs/total))
        print("avg. ssim_scs:{:.6f}".format(ssim_scs/total))
        
        diff = mix_img-train_covers
        diff = (diff-torch.min(diff))/(torch.max(diff)-torch.min(diff))
        toshow = torch.cat((train_secrets[:4]*_std_torch+_mean_torch,train_covers[:4],mix_img[:4],recover_secret[:4],diff[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)

    print("finished.")



if __name__ == "__main__":

    stage = "test" # or train

    cwd = os.getcwd()
    # Hyper Parameters
    num_epochs = 200
    #batch_size = 512
    batch_size =64
    batch_size_test = 10
    print_freq =  50
    val_freq = 50
    nrow_ = 4
    learning_rate = 0.0001
    beta = 10
    print("beta : {}".format(beta))
    imgsize=32

    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_model, ds_fetcher, is_imagenet = selector.select('svhn') #labels:0:数字1,...,9:数字10
    target_model.eval().to(device)
    kappa = 5

    modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    residual_en = True
    residual_de = True
    lambda_net = 0.8
    print("encoder residual learning:{}; decoder:{}, lambda_net:{}".format(residual_en,residual_de,lambda_net))

    net = Net(residual_en,residual_de,lambda_net)
    net = torch.nn.DataParallel(net).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
 

    train_loader = ds_fetcher(batch_size=batch_size,train=True,val=False)
    # # Creates test set
    test_loader = ds_fetcher(batch_size=batch_size_test,train=False,val=True) # 因为train的时候batch_size=32，所以用的前1024个做的val，用后面的做test就行
  
    _mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
    _std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)

    if stage == "test":
        test()
    elif stage == "train":
        net, mean_train_loss = train_model(train_loader, beta, learning_rate)

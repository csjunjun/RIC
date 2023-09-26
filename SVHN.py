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
# Directory path
# os.chdir("..")
# cwd = 'input'
cwd = os.getcwd()
# Hyper Parameters
num_epochs = 200-139
#batch_size = 512
batch_size =64
batch_size_test = 10
print_freq =  50
val_freq = 50
nrow_ = 4
learning_rate = 0.0001
beta = 10
#beta = 5
print("beta : {}".format(beta))
imgsize=32
# Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf


# TODO: Define train, validation and models
MODELS_PATH = '/data/junliu/DeepSteg/output/'
# TRAIN_PATH = cwd+'/train/'
# VALID_PATH = cwd+'/valid/'
# VALID_PATH = cwd+'/sample/valid/'
# TRAIN_PATH = cwd+'/sample/train/'
# TEST_PATH = cwd+'/test/'

TRAIN_PATH = '/data2/junliu/data/ImageNet_train_jinyu'
TEST_PATH ='/data2/junliu/data/ImageNet_val'
if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target_model, ds_fetcher, is_imagenet = selector.select('svhn') #labels:0:数字1,...,9:数字10
target_model.eval().to(device)
kappa = 5
def customized_loss(S_prime, C_prime, S, C, B):
    ''' Calculates loss specified on the paper.'''
    
    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    loss_all = loss_cover + B * loss_secret
    return loss_all, loss_cover, loss_secret

def getCEloss(labels,outputs):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1) # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool()) # get the largest logit

    return torch.clamp((i-j), min=-kappa)
    
from skimage.color import rgb2yuv
def rgbtoYUV(img_tensors):
    trans = transforms.ToPILImage()
    img_pil = trans(img_tensors)
    img_yuv = []
    for i in range(len(img_tensors)):
        img_yuv.appen(rgb2yuv(img_pil[i]))

def attack1_lossNorm(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    

    loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
    loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
    outputs = target_model((mix_img-_mean_torch)/_std_torch)
    classloss = torch.mean(getCEloss(labels,outputs))
    
    loss_all =   B*loss_secret  + loss_cover + classloss
    #loss_all = -loss_cover + B * loss_secret + classloss + loss_cover2

    
    return loss_all, loss_secret,classloss,loss_cover


def attack1_loss(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    

    loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
    loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
    outputs = target_model(mix_img)
    classloss = torch.mean(getCEloss(labels,outputs))
    
    #loss_all =   B*loss_secret  + loss_cover + classloss # 20220625 最后采用的
    #ablation
    loss_all =   B*loss_secret  + classloss 
    #loss_all =   B*loss_secret  + classloss + loss_cover 
    #loss_all = -loss_cover + B * loss_secret + classloss + loss_cover2

    
    return loss_all, loss_secret,classloss,loss_cover

from skimage.metrics import peak_signal_noise_ratio, structural_similarity,mean_squared_error


def valmetricNorm(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    outputs = target_model((mix_img-_mean_torch)/_std_torch)
    classloss = torch.mean(getCEloss(labels,outputs))
    pre = torch.argmax(outputs,dim=1)
    acc_num = len(torch.where(pre==labels)[0])    
    psnr, ssim, mse_s,mse_c,mean_pixel_error = [], [], [],[],[]
    norm_S =  convert1(S*_std_torch+_mean_torch)
    norm_S_prime = convert1(S_prime)
    norm_miximg = convert1(mix_img)
    norm_C = convert1(C)
    for i in range(len(S_prime)):
        # mse_s.append(mean_squared_error(norm_S_prime[i], norm_S[i]))
        # mse_c.append(mean_squared_error(norm_miximg[i], norm_C[i]))
        mse_s.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, multichannel=True))
        mean_pixel_error.append(float(torch.sum(torch.abs(torch.round((S*_std_torch+_mean_torch)[i]*255)-torch.round(S_prime[i]*255)))/(3*imgsize*imgsize)))
    #ssim_secret =0 
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error)

import sys
sys.path.append('/home/junliu/code/Universal-Deep-Hiding')
import PerceptualSimilarity.models
modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

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
        # mse_s.append(mean_squared_error(norm_S_prime[i], norm_S[i]))
        # mse_c.append(mean_squared_error(norm_miximg[i], norm_C[i]))
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
    #ssim_secret =0 
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error),np.sum(lpips_error),np.sum(lpips_sc),np.sum(mse_sc),np.sum(psnr_sc),np.sum(ssim_sc)


def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''
    
    img = denormalize(img, std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example '+str(idx)+', lr='+str(learning_rate)+', B='+str(beta))
    plt.show()
    return

def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''
    
    noise = torch.nn.init.normal(torch.Tensor(tensor.size()), 0, 0.1).to(device)
    return Variable(tensor + noise)
# class PrepNetwork(nn.Module):
#     def __init__(self):
#         super(PrepNetwork, self).__init__()
#         self.initialP3 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.initialP4 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.initialP5 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU())
#         self.finalP3 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.finalP4 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.finalP5 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=5, padding=2),
#             nn.ReLU())

#         self.se1 = SEModule(50,2)
#         self.se2 = SEModule(50,2)
#         self.se3 = SEModule(50,2)

#     def forward(self, p):
#         p1 = self.initialP3(p)
#         p2 = self.initialP4(p)
#         p3 = self.initialP5(p)
#         mid = torch.cat((p1, p2, p3), 1)
#         p4 = self.finalP3(mid)
#         p5 = self.finalP4(mid)
#         p6 = self.finalP5(mid)

#         p4 = self.se1(p4)
#         p5 = self.se2(p5)
#         p6 = self.se3(p6)
#         out = torch.cat((p4, p5, p6), 1)
#         return out
# #在训练的时候，由于希望分类器分类正确，最后生成的mix image会逐渐学到secret image，所以这里可以先用一个inn将原始的secret image
# #转换到特征空间，然后在decode的时候逆回去
# # Hiding Network (5 conv layers)
# class HidingNetwork(nn.Module):
#     def __init__(self):
#         super(HidingNetwork, self).__init__()
#         self.initialH3 = nn.Sequential(
#             nn.Conv2d(153, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.initialH4 = nn.Sequential(
#             nn.Conv2d(153, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.initialH5 = nn.Sequential(
#             nn.Conv2d(153, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU())
#         self.finalH3 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.finalH4 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.finalH5 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=5, padding=2),
#             nn.ReLU())
#         self.finalH = nn.Sequential(
#             nn.Conv2d(150, 3, kernel_size=1, padding=0))
        
#     def forward(self, h,cover):
#         h1 = self.initialH3(h)
#         h2 = self.initialH4(h)
#         h3 = self.initialH5(h)
#         mid = torch.cat((h1, h2, h3), 1)
#         h4 = self.finalH3(mid)
#         h5 = self.finalH4(mid)
#         h6 = self.finalH5(mid)
#         mid2 = torch.cat((h4, h5, h6), 1)
#         out = self.finalH(mid2)
        
#         out = 0.2*out + 0.8*cover
#         return out

# # Reveal Network (2 conv layers)
# class RevealNetwork(nn.Module):
#     def __init__(self):
#         super(RevealNetwork, self).__init__()
#         self.initialR3 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.initialR4 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.initialR5 = nn.Sequential(
#             nn.Conv2d(3, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=5, padding=2),
#             nn.ReLU())
#         self.finalR3 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=3, padding=1),
#             nn.ReLU())
#         self.finalR4 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=4, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(50, 50, kernel_size=4, padding=2),
#             nn.ReLU())
#         self.finalR5 = nn.Sequential(
#             nn.Conv2d(150, 50, kernel_size=5, padding=2),
#             nn.ReLU())
#         self.finalR = nn.Sequential(
#             nn.Conv2d(150, 3, kernel_size=1, padding=0))

#     def forward(self, r,cover):
#         r = (r - 0.8*cover)/0.2
#         r1 = self.initialR3(r)
#         r2 = self.initialR4(r)
#         r3 = self.initialR5(r)
#         mid = torch.cat((r1, r2, r3), 1)
#         r4 = self.finalR3(mid)
#         r5 = self.finalR4(mid)
#         r6 = self.finalR5(mid)
#         mid2 = torch.cat((r4, r5, r6), 1)
#         out = self.finalR(mid2)
#         return out



# # Join three networks in one module


# #upsampler = Upsample(size=(imgsize, imgsize), align_corners=True, mode='bilinear')

# class SEModule(nn.Module):
#     def __init__(self, channels, reduction, concat=False):
#         super(SEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         module_input = x
#         x = self.avg_pool(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return module_input * x


# # Join three networks in one module
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.m1 = PrepNetwork()
#         self.m2 = HidingNetwork()
#         self.m3 = RevealNetwork()
#         self.act = nn.Sigmoid()
#     # def forward(self, secret, cover,labels,train=True):
#     #     x_1 = self.m1(secret)
#     #     mid = torch.cat((x_1, cover), 1)
#     #     x_2 = self.m2(mid,cover)
        
#     #     x_2 = self.quan(x_2,type='noise',train=train)
#     #     #x_2 = torch.clamp(x_2,0,1)
#     #     random_cover,succ = getAdvZ(224,labels,batch_size)

#     #     x_3 = self.m3(x_2,random_cover) #训练的时候不要用clamp 或者 sigmoid
#     #     #x_3 = torch.clamp(x_3,0,1)
#     #     #x_3 = self.act(x_3)
#     #     return x_2, x_3

#     def forward(self, secret, cover,train=True):
#         x_1 = self.m1(secret)
#         mid = torch.cat((x_1, cover), 1)
#         x_2 = self.m2(mid,cover)
        
#         x_2 = self.quan(x_2,type='noise',train=train)
#         #x_2 = self.getnoistadv(x_2)
#         #cover = self.getnoistadv(cover)
#         x_3 = self.m3(x_2,cover) #训练的时候不要用clamp 或者 sigmoid
#         if train==False:
#             x_3 = torch.clamp(x_3,0,1)
#         #x_3 = self.act(x_3)
#         return x_2, x_3

#     def getnoistadv(self,x):
#         img_size = imgsize
#         #z0_noise = 0.1*torch.randn((batch_size,3,img_size,img_size)).to(device)
#         z0_noise = 0.01*torch.rand((batch_size,3,img_size,img_size)).to(device)
#         x = x+z0_noise
#         x = torch.clamp(x,0,1)
#         return x

#     def quan(self,x,type='noise',train=True):
#         #x = torch.round(torch.clamp(x,0,1)*255.)/255.
#         if type=='round':
#             x = torch.round(torch.clamp(x*255.,0,255.))/255.
#         elif type == 'noise':
#             x = x*255.
#             if train:
#                 noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5).cuda()
#                 output = x + noise
#                 output = torch.clamp(output, 0, 255.) 
#             else:
#                 output = x.round() * 1.0
#                 output = torch.clamp(output, 0, 255.)                 
#         else:
#             raise error("quan is not implemented.")
#         return output/255.
# # Creates net object

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        #self.m3 = RevealNetwork()

    def forward(self, secret, cover,train=True):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2 = self.m2(mid,cover)
        #x_3 = self.m3(x_2,cover)
        x_2 = self.quan(x_2,type='noise',train=train)
        return x_2
    def quan(self,x,type='noise',train=True):
        #x = torch.round(torch.clamp(x,0,1)*255.)/255.
        if type=='round':
            x = torch.round(torch.clamp(x*255.,0,255.))/255.
        elif type == 'noise':
            x = x*255.
            if train:
                noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5).cuda()
                output = x + noise
                output = torch.clamp(output, 0, 255.) 
            else:
                output = x.round() * 1.0
                output = torch.clamp(output, 0, 255.)                 
        else:
            raise error("quan is not implemented.")
        return x/255.
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.m3 = RevealNetwork()
        self.act = nn.Sigmoid()

    def forward(self, container, cover):

        x_3 = self.m3(container,cover)
        #x_3 = self.act(x_3)
        #x_3 = torch.clamp(x_3,0,1)
        return x_3    



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


#mode = 'split'
mode = 'combine'
residual_en = True
residual_de = True
lambda_net = 0.8
print("encoder residual learning:{}; decoder:{}, lambda_net:{}".format(residual_en,residual_de,lambda_net))
if mode== 'combine':
    net = Net(residual_en,residual_de,lambda_net)
    net = torch.nn.DataParallel(net).cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    #scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5,mode='min') 
    modelname = 'Epoch_139_svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@2_correctbest_latest_checkpoint.pth.tar' 
    load_checkpoint(MODELS_PATH+modelname)
else:
    encoder = Encoder().to(device)
    decoder= Decoder().to(device)
# myconfig={'H':224,'W':224,
# 'encoder_channels':64,
# 'encoder_blocks':4,
# 'decoder_channels':64,
# 'decoder_blocks':7,
# 'message_length':100,
# }
# from hidden.options import HiDDenConfiguration
# hidden_config = HiDDenConfiguration(H=224, W=224,
#                                     message_length=30,
#                                     encoder_blocks=4, encoder_channels=64,
#                                     decoder_blocks=7, decoder_channels=64,
#                                     use_discriminator=False,
#                                     use_vgg=False,
#                                     discriminator_blocks=3, discriminator_channels=64,
#                                     decoder_loss=1,
#                                     encoder_loss=0.7,
#                                     adversarial_loss=1e-3,
#                                     enable_fp16=False
#                                     )
# from hidden.noise_layers import noiser
# mynoiser = noiser.Noiser(noise_layers=[],device=device)
# net = encoder_decoder.EncoderDecoder(config=hidden_config,noiser=mynoiser
# )
# Creates training set
# train_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(
#         TRAIN_PATH,
#         transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(imgsize),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean,std=std)
#         ])), batch_size=batch_size, num_workers=4, 
#         pin_memory=True, shuffle=True, drop_last=True)
# ds_fetcher 内部已经被-0.5/0.5 至[-1,1]区间
train_loader = ds_fetcher(batch_size=batch_size,train=True,val=False)
# # Creates test set
test_loader = ds_fetcher(batch_size=batch_size_test,train=False,val=True) # 因为train的时候batch_size=32，所以用的前1024个做的val，用后面的做test就行
# test_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(
#         TEST_PATH, 
#         transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(imgsize),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean,std=std)
#         ])), batch_size=batch_size, num_workers=1, 
#         pin_memory=True, shuffle=False, drop_last=True)
_mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
_std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)


def getAdvZ(img_size,labels,batch_size,minibs,typen='uniform',mode='equal',zo=False):
    # for train
    # c=10
    # k=5
    # st=60
    c=10
    k=5
    if typen =='uniform':
        st = 100
        z0 = torch.rand((minibs,3,img_size,img_size))    
        cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels,noise_type='uniform',mode=mode)  
    else:
        st = 60
        z0 = torch.randn((minibs,3,img_size,img_size))
        z0 = (z0-torch.min(z0))/(torch.max(z0)-torch.min(z0))
        cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels,mode=mode)

    #st=100 #for uniform noise
    succ = False
    #init random noise
    # z0 = nn.init.trunc_normal_(torch.zeros((1,3,img_size,img_size)),0,1,-2,2)
    # z0 = 0.25*(2+z0).to(device) 
    
    #
    
    
    adv = cwatt(z0,labels)
    del cwatt
    #outputs = target_model((adv-_mean_torch)/_std_torch)
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)
    # if torch.all(pre==labels):
    #     succ = True
    succ = len(torch.where(pre==labels)[0])
    if zo==True:
        return adv,succ,z0
    return adv,succ


def getAdvZNorm(img_size,labels,batch_size,minibs):
    # for train
    # c=10
    # k=5
    # st=60
    c=10
    k=5
    st=60
    succ = False
    #init random noise
    # z0 = nn.init.trunc_normal_(torch.zeros((1,3,img_size,img_size)),0,1,-2,2)
    # z0 = 0.25*(2+z0).to(device) 
    z0 = torch.randn((minibs,3,img_size,img_size))
    z0 = (z0-torch.min(z0))/(torch.max(z0)-torch.min(z0))
    cwatt = juncwNorm.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
    adv = cwatt(z0,labels)
    del cwatt
    outputs = target_model((adv-_mean_torch)/_std_torch)
    
    _,pre = torch.max(outputs,1)
    # if torch.all(pre==labels):
    #     succ = True
    succ = len(torch.where(pre==labels)[0])
    return adv,succ

def getNoisyAdvZ(img_size,labels,batch_size):
    c=10
    k=10
    st=30
    succ = False
    #init random noise
    # z0 = nn.init.trunc_normal_(torch.zeros((1,3,img_size,img_size)),0,1,-2,2)
    # z0 = 0.25*(2+z0).to(device) 
    z0 = torch.randn((batch_size,3,img_size,img_size))
    z0 = (z0-torch.min(z0))/(torch.max(z0)-torch.min(z0))
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
    adv = cwatt(z0,labels)
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)
    # if torch.all(pre==labels):
    #     succ = True
    z0_noise = torch.randn((batch_size,3,img_size,img_size)).to(device)
    #z0_noise = torch.rand((batch_size,3,img_size,img_size))
    adv = adv+z0_noise
    adv = torch.clamp(adv,0,1)
    succ = len(torch.where(pre==labels)[0])
    return adv,succ



def val(outputname,epoch):
    # net.load_state_dict(torch.load(MODELS_PATH+'Epoch N4.pkl'))

    # Switch to evaluate mode
    net.eval()

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
    # Show images
    valcnt = 0
    outputname = outputname+'_val'
    # if not os.path.exists(outputname):
    #     os.mkdir(outputname)


    for idx, test_batch in enumerate(test_loader):
        
        test_secrets, labels  = test_batch
        test_secrets = test_secrets.to(device)       
        if valcnt >= 500:
            break

        labels = labels.to(device)
        test_covers,succ = getAdvZ(imgsize,labels,len(test_secrets),len(test_secrets),typen='uniform',mode='equal')

        
        # Creates variable from secret and cover images
        test_secrets = Variable(test_secrets, requires_grad=False)
        test_covers = Variable(test_covers, requires_grad=False)

        mix_img,recover_secret = net(test_secrets,test_covers,train=False) # to be [-1,1]
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
            valmetric(recover_secret,mix_img, test_secrets,test_covers,beta,labels)

        l2loss_secret_history.append(l2loss_secret)
        psnr_secret_history.append(psnr_secret)
        ssim_secret_history.append(ssim_secret)
        mean_pixel_error_history.append(mean_pixel_error)
        lpips_error_history.append(lpips_error)
        psnr_sc_history.append(psnr_sc)
        ssim_sc_history.append(ssim_sc)
        lpips_sc_history.append(lpips_sc)
        mse_sc_history.append(mse_sc)
        valcnt += len(test_secrets)      

    mean_mse = np.sum(l2loss_secret_history)/valcnt  
    mean_psnr = np.sum(psnr_secret_history)/valcnt
    mean_ssim = np.sum(ssim_secret_history)/valcnt
    mean_ape = np.sum(mean_pixel_error_history)/valcnt
    mean_lpips = np.sum(lpips_error_history)/valcnt
    mean_psnr_sc = np.sum(psnr_sc_history)/valcnt
    mean_ssim_sc = np.sum(ssim_sc_history)/valcnt
    mean_mse_sc = np.sum(mse_sc_history)/valcnt
    mean_lpips_sc = np.sum(lpips_sc_history)/valcnt

    print (' mean_mse on test set: {:.2f}'.format(mean_mse))
    print (' mean_psnr on test set: {:.2f}'.format(mean_psnr))
    print (' mean_ssim on test set: {:.2f}'.format(mean_ssim))
    print (' mean_ape on test set: {:.2f}'.format(mean_ape))
    print (' mean_lpips on test set: {:.2f}'.format(mean_lpips))
    print (' mean_psnr_sc on test set: {:.2f}'.format(mean_psnr_sc))
    print (' mean_ssim_sc on test set: {:.2f}'.format(mean_ssim_sc))
    print (' mean_mse_sc on test set: {:.2f}'.format(mean_mse_sc))
    print (' mean_lpips_sc on test set: {:.2f}'.format(mean_lpips_sc))

    return mean_mse,mean_psnr,mean_ssim,mean_ape,mean_lpips,mean_psnr_sc,mean_ssim_sc,mean_mse_sc,mean_lpips_sc

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
    #outputname = 'datanormadvnonorm_rightmetric'
    #outputname = 'datanormadvnonorm_rightmetric_splitloss'
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_again' #pid 16882
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_sigmoid' #pid 53069
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_sigmoid_quan' #pid 34997  kill 有色差
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_clamp_quan' #pid 42931 kill 一直都是黄色，不对
    #outputname = 'datanormadvnonorm_rightmetric_clamp_quan' #pid 56913 run
    #outputname = 'datanormadvnonorm_rightmetric_quan' #pid 10509 run
    #outputname = 'datanormadvnonorm_rightmetric_sigmoid_quan'
    #outputname = 'datanormadvnonorm_rightmetric_clamp_clamp_noquan'
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_quan' #26496 run
    #outputname = 'svhnagain'
    #outputname = 'svhn_advnorm_v2'
    #outputname = 'svhn_noval'
    #outputname = 'svhn_noval_uniform'# cw 中的loss项：攻击loss,距离最大化
    #outputname = 'svhn_noval_uniform_rightcw_nonormz' # cw 中的loss项只有攻击loss,不包含距离
    #outputname = 'svhn_noval_uniform_cwmindis_nonormz'# cw 中的loss项：攻击loss,距离最小化
    #outputname = 'svhn_noval_gaussian_nodisloss'# cw 中的loss项：攻击loss,距离最小化
    #outputname = 'svhn_noval_gaussian_cwmindis'# 
    #outputname = 'svhn_noval_uniform_nodisloss_ablation_rec_adv'# 
    #outputname = 'svhn_noval_uniformnoiscover'# 
    #outputname = 'svhn_noval_uniform_nodisloss_ablation_rec_pri'
    #outputname = 'svhn_noval_uniform_nodisloss_ablation_nores_lossrec_adv'
    #outputname = 'svhn_noval_uniformnoiscover_lossrec_adv'
    #outputname = 'svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_a0@1'
    #outputname = 'svhn_noval_uniformnoiscover_lossrec_adv_nonormz_addquannoise'
    #outputname = 'svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_endenores'
    #outputname = 'svhn_noval_uniform_lossrec_adv_pri_nonormz_addquannoise'
    #outputname = 'svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_beta5'
    outputname = 'svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@2_correctbest_resumefrom139'
    #outputname = 'svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@8'
    #outputname = 'svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@1_correctbest'
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
            if mode== 'combine':
                net.train()
            else:
                encoder.train()
                decoder.train()
            train_secrets, labels  = train_batch
            train_secrets = train_secrets.to(device)
            

            labels = labels.to(device)
            train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets),typen='uniform')
            #train_covers = torch.rand((len(train_secrets),3,imgsize,imgsize)).cuda()
            cover_succ+= int(succ)
            print("epo:{} att z succ rate:{}".format(epoch,cover_succ/((idx+1)*batch_size)))
            #train_covers = (train_covers-torch.min(train_covers))/(torch.max(train_covers)-torch.min(train_covers))
            #train_covers = (train_covers-_mean_torch)/_std_torch

            # train_covers = (train_covers-mean_torch)/std_torch
            # Saves secret images and secret covers
            #train_covers = data[:len(data)//2]
            #train_secrets = data[len(data)//2:]
            
            # Creates variable from secret and cover images
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

   
            # Forward + Backward + Optimize
            if mode== 'combine':
                optimizer.zero_grad()
                mix_img,recover_secret = net(train_secrets,train_covers) # to be [-1,1]
            else:
                optimizerEn.zero_grad()
                optimizerDe.zero_grad()
            
            
                mix_img = encoder(train_secrets,train_covers)
                
                #mix_img = torch.round(torch.clamp(mix_img,0,1)*255.)/255.
                recover_secret = decoder(mix_img,train_covers)

            train_loss, train_loss_secret,attloss,loss_cover = attack1_loss(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
            #show
            # train_covers_ = train_covers*_std_torch+_mean_torch
            # train_secrets_ = train_secrets*_std_torch+_mean_torch
            # train_output_ = train_output*_std_torch+_mean_torch
            # train_hidden_ = train_hidden*_std_torch+_mean_torch
            if (idx+1)%print_freq == 0 or idx==1:
                #toshow = torch.cat((train_secrets[:4],train_covers[:4],(train_output[:4]+1)*0.5,(train_hidden[:4]+1)*0.5),dim=0)
                toshow = torch.cat((train_secrets[:4]*_std_torch+_mean_torch,train_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
                imgg = make_grid(toshow,nrow=nrow_)
                save_image(imgg,'{}/{}_{}.png'.format(outputname,epoch,idx),normalize=False)

                    
               
                plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch)
                  #     
            # Calculate loss and perform backprop  attack1_loss(S_prime, C_prime, S,C,  B,labels):
            

            if mode== 'combine':
                train_loss.backward()
                optimizer.step()
            else: # 最终效果是分开train encoder 和decoder 比较好，combine 模式下有的图像仍有色差，如datanormadvnonorm_rightmetric/10259.png
                train_loss_secret.backward(retain_graph=True)
                train_loss.backward()
                optimizerDe.step()
                decoder.eval()
                optimizerEn.step()            
            # Saves training loss
            train_losses.append(float(train_loss.data))
            

            train_loss_secret_history.append(float(train_loss_secret.data))
            attloss_history.append(float(attloss.data))
            loss_cover_history.append(float(loss_cover.data))
            # Prints mini-batch losses
            print('Training: Batch {0}/{1}. Loss of {2:.4f},secret loss of {3:.4f}, attack1_loss of {4:.4f}, loss_cover {5:.5f}'.format(idx+1, len(train_loader), train_loss.data,  train_loss_secret.data,attloss.data,loss_cover))
    
        #val per epoch 

        mean_mse,mean_psnr,mean_ssim,mean_ape,mean_lpips,mean_psnr_sc,mean_ssim_sc,mean_mse_sc,mean_lpips_sc = val(outputname,epoch)
        l2loss_secret_history.append(mean_mse)
        psnr_secret_history.append(mean_psnr)
        ssim_secret_history.append(mean_ssim)
        mean_pixel_error_history.append(mean_ape)
        lpips_error_history.append(mean_lpips)
        psnr_sc_history.append(mean_psnr_sc)
        ssim_sc_history.append(mean_ssim_sc)
        lpips_sc_history.append(mean_lpips_sc)
        mse_sc_history.append(mean_mse_sc)
        plotval(epoch,vallabel,outputname,l2loss_secret_history,psnr_secret_history,ssim_secret_history,mean_pixel_error_history,\
    lpips_error_history,psnr_sc_history,ssim_sc_history,lpips_sc_history,mse_sc_history) 
            #scheduler.step(float(val_loss_secret))
            
        if mean_psnr> best_val_psnr_secret:
            modelsavepath =MODELS_PATH+'Epoch_{}_psnrbest'.format(outputname)
            
            save_checkpoint({
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    #'scheduler' : scheduler.state_dict()
                },filename=modelsavepath)
            print("update best_val_psnr_secret from {} to {}".format(float(best_val_psnr_secret),float(mean_psnr)))
            best_val_psnr_secret= mean_psnr

        if mean_psnr_sc< best_val_psnr_sc:
            modelsavepath =MODELS_PATH+'Epoch_{}_psnrscbest'.format(outputname)
            
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                #'scheduler' : scheduler.state_dict()
            },filename=modelsavepath)
            print("update best_val_psnr_sc from {} to {}".format(float(best_val_psnr_sc),float(mean_psnr_sc)))
            best_val_psnr_sc = mean_psnr_sc
        if mode== 'combine' and (epoch==0 or (epoch+1)%20==0):
            modelsavepath =MODELS_PATH+'Epoch_{}_{}_latest'.format(epoch,outputname)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()},modelsavepath)
                
        mean_train_loss = np.mean(train_losses)
    
        # Prints epoch average loss
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
        plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch) 
    if mode== 'combine':
    
        return net,mean_train_loss
    else:
        return encoder,decoder, mean_train_loss


# def load_checkpoint(filepath):
#     if os.path.isfile(filepath):
#         print("=> loading checkpoint '{}'".format(filepath))
#         checkpoint = torch.load(filepath)

#         start_epoch = checkpoint['epoch']
        
#         net.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         print("=> loaded checkpoint (epoch {})"
#                 .format(checkpoint['epoch']))
#     else:
#         print("=> no checkpoint found at '{}'".format(filepath))

def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

def threatModel():
    modelname='Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
    net.eval()
    load_checkpoint(MODELS_PATH+modelname)
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

    outputname=modelname.split(".")[0]+"_forthreat"
    miximgs = None
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    ii=0
    #target_labels = 0
    foldername = 'data_samenae_class0'
    basepath='/home/junliu/code/DeepSteg/Pytorch-UNet/data/SVHN'
    #foldername = 'data' # 这是每张图一个nae的情况
    # 1000-1500 for  (450/50) training /validation UNET, 1500-1600 for testing
    for idx, train_batch in enumerate(test_loader):

        train_secrets, labels  = train_batch
        train_secrets = train_secrets.to(device)
        labels = labels.to(device)
        total_all += train_secrets.shape[0]
        if total_all<=1500:
            continue
        # if total_all>5000:
        #     break
        # if int(labels)!=target_labels:
        #     continue
        
        total  += train_secrets.shape[0] 
        if total>100:
            break       
        if total == train_secrets.shape[0]:
            train_covers,succ = getAdvZ(imgsize,labels,len(train_secrets),len(train_secrets),typen='uniform',mode='equal')
            train_covers = Variable(train_covers, requires_grad=False)
        train_secrets = Variable(train_secrets, requires_grad=False)
        

        mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]
        for i in range(len(mix_img)):
            save_image(mix_img[i],'{}/test_miximg/{}_{}.png'.format(basepath,ii,int(labels[i])))
            save_image(train_covers[i],'{}/test_nae/{}_{}.png'.format(basepath,ii,int(labels[i])))
            save_image(recover_secret[i],'{}/test_recover/{}_{}.png'.format(basepath,ii,int(labels[i])))
            ii+=1
        # if miximgs is None:
        #     miximgs = mix_img
        #     naes = train_covers
        #     recover = recover_secret
        # else:
        #     miximgs = torch.cat((miximgs,mix_img),dim=0)
        #     naes = torch.cat((naes,train_covers),dim=0)
        #     recover = torch.cat((recover,recover_secret),dim=0)
        # miximgs.append(mix_img.detach().cpu().numpy())
        # naes.append(train_covers.detach().cpu().numpy())
        # recover.append(recover_secret.detach().cpu().numpy())
    # np.save('/data2/junliu/DeepSteg/data/svhn500testmiximg.npy',miximgs.detach().cpu().numpy())
    # np.save('/data2/junliu/DeepSteg/data/svhn500testnaes.npy',naes.detach().cpu().numpy())
    # np.save('/data2/junliu/DeepSteg/data/svhn500testrecover.npy',recover.detach().cpu().numpy())
 
    print("finished.")

def threatSplitDataset():
    miximg = np.load('/data2/junliu/DeepSteg/data/svhn500testmiximg.npy')
    naes = np.load('/data2/junliu/DeepSteg/data/svhn500testnaes.npy')
    rec = np.load('/data2/junliu/DeepSteg/data/svhn500testrecover.npy')
    num = len(miximg)
    idx = np.linspace(0,num,num=num,endpoint=False,dtype=np.uint16)
    np.random.shuffle(idx)
    np.save('/data2/junliu/DeepSteg/data/shuffledidx.npy',idx)
    trainmiximg = miximg[idx][:int(num*0.9)]
    print("finished.")

def test():
    if mode == 'combine':
        #modelname='Epoch_N1_datanormadvnonorm_rightmetric_splitloss.pkl' #量化后再保存颜色会有点状噪声
        #modelname='Epoch_N1_datanormadvnonorm_rightmetric_clamp_quan.pkl'
        modelname='Epoch_N1_datanormadvnonorm_rightmetric_quan_ok.pkl'
        modelname='Epoch_N1_datanormadvnonorm_rightmetric_quan.pkl'
        modelname = 'dgx2imagenet.pkl'
        modelname = 'Epoch_28_svhnagain_model_best.pth.tar'
        modelname= 'Epoch_1_faceclassification_se_resume_resume_rightvalset_model_best.pth.tar'
        modelname = 'Epoch_100_svhn_noval_uniform_checkpoint.pth.tar' 
        modelname = 'Epoch_99_svhn_noval_uniform_rightcw_nonormz_checkpoint.pth.tar'
        #modelname ='Epoch_199_svhn_noval_uniform_rightcw_nonormz_checkpoint.pth.tar'
        #modelname='Epoch_179_svhn_noval_uniform_checkpoint.pth.tar'
        #modelname ='Epoch_199_svhn_noval_uniform_cwmindis_nonormz_checkpoint.pth.tar'
        #modelname = 'Epoch_179_svhn_noval_gaussian_nodisloss_checkpoint.pth.tar'
        #modelname = 'Epoch_79_svhn_noval_gaussian_cwmindis_checkpoint.pth.tar'
        #modelname='Epoch_1_3399_faceclassification_se_rightface_noval_uniformcover_resume2_checkpoint.pth.tar'
        #modelname = 'Epoch_79_svhn_noval_uniform_nodisloss_ablation_rec_adv_checkpoint.pth.tar'
        #modelname='Epoch_99_svhn_noval_uniformnoiscover_checkpoint.pth.tar'
        #MODELS_PATH = '/data2/junliu/DeepSteg/output/'
        modelname='Epoch_199_svhn_noval_uniform_nodisloss_ablation_rec_pri_checkpoint.pth.tar'
        #modelname='Epoch_199_svhn_noval_uniform_nodisloss_ablation_nores_lossrec_adv_checkpoint.pth.tar'
        modelname='Epoch_199_svhn_noval_uniform_nodisloss_ablation_nores_checkpoint.pth.tar'
        modelname='Epoch_199_svhn_noval_uniformnoiscover_lossrec_adv_checkpoint.pth.tar'
        #modelname='Epoch_99_svhn_noval_uniform_rightcw_nonormz_checkpoint.pth.tar'
        modelname='Epoch_159_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        modelname='Epoch_119_svhn_noval_uniformnoiscover_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        #modelname='Epoch_199_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_a0@7_checkpoint.pth.tar'
        #modelname='Epoch_199_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_endenores_checkpoint.pth.tar'
        modelname='Epoch_159_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_a0@5_checkpoint.pth.tar'
        #modelname='Epoch_79_svhn_noval_uniform_lossrec_adv_pri_nonormz_addquannoise_checkpoint.pth.tar'
        modelname='Epoch_svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@6_psnrbest_checkpoint.pth.tar'
        #modelname='Epoch_139_svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@6_latest_checkpoint.pth.tar'
        #modelname='Epoch_1_2599_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume4_checkpoint.pth.tar'
        modelname ='Epoch_1_199_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume10_checkpoint.pth.tar'
        modelname='Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        #modelname='Epoch_139_svhn_valsave_uniform_lossrec_adv_nonormz_addquannoise_a0@8_resumefrom199_latest_checkpoint.pth.tar'
        modelname='Epoch_199_svhn_noval_uniformnoiscover_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        net.eval()
        load_checkpoint(MODELS_PATH+modelname)
    elif mode == 'split':
        enmodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_sigmoid_quan_encoder.pkl'
        demodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_sigmoid_quan_decoder.pkl'
        encoder.eval()
        decoder.eval()
        encoder.load_state_dict(torch.load(MODELS_PATH+enmodelname))
        decoder.load_state_dict(torch.load(MODELS_PATH+demodelname))
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
    if mode == 'combine':
        #outputname=modelname.split(".")[0]+"_svhnbyvggface2model"
        #outputname=modelname.split(".")[0]+"_gaussiancover"
        outputname=modelname.split(".")[0]
        #outputname=modelname.split(".")[0]+"_svhnbyvggface2_uniform"
        #outputname=modelname.split(".")[0]+"_svhnbyablation_lossrec_adv_uniform"
        #outputname=modelname.split(".")[0]+"forpaper"
    else:
        outputname=demodelname.split(".")[0]
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

        # outputs = target_model(train_secrets)
        # pre = torch.argmax(outputs,dim=1)
        # train_secrets = train_secrets[torch.where(pre==labels)]
        # labels = labels[torch.where(pre==labels)]
        # if len(train_secrets) == 0:
        #     continue
        
        train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets),typen='uniform',mode='equal')
        #cover_succ+= int(succ)
        #print("att z succ rate:{}".format(cover_succ/total))
        #train_covers = torch.rand((len(train_secrets),3,imgsize,imgsize)).cuda()
        
        
        
        #train_covers = (train_covers-torch.min(train_covers))/(torch.max(train_covers)-torch.min(train_covers))
        #train_covers = (train_covers-_mean_torch)/_std_torch

        # train_covers = (train_covers-mean_torch)/std_torch
        # Saves secret images and secret covers
        #train_covers = data[:len(data)//2]
        #train_secrets = data[len(data)//2:]
        
        # Creates variable from secret and cover images
        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)


        if mode == 'combine':
            mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]
        else:
            mix_img = encoder(train_secrets,train_covers,train=False)
            recover_secret = decoder(mix_img,train_covers)
        #begin testSafety
        # diff = (mix_img-0.8*train_covers)/0.2
        #toshow = diff*10
        # imgg = make_grid(diff*10,nrow=nrow_)
        # save_image(imgg,'{}/{}_testdiff_coef10x.png'.format(outputname,idx),normalize=True)
        #end testSafety
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


def testAblationAll():
    if mode == 'combine':
        
        modelname='Epoch_199_svhn_noval_uniformnoiscover_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        net.eval()
        load_checkpoint(MODELS_PATH+modelname)
    elif mode == 'split':
        enmodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_sigmoid_quan_encoder.pkl'
        demodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_sigmoid_quan_decoder.pkl'
        encoder.eval()
        decoder.eval()
        encoder.load_state_dict(torch.load(MODELS_PATH+enmodelname))
        decoder.load_state_dict(torch.load(MODELS_PATH+demodelname))
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
    if mode == 'combine':
        #outputname=modelname.split(".")[0]+"_svhnbyvggface2model"
        #outputname=modelname.split(".")[0]+"_gaussiancover"
        outputname=modelname.split(".")[0]
        #outputname=modelname.split(".")[0]+"_svhnbyvggface2_uniform"
        #outputname=modelname.split(".")[0]+"_svhnbyablation_lossrec_adv_uniform"
        #outputname=modelname.split(".")[0]+"forpaper"
    else:
        outputname=demodelname.split(".")[0]
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    
    for idx, train_batch in enumerate(test_loader):

        train_secrets, labels  = train_batch
        train_secrets = train_secrets.to(device)
        total_all += train_secrets.shape[0]
        if total_all<=1000:
            continue
        if total>500:
            break
        

        labels = labels.to(device)

        outputs = target_model(train_secrets)
        pre = torch.argmax(outputs,dim=1)
        train_secrets = train_secrets[torch.where(pre==labels)]
        labels = labels[torch.where(pre==labels)]
        if len(train_secrets) == 0:
            continue
        
        total  += train_secrets.shape[0]

        #cover_succ+= int(succ)
        #print("att z succ rate:{}".format(cover_succ/total))
        train_covers = torch.rand((len(train_secrets),3,imgsize,imgsize)).cuda()
        
        
        
        #train_covers = (train_covers-torch.min(train_covers))/(torch.max(train_covers)-torch.min(train_covers))
        #train_covers = (train_covers-_mean_torch)/_std_torch

        # train_covers = (train_covers-mean_torch)/std_torch
        # Saves secret images and secret covers
        #train_covers = data[:len(data)//2]
        #train_secrets = data[len(data)//2:]
        
        # Creates variable from secret and cover images
        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)


        if mode == 'combine':
            mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]
        else:
            mix_img = encoder(train_secrets,train_covers,train=False)
            recover_secret = decoder(mix_img,train_covers)
        #begin testSafety
        # diff = (mix_img-0.8*train_covers)/0.2
        #toshow = diff*10
        # imgg = make_grid(diff*10,nrow=nrow_)
        # save_image(imgg,'{}/{}_testdiff_coef10x.png'.format(outputname,idx),normalize=True)
        #end testSafety
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
        print("attack success rate in correct prediction images:{}/{}={:.6f}".format(acctotal,total_all-1010,acctotal/(total_all-1010)))
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


def intkey(ele):
    return int(ele[0])
def drawDict(mydict,mydict2,ylabel,filename=None,threattype='JPEG'):
    if filename is None:
        filename = ylabel
    plt.clf()
    plt.figure(figsize=(11,8))
    if threattype=='JPEG':
        mydict['99']=1
        mydict2['99']=1
        lists = sorted(mydict.items(),key=intkey) # sorted by key, return a list of tuples
        lists2 = sorted(mydict2.items(),key=intkey)
    else:
        lists = sorted(mydict.items()) 
        mydict2['0'] =1
        lists2 = sorted(mydict2.items())        
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    x2, y2 = zip(*lists2)
    plt.plot(np.asarray(x, float),np.asarray(y)*96,'bo-',label='SVHN',linewidth=3,markersize=10)
    plt.plot(np.asarray(x2, float),np.asarray(y2)*90.2,'r^-',label='VGGFace2',linewidth=3,markersize=10)
    #plt.plot()
    # plt.rc('font', size=15)
    # plt.rc('axes', titlesize=15)
    #plt.xticks([50,55,60,65,70,75,80,85,90,95,100],fontsize=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    #ax.set_xticks(np.round(np.linspace(50, 100, 5)))
    #plt.xlim([45,105])
    #plt.xlabel("Gaussian variance", fontsize=25)
    plt.xlabel("Quality factor", fontsize=25)
    plt.ylabel("Percentage($\%$)", fontsize=25)
    #plt.title("Gaussian Noise", fontsize=25)
    plt.title("JPEG Compression", fontsize=25)
    #plt.savefig('/data2/junliu/DeepSteg/data/figures/{}_JPEG.jpg'.format(filename))
    plt.savefig('/data2/junliu/DeepSteg/data/figures/{}_JPEG_v2.jpg'.format(filename))

def drawDictTotal():
    # with open('/data2/junliu/DeepSteg/data/testaccSVHNJPEG.json') as f:
    #     acctotal =json.load(f)
    with open('/data2/junliu/DeepSteg/data/testaccV2SVHNJPEG_modelcorrect.json') as f:
        acctotal =json.load(f)
    # with open('/data2/junliu/DeepSteg/data_vggface2/testaccSVHNJPEG.json') as f:
    #     acctotal2 =json.load(f)
    with open('/data2/junliu/DeepSteg/data_vggface2/testaccV2JPEG.json') as f:
        acctotal2 =json.load(f)
    # with open('/data2/junliu/DeepSteg/data/testaccSVHNGaussian.json') as f:
    #     acctotal3 =json.load(f)
    with open('/data2/junliu/DeepSteg/data/testaccV2SVHNGaussian_modelcorrect.json') as f:
        acctotal3 =json.load(f)
    # with open('/data2/junliu/DeepSteg/data_vggface2/testaccSVHNGaussian.json') as f:
    #     acctotal4 =json.load(f)   

    with open('/data2/junliu/DeepSteg/data_vggface2/testaccV2Gaussian.json') as f:
        acctotal4 =json.load(f)   

    del acctotal3['0.21'],acctotal3['0.24'],acctotal3['0.27'],acctotal3['0.3']
    del acctotal4['0.21'],acctotal4['0.24'],acctotal4['0.27'],acctotal4['0.3']

    # for k in acctotal:
    #     acctotal[k] = 0.96
    # for k in acctotal2:
    #     acctotal2[k] = 0.902

    # for i in range(30,50):
    #     acctotal[i] = 0.96
    #     acctotal2[i] = 0.902

    # with open('/data2/junliu/DeepSteg/data/testaccSVHNJPEG_p2.json') as f:
    #     acctotal5 =json.load(f)
    plt.clf()
    plt.figure(figsize=(20,8))
    plt.subplot(121)
    #acctotal.update(acctotal5)

    lists = sorted(acctotal.items(),key=intkey) # sorted by key, return a list of tuples
    lists2 = sorted(acctotal2.items(),key=intkey)
     
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    x2, y2 = zip(*lists2)
    plt.plot(np.asarray(x, float),np.asarray(y)*100,'bo-',label='SVHN',linewidth=3,markersize=10)
    plt.plot(np.asarray(x2, float),np.asarray(y2)*100,'r^-',label='VGGFace2',linewidth=3,markersize=10)
    #plt.plot()
    # plt.rc('font', size=15)
    # plt.rc('axes', titlesize=15)
    #plt.xticks([50,55,60,65,70,75,80,85,90,95,100],fontsize=15)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    #ax.set_xticks(np.round(np.linspace(50, 100, 5)))
    plt.ylim([0,100])
    #plt.xlabel("Gaussian variance", fontsize=25)
    plt.xlabel("Quality factor", fontsize=25)
    plt.ylabel("Percentage($\%$)", fontsize=25)
    #plt.title("Gaussian Noise", fontsize=25)
    plt.title("JPEG Compression", fontsize=25)

    plt.subplot(122)    
    lists = sorted(acctotal3.items()) 
    
    lists2 = sorted(acctotal4.items())   
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    x2, y2 = zip(*lists2)
    plt.plot(np.asarray(x, float),np.asarray(y)*100,'bo-',label='SVHN',linewidth=3,markersize=10)
    plt.plot(np.asarray(x2, float),np.asarray(y2)*100,'r^-',label='VGGFace2',linewidth=3,markersize=10)

    plt.xticks(np.arange(0,0.24,0.04),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.ylim([50,100])
    #plt.xlim([0,0.2])
    plt.xlabel("Gaussian variance", fontsize=25)
    plt.ylabel("Percentage($\%$)", fontsize=25)
    plt.title("Gaussian Noise", fontsize=25)
    #plt.savefig('/data2/junliu/DeepSteg/data/figures/Accuracy_total.jpg') 
    plt.savefig('/data2/junliu/DeepSteg/data/figures/Accuracy_total_v2.jpg')                     
def testAcc(stage=1):
    #threattype='Gaussian'
    threattype='JPEG'
    if stage==1:
        modelname='Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        net.eval()
        load_checkpoint(MODELS_PATH+modelname)
        total = 0
        inte_num = 10
        # qf
        #intensitylist = np.linspace(50,100,50,endpoint=True)
        #intensitylist = np.linspace(10,50,40,endpoint=True)
        intensitylist = np.linspace(0,9,40,endpoint=True)
        #intensitylist = np.linspace(50,100,2,endpoint=True,dtype=int)
        #gaussian
        #intensitylist = np.linspace(1,30,11,endpoint=True,dtype=np.uint8)*0.01
        intensitylist = list(intensitylist)
        intensitylist = [int(x) for x in intensitylist]
        #intensitylist.insert(0,0)
        intensity_res = [0]*len(intensitylist)
        acctotal=dict(zip(intensitylist,intensity_res))
        l2loss_secrets=dict(zip(intensitylist,intensity_res))
        l2loss_covers =dict(zip(intensitylist,intensity_res))
        psnr_secrets =dict(zip(intensitylist,intensity_res)) 
        ssim_secrets =dict(zip(intensitylist,intensity_res))
        cover_succ=0 
        mean_pixel_errors =dict(zip(intensitylist,intensity_res))
        total_all = 0
        lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs =dict(zip(intensitylist,intensity_res))\
            ,dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res))

        outputname=modelname.split(".")[0]+"_threat0"
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
            train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets),typen='uniform',mode='equal')        
            # Creates variable from secret and cover images
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

            for intensity in intensitylist:                
                mix_img,recover_secret = net(train_secrets,train_covers,train=False,\
                    threat_type=1,intensity=intensity,device=device) # to be [-1,1]
                acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
                    valmetric(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
                
                acctotal[int(intensity)] += int(acc_num)
                l2loss_secrets[int(intensity)] += float(l2loss_secret)
                l2loss_covers[int(intensity)] += float(l2loss_cover)
                psnr_secrets[int(intensity)] += float(psnr_secret)
                ssim_secrets[int(intensity)] += float(ssim_secret)
                mean_pixel_errors[int(intensity)] += float(mean_pixel_error)
                lpips_errors[int(intensity)] += float(lpips_error)
                lpips_scs[int(intensity)] += float(lpips_sc)
                mse_scs[int(intensity)] += float(mse_sc)
                psnr_scs[int(intensity)] += float(psnr_sc)
                ssim_scs[int(intensity)] += float(ssim_sc)
        for k in acctotal.keys():
            acctotal[k] = acctotal[k]/total
        for k in l2loss_secrets.keys():
            l2loss_secrets[k] = l2loss_secrets[k]/total
        for k in l2loss_covers.keys():
            l2loss_covers[k] = l2loss_covers[k]/total
        for k in psnr_secrets.keys():
            psnr_secrets[k] = psnr_secrets[k]/total
        for k in ssim_secrets.keys():
            ssim_secrets[k] = ssim_secrets[k]/total
        for k in mean_pixel_errors.keys():
            mean_pixel_errors[k] = mean_pixel_errors[k]/total
        for k in lpips_errors.keys():
            lpips_errors[k] = lpips_errors[k]/total
        for k in lpips_scs.keys():
            lpips_scs[k] = lpips_scs[k]/total
        for k in mse_scs.keys():
            mse_scs[k] = mse_scs[k]/total
        for k in psnr_scs.keys():
            psnr_scs[k] = psnr_scs[k]/total
        for k in ssim_scs.keys():
            ssim_scs[k] = ssim_scs[k]/total

        with open('/data2/junliu/DeepSteg/data/testaccSVHN{}_p3.json'.format(threattype),'w') as f:
            json.dump(acctotal,f)
        with open('/data2/junliu/DeepSteg/data/testPSNRSVHN{}_p3.json'.format(threattype),'w') as f:
            json.dump(psnr_secrets,f)
        with open('/data2/junliu/DeepSteg/data/testSSIMSVHN{}_p3.json'.format(threattype),'w') as f:
            json.dump(ssim_secrets,f)
        with open('/data2/junliu/DeepSteg/data/testLIPIPSSVHN{}_p3.json'.format(threattype),'w') as f:
            json.dump(lpips_errors,f)
        with open('/data2/junliu/DeepSteg/data/testLIPIPPSVHN{}_p3.json'.format(threattype),'w') as f:
            json.dump(lpips_scs,f)
        with open('/data2/junliu/DeepSteg/data/testaccPSNRP{}_p3.json'.format(threattype),'w') as f:
            json.dump(psnr_scs,f)
        with open('/data2/junliu/DeepSteg/data/testaccSSIMP{}_p3.json'.format(threattype),'w') as f:
            json.dump(ssim_scs,f)
        
        for intensity in intensitylist: 
            print("attack success rate:{:.6f}".format(acctotal[intensity]))
            print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets[intensity]))
            print("avg. l2loss_covers:{:.6f}".format(l2loss_covers[intensity]))
            print("avg. psnr_secrets:{:.6f}".format(psnr_secrets[intensity]))
            print("avg. ssim_secrets:{:.6f}".format(ssim_secrets[intensity]))
            print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors[intensity]))
            print("avg. lpips_errors:{:.6f}".format(lpips_errors[intensity]))
            print("avg. lpips_scs:{:.6f}".format(lpips_scs[intensity]))
            print("avg. mse_scs:{:.6f}".format(mse_scs[intensity]))
            print("avg. psnr_scs:{:.6f}".format(psnr_scs[intensity]))
            print("avg. ssim_scs:{:.6f}".format(ssim_scs[intensity]))
    if stage==2:
        # with open('/data2/junliu/DeepSteg/data/testaccSVHN{}.json'.format(threattype)) as f:
        #     acctotal =json.load(f)
        # with open('/data2/junliu/DeepSteg/data_vggface2/testaccSVHN{}.json'.format(threattype)) as f:
        #     acctotal2 =json.load(f)
        # drawDict(acctotal,acctotal2,"Accuracy",threattype='JPEG')
        # drawDict(l2loss_secrets,"MSE")
        # drawDict(psnr_secrets,"PSNR")
        # drawDict(ssim_secrets,"SSIM")
        # drawDict(mean_pixel_errors,"APE")
        # drawDict(lpips_errors,"LIPIPS")
        # drawDict(lpips_scs,"LIPIPS_p")
        # drawDict(mse_scs,"MSE_p")
        # drawDict(psnr_scs,"PSNR_p")
        # drawDict(ssim_scs,"SSIM_p")
        drawDictTotal()
    print("finished.")

def testAccCorrect(stage=1,dataset=0,threattype_num=0):
    #threattype='Gaussian'
    if dataset ==0:
        datasetname='SVHN'
    else:
        print("not supported now.")
        return
    if threattype_num ==1:
        threattype='JPEG'
    elif threattype_num ==0:
        threattype='Gaussian'
    else:
        return
    print("data:{},threattype:{}".format(datasetname,threattype))
    if stage==1:
        if dataset==0:
            modelname='Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
        else:
            modelname='Epoch_1_3999_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume15_checkpoint.pth.tar'
        net.eval()
        load_checkpoint(MODELS_PATH+modelname)
        total = 0
        inte_num = 10
        if threattype_num==1:
            intensitylist = np.linspace(1,100,40,endpoint=True)
            #intensitylist = np.linspace(10,50,40,endpoint=True)
            #intensitylist = np.linspace(0,9,40,endpoint=True)
            #intensitylist = np.linspace(50,100,2,endpoint=True,dtype=int)
            intensitylist = [int(x) for x in intensitylist]
        #gaussian
        else:
            intensitylist = np.linspace(1,30,11,endpoint=True,dtype=np.uint8)*0.01
        intensitylist = list(intensitylist)
        
        #intensitylist.insert(0,0)
        intensity_res = [0]*len(intensitylist)
        acctotal_modelcorrect = dict(zip(intensitylist,intensity_res))
        acctotal=dict(zip(intensitylist,intensity_res))
        l2loss_secrets=dict(zip(intensitylist,intensity_res))
        l2loss_covers =dict(zip(intensitylist,intensity_res))
        psnr_secrets =dict(zip(intensitylist,intensity_res)) 
        ssim_secrets =dict(zip(intensitylist,intensity_res))
        cover_succ=0 
        mean_pixel_errors =dict(zip(intensitylist,intensity_res))
        total_all = 0
        lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs =dict(zip(intensitylist,intensity_res))\
            ,dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res)),dict(zip(intensitylist,intensity_res))

        outputname=modelname.split(".")[0]+"_threat_{}".format(threattype)
        print(outputname)
        if not os.path.exists(outputname):
            os.mkdir(outputname)
        totalModel = 0
        for idx, train_batch in enumerate(test_loader):

            train_secrets, labels  = train_batch
            train_secrets = train_secrets.to(device)
            labels = labels.to(device)
            total_all += train_secrets.shape[0]
            if dataset==0:
                if total_all<=1000:
                    continue
            totalModel += train_secrets.shape[0]
            outputs = target_model(train_secrets)
            pre = torch.argmax(outputs,dim=1)  
            if torch.all(pre!=labels):
                continue
            train_secrets = train_secrets[torch.where(pre==labels)]
            labels = labels[torch.where(pre==labels)]


            total  += train_secrets.shape[0]
            print("model recg acc:{}".format(total/totalModel))

            train_covers,succ = getAdvZ(imgsize,labels,len(train_secrets),len(train_secrets),typen='uniform',mode='equal')        
            # Creates variable from secret and cover images
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)

            for intensity in intensitylist:                
                mix_img,recover_secret = net(train_secrets,train_covers,train=False,\
                    threat_type=threattype_num,intensity=intensity,device=device) # to be [-1,1]
                acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
                    valmetric(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
                if threattype_num==1:
                    kk = int(intensity)
                else:
                    kk = intensity
                acctotal[kk] += int(acc_num)
                acctotal_modelcorrect[kk] += int(acc_num)
                l2loss_secrets[kk] += float(l2loss_secret)
                l2loss_covers[kk] += float(l2loss_cover)
                psnr_secrets[kk] += float(psnr_secret)
                ssim_secrets[kk] += float(ssim_secret)
                mean_pixel_errors[kk] += float(mean_pixel_error)
                lpips_errors[kk] += float(lpips_error)
                lpips_scs[kk] += float(lpips_sc)
                mse_scs[kk] += float(mse_sc)
                psnr_scs[kk] += float(psnr_sc)
                ssim_scs[kk] += float(ssim_sc)

            if total>=500:
                break
        for k in acctotal.keys():
            acctotal[k] = acctotal[k]/total
        for k in acctotal.keys():
            acctotal_modelcorrect[k] = acctotal_modelcorrect[k]/totalModel
        for k in l2loss_secrets.keys():
            l2loss_secrets[k] = l2loss_secrets[k]/total
        for k in l2loss_covers.keys():
            l2loss_covers[k] = l2loss_covers[k]/total
        for k in psnr_secrets.keys():
            psnr_secrets[k] = psnr_secrets[k]/total
        for k in ssim_secrets.keys():
            ssim_secrets[k] = ssim_secrets[k]/total
        for k in mean_pixel_errors.keys():
            mean_pixel_errors[k] = mean_pixel_errors[k]/total
        for k in lpips_errors.keys():
            lpips_errors[k] = lpips_errors[k]/total
        for k in lpips_scs.keys():
            lpips_scs[k] = lpips_scs[k]/total
        for k in mse_scs.keys():
            mse_scs[k] = mse_scs[k]/total
        for k in psnr_scs.keys():
            psnr_scs[k] = psnr_scs[k]/total
        for k in ssim_scs.keys():
            ssim_scs[k] = ssim_scs[k]/total

        with open('/data2/junliu/DeepSteg/data/testacc{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(acctotal,f)
        with open('/data2/junliu/DeepSteg/data/testaccV2{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(acctotal_modelcorrect,f)
        with open('/data2/junliu/DeepSteg/data/testPSNR{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(psnr_secrets,f)
        with open('/data2/junliu/DeepSteg/data/testSSIM{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(ssim_secrets,f)
        with open('/data2/junliu/DeepSteg/data/testLIPIPS{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(lpips_errors,f)
        with open('/data2/junliu/DeepSteg/data/testLIPIPP{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(lpips_scs,f)
        with open('/data2/junliu/DeepSteg/data/testaccPSNRP{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(psnr_scs,f)
        with open('/data2/junliu/DeepSteg/data/testaccSSIMP{}{}_modelcorrect.json'.format(datasetname,threattype),'w') as f:
            json.dump(ssim_scs,f)
        
        for intensity in intensitylist: 
            print("attack success rate:{:.6f}".format(acctotal[intensity]))
            print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets[intensity]))
            print("avg. l2loss_covers:{:.6f}".format(l2loss_covers[intensity]))
            print("avg. psnr_secrets:{:.6f}".format(psnr_secrets[intensity]))
            print("avg. ssim_secrets:{:.6f}".format(ssim_secrets[intensity]))
            print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors[intensity]))
            print("avg. lpips_errors:{:.6f}".format(lpips_errors[intensity]))
            print("avg. lpips_scs:{:.6f}".format(lpips_scs[intensity]))
            print("avg. mse_scs:{:.6f}".format(mse_scs[intensity]))
            print("avg. psnr_scs:{:.6f}".format(psnr_scs[intensity]))
            print("avg. ssim_scs:{:.6f}".format(ssim_scs[intensity]))
    if stage==2:
        # with open('/data2/junliu/DeepSteg/data/testaccSVHN{}.json'.format(threattype)) as f:
        #     acctotal =json.load(f)
        # with open('/data2/junliu/DeepSteg/data_vggface2/testaccSVHN{}.json'.format(threattype)) as f:
        #     acctotal2 =json.load(f)
        # drawDict(acctotal,acctotal2,"Accuracy",threattype='JPEG')
        # drawDict(l2loss_secrets,"MSE")
        # drawDict(psnr_secrets,"PSNR")
        # drawDict(ssim_secrets,"SSIM")
        # drawDict(mean_pixel_errors,"APE")
        # drawDict(lpips_errors,"LIPIPS")
        # drawDict(lpips_scs,"LIPIPS_p")
        # drawDict(mse_scs,"MSE_p")
        # drawDict(psnr_scs,"PSNR_p")
        # drawDict(ssim_scs,"SSIM_p")
        drawDictTotal()
    print("finished.")

def getmeanbit():
    img_all = torch.zeros((batch_size,3,imgsize,imgsize)).to(device)
    for idx, train_batch in enumerate(train_loader):    
        imgs,labels = train_batch
        imgs = imgs.to(device)
        imgs = torch.round((imgs*_std_torch+_mean_torch)*255.)
        img_all += imgs
    img_train_mean = torch.sum(img_all,dim=0)/((idx+1)*batch_size)
    for idx_, train_batch_ in enumerate(test_loader):    
        imgs,labels = train_batch_
        imgs = imgs.to(device)
        imgs = torch.round((imgs*_std_torch+_mean_torch)*255.)
        img_all += imgs
    img_svhn_mean = torch.sum(img_all,dim=0)/((idx_+1+idx+1)*batch_size)
    #np.save('svhntestloadermeanpixel.npy',img_mean.cpu().numpy())
    np.save('svhntrainloadermeanpixel.npy',img_train_mean.cpu().numpy())
    np.save('svhnmeanpixel.npy',img_svhn_mean.cpu().numpy())
    print("finished.")
    for i in range(0,5):
        print("inner i:{}".format(i))
    print("out i:{}".format(i))

#torch.bitwise_and
def getlsb(secret,cover):
    #secret,cover : uint8 numpy array
    svhnmean = np.load('svhnmeanpixel.npy')
    
    #secret,cover : uint8
    bit_c = 4
    msb =0
    lsb = 0
    for i in range(0,bit_c):
        msb+= pow(2,7-i)
        lsb+= pow(2,i)
    secret_msb = np.bitwise_and(msb,secret)
    secret_msb_ = np.right_shift(secret_msb,4)
    cover_lsb0 = np.bitwise_and(pow(2,8)-1-lsb,cover) #将cover的lsb设置为0
    mix_img = secret_msb_+cover_lsb0
    svhnmean_lsb = np.bitwise_and(np.asarray(np.round(svhnmean),np.uint8),lsb)
    recover_secret =  np.left_shift(np.bitwise_and(mix_img,lsb),4)+svhnmean_lsb
    return mix_img,recover_secret
def testLsb():
    #outputname='lsb'
    outputname='lsb_svhn_uniform'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    acctotal_all,total,acctotal = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
    for idx, test_batch in enumerate(test_loader):
        test_secrets, labels  = test_batch
        acctotal_all += len(test_secrets)        
        if acctotal_all < 1000:
            continue
        if acctotal_all>=1500:
            break

        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        total += len(test_secrets)
        test_covers,succ = getAdvZ(imgsize,labels,batch_size,len(test_secrets),typen='uniform',mode='equal')
        test_secrets_integer = np.asarray(np.round(((test_secrets*_std_torch+_mean_torch)*255.).cpu().numpy()),dtype=np.uint8)
        test_covers_integer = np.asarray(np.round(test_covers.cpu().numpy()*255.),dtype=np.uint8)
        mix_img,recover_secret = getlsb(test_secrets_integer,test_covers_integer)
        recover_secret = torch.Tensor(recover_secret/255.).to(device)
        mix_img = torch.Tensor(mix_img/255.).to(device)

        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= valmetric(recover_secret,mix_img, test_secrets,test_covers,beta,labels)
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
        
        diff = mix_img-test_covers
        diff = (diff-torch.min(diff))/(torch.max(diff)-torch.min(diff))
        toshow = torch.cat((test_secrets[:4]*_std_torch+_mean_torch,test_covers[:4],mix_img[:4],recover_secret[:4],diff[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)


        # attack success rate:600/600=1.000000
        # avg. l2loss_secrets:1.514811
        # avg. l2loss_covers:1.270829
        # avg. psnr_secrets:31.273988
        # avg. ssim_secrets:0.955259
        # avg. mean_pixel_errors:5.699933

        #all test dataset:
        # attack success rate:26032/26032=1.000000
        # avg. l2loss_secrets:1.513998
        # avg. l2loss_covers:1.267719
        # avg. psnr_secrets:31.278764
        # avg. ssim_secrets:0.954861
        # avg. mean_pixel_errors:5.698281
    print("finished.")

def testRandomSeed():
    def _int32(x):
        return int(0xFFFFFFFF & x) #0xFFFFFFFF=2**32-1 相当于32位全是1的数 和x 做&操作, 相当于限制结果在0-(2**32-1)之间

    class MT19937:
        def __init__(self, seed):
            self.mt = [0] * 624
            self.mt[0] = seed
            self.mti = 0
            for i in range(1, 624):
                self.mt[i] = _int32(1812433253 * (self.mt[i - 1] ^ self.mt[i - 1] >> 30) + i) #先>>再^


        def extract_number(self):
            if self.mti == 0:
                self.twist()
            y = self.mt[self.mti]
            y = y ^ y >> 11
            y = y ^ y << 7 & 2636928640
            y = y ^ y << 15 & 4022730752
            y = y ^ y >> 18
            self.mti = (self.mti + 1) % 624
            return _int32(y)


        def twist(self):
            for i in range(0, 624):
                y = _int32((self.mt[i] & 0x80000000) + (self.mt[(i + 1) % 624] & 0x7fffffff))
                self.mt[i] = (y >> 1) ^ self.mt[(i + 397) % 624]

                if y % 2 != 0:
                    self.mt[i] = self.mt[i] ^ 0x9908b0df
    gen = MT19937(1)
    for i in range(0,10):
        data = gen.extract_number()
        print(data)
    print("finished.")



def testSingle():


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
    outputname='20221122forpaper'
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
        train_covers,succ = getAdvZ(imgsize,labels,batch_size,len(train_secrets),typen='uniform',mode='equal')

        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)

        lidx = 0
        best_privacyeps=[139,159,159,199,159,199,119,179,179]
        save_image(train_covers[0],'{}/cover.png'.format(outputname),normalize=False)
        for lambda_ in range(1,10):
            ep = best_privacyeps[lidx]
            lidx = lidx+1
            if lambda_ == 8:
                modelname='Epoch_{}_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'.format(ep)
            else:
                modelname='Epoch_{}_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_a0@{}_checkpoint.pth.tar'.format(ep,lambda_)

            net.eval()
            load_checkpoint(MODELS_PATH+modelname)

        
            mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]
            
            acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc= \
                valmetric(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
            diff = mix_img-train_covers
            diff = (diff-torch.min(diff))/(torch.max(diff)-torch.min(diff))

            save_image(train_secrets[0]*_std_torch+_mean_torch,'{}/{}_secret_lambda{}.png'.format(outputname,idx,lambda_),normalize=False)
            save_image(recover_secret[0],'{}/{}_recovered_lambda{}.png'.format(outputname,idx,lambda_),normalize=False)
            save_image(mix_img[0],'{}/{}_miximg_lambda{}.png'.format(outputname,idx,lambda_),normalize=False)
            save_image(diff[0],'{}/{}_diff_lambda{}.png'.format(outputname,idx,lambda_),normalize=False)
        print("a")
    print("finished.")

def testAblation():
    total_all = 0
    total = 0
    for idx, train_batch in enumerate(test_loader):
        train_secrets, labels  = train_batch
        train_secrets = train_secrets.to(device)
        total_all += train_secrets.shape[0]
        if total_all<=1000:
            continue
        if idx==1:
            break
        total  += train_secrets.shape[0]
        labels = labels.to(device)
        train_covers,succ,n = getAdvZ(imgsize,labels,batch_size,len(train_secrets),typen='uniform',mode='equal',zo=True)
        #np.save('/data2/junliu/DeepSteg/data/uniformnoise.npy',n.detach().cpu().numpy())
        # Creates variable from secret and cover images
        train_secrets = Variable(train_secrets, requires_grad=False)
        train_covers = Variable(train_covers, requires_grad=False)

        residual_en = [True,True,False]
        residual_de = [True,True,False]
        lambda_net = 0.8
        modelnames=['Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar',
            'Epoch_199_svhn_noval_uniformnoiscover_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar',
            'Epoch_159_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_endenores_checkpoint.pth.tar']
        mix_imgs = None
        recover_secrets = None
        for j in range(0,3):  
        
            print("encoder residual learning:{}; decoder:{}, lambda_net:{}".format(residual_en,residual_de,lambda_net))
            net = Net(residual_en[j],residual_de[j],lambda_net)
            net = torch.nn.DataParallel(net).cuda()

            net.eval()
            load_checkpoint(MODELS_PATH+modelnames[j],net)
            if j==2:
                mix_img,recover_secret = net(train_secrets,n,train=False) # to be [-1,1]  
            else:
                mix_img,recover_secret = net(train_secrets,train_covers,train=False) # to be [-1,1]    
            if mix_imgs is None:
                mix_imgs = mix_img
                recover_secrets = recover_secret
            else:
                mix_imgs = torch.cat((mix_imgs,mix_img),dim=0)
                recover_secrets = torch.cat((recover_secrets,recover_secret),dim=0)
        print("a")
        idx = 1
        toshow= torch.cat(
            (
            torch.cat(((train_secrets[idx].unsqueeze(0)+1)*0.5,(train_secrets[idx].unsqueeze(0)+1)*0.5,(train_secrets[idx].unsqueeze(0)+1)*0.5),dim=0),\
            torch.cat((train_covers[idx].unsqueeze(0),n[idx].unsqueeze(0).to(device),train_covers[idx].unsqueeze(0)),dim=0),
            torch.cat((mix_imgs[idx].unsqueeze(0),mix_imgs[len(train_secrets)+idx].unsqueeze(0),mix_imgs[len(train_secrets)*2+idx].unsqueeze(0)),dim=0),
            torch.cat((recover_secrets[idx].unsqueeze(0),recover_secrets[len(train_secrets)+idx].unsqueeze(0),recover_secrets[len(train_secrets)*2+idx].unsqueeze(0)),dim=0)),dim=0)
        toshow= torch.cat((
            (train_secrets+1)*0.5,
            mix_imgs
        ),dim=0)
        
        imgg = make_grid(toshow,nrow=len(train_secrets),padding=1)
        save_image(imgg,'ablation13.jpg')


def testModelAcc():

    total_all = 0
    acc_num_total=0
    for idx, test_batch in enumerate(test_loader):

        test_secrets, labels  = test_batch
        total_all += test_secrets.shape[0]
        if total_all<=1000:
            continue
        if total_all>1500:
            break
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        
        outputs = target_model(test_secrets)
        #outputs = target_model((test_secrets+1)/2)

        pre = torch.argmax(outputs,dim=1)
        acc_num = len(torch.where(pre==labels)[0])   
        acc_num_total += acc_num 
    print("acc:{}".format(acc_num_total/500)) #96 [-1,1] 90.6:[0,1]

if __name__ =="__main__":
    #testAccCorrect(stage=1,dataset=1,threattype_num=1)
    #testModelAcc()
    #testAblation()
    #drawDictTotal()
    #testAcc(2)
    #threatSplitDataset()#
    threatModel()
    #testSingle()
    #testRandomSeed()
    #testLsb()
    #getmeanbit()
    #testAblationAll()
    #test()
    # if mode== 'combine':
    #     net, mean_train_loss = train_model(train_loader, beta, learning_rate)
    # else:
    #     encoder,decoder, mean_train_loss = train_model(train_loader, beta, learning_rate)
    # Plot loss through epochs
    # plt.plot(loss_history)
    # plt.title('Model loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Batch')
    # plt.show()

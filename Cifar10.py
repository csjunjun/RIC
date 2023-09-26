# Imports necessary libraries and modules
#from itertools import islice
from distutils.log import error
from doctest import testfile
import os,io
from pickletools import optimize
from sympy import sec
from PIL import Image
from hidden.noise_layers.noiser import Noiser 
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import csv
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
from torchvision.transforms import ToPILImage,ToTensor
from random import shuffle
from torchvision.utils import make_grid,save_image
from facenet_pytorch import MTCNN, InceptionResnetV1
from junINN import innV
import juncw,juncwFace,junMIFGSMFace
from torch.utils.data import Dataset
import pandas as pd
import shutil
from datetime import datetime
import CifarResnetModel
from RIC import Net
# Directory path
# os.chdir("..")
# cwd = 'input'
testmode = False
cwd = os.getcwd()
# Hyper Parameters
ver_threshold = 1.242
print_freq = 200
val_freq = 200
valnum = 50
testnum = 500
num_epochs = 10
#batch_size = 32
batch_size = 10

face_size = 160

nrow_ = 4
learning_rate = 0.0001
beta = 10
imgsize=32
# Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
cuda = torch.cuda.is_available()

def setSeed(seed):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size        
        torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False    #现实的test阶段需要实现相同的attack的时候设置为false,否则model输出会稍有不同

setSeed(1337)
# TODO: Define train, validation and models
MODELS_PATH = '/data2/junliu/DeepSteg/output/'
# TRAIN_PATH = cwd+'/train/'
# VALID_PATH = cwd+'/valid/'
# VALID_PATH = cwd+'/sample/valid/'
# TRAIN_PATH = cwd+'/sample/train/'
# TEST_PATH = cwd+'/test/'
VGGface2_basepath = '/data2/junliu/data/vggface2/'
TRAIN_PATH = '/data2/junliu/data/ImageNet_train_jinyu'
TEST_PATH ='/data2/junliu/data/ImageNet_val'
if not os.path.exists(MODELS_PATH): os.mkdir(MODELS_PATH)
device = torch.device("cuda:0" if cuda else "cpu")
target_model = CifarResnetModel.ResNet18()
cp = torch.load('/data2/junliu/data/modelweights/ckpt_resnet18_400e.pth')

target_model= torch.nn.DataParallel(target_model).cuda()
target_model.load_state_dict(cp['net'])
target_model.eval()
#target_model = torch.nn.DataParallel(target_model).to(device) #如果这个模型用parallel,攻击成功率会下降很多，原因还有待考察
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
    


def attack1_loss(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    

    loss_secret = torch.nn.functional.mse_loss(S_prime,  S*_std_torch+_mean_torch)
    loss_cover= torch.nn.functional.mse_loss(mix_img,  C)
    outputs = target_model(mix_img)
    classloss = torch.mean(getCEloss(labels,outputs))
    
    loss_all =   B*loss_secret  + loss_cover + classloss
    #loss_all = -loss_cover + B * loss_secret + classloss + loss_cover2

    
    return loss_all, loss_secret,classloss,loss_cover

from skimage.metrics import peak_signal_noise_ratio, structural_similarity,mean_squared_error
import sys
sys.path.append('/home/junliu/code/Universal-Deep-Hiding')
import PerceptualSimilarity.models
modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])


def valmetric_nonorm(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    outputs = target_model(mix_img)
    pre = torch.argmax(outputs,dim=1)
    acc_num = len(torch.where(pre==labels)[0])    
    psnr, ssim, mse_s,mse_c,mean_pixel_error,lpips_error = [], [], [],[],[],[]
    norm_S =  convert1(S)
    norm_S_prime = convert1(S_prime)
    norm_mix_image = convert1(mix_img)
    mse_sc,psnr_sc,ssim_sc,lpips_sc = [], [], [],[]
    for i in range(len(S_prime)):
        # mse_s.append(mean_squared_error(norm_S_prime[i], norm_S[i]))
        # mse_c.append(mean_squared_error(norm_miximg[i], norm_C[i]))
        mse_s.append(float(torch.norm((S)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, multichannel=True))
        mean_pixel_error.append(float(torch.sum(torch.abs(torch.round((S)[i]*255)-torch.round(S_prime[i]*255)))/(3*imgsize*imgsize)))
        tmp = modellp.forward((S)[i], S_prime[i],normalize=True)
        lpips_error.append(float(tmp))
        #mix_image and secret image
        mse_sc.append(float(torch.norm((S)[i]-mix_img[i])))
        psnr_sc.append(peak_signal_noise_ratio(norm_S[i],norm_mix_image[i],data_range=255))
        ssim_sc.append(structural_similarity(norm_S[i],norm_mix_image[i],win_size=11, data_range=255.0, multichannel=True))
        tmp = modellp.forward((S)[i], mix_img[i],normalize=True)
        lpips_sc.append(float(tmp))
    #ssim_secret =0 
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error),np.sum(lpips_error),np.sum(lpips_sc),np.sum(mse_sc),np.sum(psnr_sc),np.sum(ssim_sc)



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

def facevalmetric(S_prime, mix_img, S,C):
    ''' Calculates loss specified on the paper.'''
    embed_anchor = target_model(S)
    embed = target_model(mix_img)
    diff = torch.norm(embed-embed_anchor,dim=1)

    acc_num = len(torch.where(diff<ver_threshold)[0])    
    psnr, ssim, mse_s,mse_c = [], [], [],[]
    norm_S =  convert1(S)
    norm_S_prime = convert1(S_prime)
    norm_miximg = convert1(mix_img)
    norm_C = convert1(C)
    for i in range(len(S_prime)):
        # mse_s.append(mean_squared_error(norm_S_prime[i], norm_S[i]))
        # mse_c.append(mean_squared_error(norm_miximg[i], norm_C[i]))
        mse_s.append(float(torch.norm((S)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, multichannel=True))
    #ssim_secret =0 
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim)

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


# class SCSEModule(nn.Module):
#     # according to https://arxiv.org/pdf/1808.08127.pdf concat is better
#     def __init__(self, channels, reduction=16, mode='concat'):
#         super(SCSEModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()

#         self.spatial_se = nn.Sequential(
#             nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Sigmoid(),
#         )
#         self.mode = mode

#     def forward(self, x):
#         module_input = x

#         x = self.avg_pool(x)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         chn_se = self.sigmoid(x)
#         chn_se = chn_se * module_input

#         spa_se = self.spatial_se(module_input)
#         spa_se = module_input * spa_se
#         if self.mode == 'concat':
#             return torch.cat([chn_se, spa_se], dim=1)
#         elif self.mode == 'maxout':
#             return torch.max(chn_se, spa_se)
#         else:
#             return chn_se + spa_se



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
#         # self.scse1 = SCSEModule(50,2,'maxout') #concat模式的话输出是# 4,100,160,160
#         # self.scse2 = SCSEModule(50,2,'maxout')
#         # self.scse3 = SCSEModule(50,2,'maxout')
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

#         # p4 = self.scse1(p4) 
#         # p5 = self.scse2(p5)
#         # p6 = self.scse3(p6)

#         out = torch.cat((p4, p5, p6), 1)
#         return out #(bs,150,160,160)

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
        
#         #out = gaussian(out.data, 0, 0.1)
#         out = 0.2*out + 0.8*cover
#         return out #(bs,3,160,160)

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
#         return out #(bs,3,160,160)




from torch.nn.functional import avg_pool2d, interpolate, softmax
from torch.nn import UpsamplingBilinear2d, Upsample
#upsampler = Upsample(size=(imgsize, imgsize), align_corners=True, mode='bilinear')

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

#     def forward(self, secret, cover,train=True,testSensitivity=False,labels=None):
#         x_1 = self.m1(secret)
#         mid = torch.cat((x_1, cover), 1)
#         x_2 = self.m2(mid,cover)
        
#         x_2 = self.quan(x_2,type='noise',train=train)
#         #x_2 = self.getnoistadv(x_2)
#         if testSensitivity==True:
#             #setSeed(1337)
#             newcover,succ = getAdvZ(160,labels,len(secret),train=False)
#             diff = torch.norm(newcover-cover)
#             print(diff)
#             x_3 = self.m3(x_2,newcover)
#         else:
#             x_3 = self.m3(x_2,cover) #训练的时候不要用clamp 或者 sigmoid
#         if train== False:
#             x_3 = torch.clamp(x_3,0,1)
#         #x_3 = self.act(x_3)
#         return x_2, x_3

#     def getnoistadv(self,x):
#         img_size = 224
#         z0_noise = 0.1*torch.randn((batch_size,3,img_size,img_size)).to(device)
#         #z0_noise = 0.1*torch.rand((batch_size,3,img_size,img_size)).to(device)
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
#         return x/255.
# # Creates net object

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork()
        #self.m3 = RevealNetwork()

    def forward(self, secret, cover):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2 = self.m2(mid,cover)
        #x_3 = self.m3(x_2,cover)
        x_2 = self.quan(x_2)
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
        x_3 = self.act(x_3)
        return x_3    


def load_checkpoint(filepath):
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)

        start_epoch = checkpoint['epoch']
        
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))


#mode = 'split'
mode = 'combine'
if mode== 'combine':
    net = Net()
    
    net = torch.nn.DataParallel(net).to(device)
# Save optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5,mode='min')    
    #load_checkpoint(MODELS_PATH+'Epoch_1_faceclassification_noval_model_best_31331.pth.tar')
    #load_checkpoint(MODELS_PATH+'Epoch_1_faceclassification_se_resume_model_best.pth.tar')
else:
    encoder = Encoder().to(device)
    decoder= Decoder().to(device)

def get_id_label_map(meta_file):
    N_IDENTITY = 9131  # total number of identities in VGG Face2
    N_IDENTITY_PRETRAIN = 8631  # [0,8630]:training, [8631,9130] : tesing
    identity_list = meta_file
    #df = pd.read_csv(identity_list, sep=',\s+', quoting=csv.QUOTE_ALL, encoding="utf-8")
    df = pd.read_csv(identity_list, sep=',')
    df["class"] = -1
    df.loc[df["Flag"] == 1, "class"] = range(N_IDENTITY_PRETRAIN)
    df.loc[df["Flag"] == 0, "class"] = range(N_IDENTITY_PRETRAIN, N_IDENTITY)
    # print(df)
    key = df["Class_ID"].values
    val = df["class"].values
    id_label_dict = dict(zip(key, val))
    return id_label_dict
id_label_map = get_id_label_map(VGGface2_basepath+'label/identity_meta2.csv')
#https://discuss.pytorch.org/t/how-to-read-dataset-from-tar-files/47425

def split_train_val_test(txt_path):
    now = datetime.now() 
    timestamp = datetime.timestamp(now)
    timestamp = str(timestamp).split('.')[0]
    df = pd.read_csv(txt_path, sep=' ', index_col=0)
    img_names = df.index.values
    np.random.shuffle(img_names)
    trainfiles = img_names[:-(valnum+testnum)]
    valfiles = img_names[-(valnum+testnum):-testnum]
    testfiles = img_names[-testnum:]
    np.save(MODELS_PATH+'trainfiles_{}.npy'.format(timestamp),trainfiles)
    np.save(MODELS_PATH+'valfiles_{}.npy'.format(timestamp),valfiles)
    np.save(MODELS_PATH+'testfile_{}.npy'.format(timestamp),testfiles)
    print("trainfiles is saved as {}trainfiles_{}.npy".format(MODELS_PATH,timestamp))
    print("valfiles is saved as {}valfiles_{}.npy".format(MODELS_PATH,timestamp))
    print("testfiles is saved as {}testfile_{}.npy".format(MODELS_PATH,timestamp))
    return trainfiles,valfiles
#trainfiles,valfiles = split_train_val_test(VGGface2_basepath+'VGGFace2_train_list.txt')


# Creates test set 
# lwf_test_data = datasets.LFWPairs('/data2/junliu/data/',split='test',
#         transform=train_transform,download=False)
# lwf_test_data = datasets.LFWPeople('/data2/junliu/data/',split='test',
#         transform=train_transform,download=False)
# lwf_test_data = datasets.LFWPeople('/data2/junliu/data/',split="test",
#         transform=transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(imgsize),
#         transforms.ToTensor()
#         ]),download=False)

import torchvision
from torch.utils.data import DataLoader
def getCifar10(data_dir,batchsize,train=True,resize=False,isshuffle=False,resize_v=32):
    #transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    #transform_test  = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    transform_test  = transforms.Compose([       
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,std=std)
        ])

    #train_data = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform_train, download=True)
    test_data  = torchvision.datasets.CIFAR10(data_dir, train=True, transform=transform_test, download=False)

    #train_loader  = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4,pin_memory=True, drop_last=True)
    test_loader   = DataLoader(test_data,  batch_size=batchsize, shuffle=isshuffle, num_workers=1,
                            pin_memory=True, drop_last=True)
    return test_loader
# Creates test set
test_loader  = getCifar10('/home/junliu/data/cifar-10_data/',train=False,batchsize=batch_size,resize=False,isshuffle=False)

_mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
_std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)


def getAdvZ(img_size,labels,minibs,train=False):
    # for train
    # c=10
    # k=5
    # st=60
    c=10
    k=5
    #st=60
    #st=100
    st=150
    succ = False
    print("getadvz c:{},k:{},st:{}".format(c,k,st))
    #init random noise
    # z0 = nn.init.trunc_normal_(torch.zeros((1,3,img_size,img_size)),0,1,-2,2)
    # z0 = 0.25*(2+z0).to(device) 
    z0 = torch.rand((minibs,3,img_size,img_size))
    #z0 = (z0-torch.min(z0))/(torch.max(z0)-torch.min(z0))
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
    adv = cwatt(z0,labels)
    del cwatt
    #outputs = target_model((adv-_mean_torch)/_std_torch)
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)
    # if torch.all(pre==labels):
    #     succ = True
    succ = len(torch.where(pre==labels)[0])
    return adv,succ


# def getAdvZ(img_size,labels,batch_size,train=False):
#     c=10
#     #k=10
#     k=5
    
#     if train:
#         st=50 #train
#     else:
#         st = 80
#     #st = 300
#     st = 150
#     succ = False
#     print("getadvz c:{},k:{},st:{}".format(c,k,st))
#     #init random noise
#     # z0 = nn.init.trunc_normal_(torch.zeros((1,3,img_size,img_size)),0,1,-2,2)
#     # z0 = 0.25*(2+z0).to(device) 
#     z0 = torch.rand((batch_size,3,img_size,img_size))
#     #z0 = (z0-torch.min(z0))/(torch.max(z0)-torch.min(z0))
#     cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
#     adv = cwatt(z0,labels)
#     outputs = target_model(adv)
#     _,pre = torch.max(outputs,1)

#     # outputs_normed = target_model((adv-_mean_torch)/_std_torch)
#     # _,pre2 = torch.max(outputs_normed,1)

#     # if torch.any(pre!=pre2):
#     #     print("insame")
#     # if torch.all(pre==labels):
#     #     succ = True
#     succ = len(torch.where(pre==labels)[0])
#     return adv,succ


def val(outputname,epoch):
    # net.load_state_dict(torch.load(MODELS_PATH+'Epoch N4.pkl'))

    # Switch to evaluate mode
    net.eval()

    loss_history = []
    # Iterate over batches performing forward and backward passes
    test_loss_secret_history = []
    attloss_history = []
    z_loss_history = []  
    test_losses = []
    # Show images
    valcnt = 0
    outputname = outputname+'_val'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    for idx, test_batch in enumerate(val_loader):
        
        test_secrets, labels  = test_batch
        test_secrets = test_secrets.to(device)
        valcnt += len(test_secrets)
        # if valcnt > valnum:
        #     break

        labels = labels.to(device)
        test_covers,succ = getAdvZ(face_size,labels,len(test_secrets))
        
        # Creates variable from secret and cover images
        test_secrets = Variable(test_secrets, requires_grad=False)
        test_covers = Variable(test_covers, requires_grad=False)


        mix_img,recover_secret = net(test_secrets,test_covers,train=False) # to be [-1,1]
        val_loss, val_loss_secret,val_attloss,val_loss_cover = attack1_loss(recover_secret,mix_img, test_secrets,test_covers,beta,labels)                   

        #toshow = torch.cat((train_secrets[:4],train_covers[:4],(train_output[:4]+1)*0.5,(train_hidden[:4]+1)*0.5),dim=0)
        toshow = torch.cat((test_secrets[:4]*_std_torch+_mean_torch,test_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)

        test_losses.append(float(val_loss.data))
        

        test_loss_secret_history.append(float(val_loss_secret.data))
        attloss_history.append(float(val_attloss.data))
        z_loss_history.append(float(val_loss_cover.data))
                
        plotloss(test_losses,test_loss_secret_history,attloss_history,z_loss_history,outputname,epoch)   
            
        
            
    mean_test_loss = np.mean(test_losses)

    print ('Average loss on test set: {:.2f}'.format(mean_test_loss))
    return mean_test_loss, np.mean(test_loss_secret_history),np.mean(attloss_history),np.mean(z_loss_history)

def plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch):
    plt.clf()
    plt.plot(train_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/lossCurve_{}.png'.format(outputname,epoch))  

    plt.clf()
    plt.plot(train_loss_secret_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/secrectlossCurve_{}.png'.format(outputname,epoch))  

    plt.clf()
    plt.plot(attloss_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/attlossCurve_{}.png'.format(outputname,epoch))  

    plt.clf()
    plt.plot(loss_cover_history)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Batch')
    plt.savefig('{}/coverlossCurve_{}.png'.format(outputname,epoch))  

   
def train_model(train_loader, beta, learning_rate):
    

    
    #load_checkpoint(MODELS_PATH+'Epoch_1_faceclassification_model_best.pth.tar')
    # Iterate over batches performing forward and backward passes
    train_loss_secret_history = []
    attloss_history = []
    loss_cover_history = []  
    #outputname = 'datanormadvnonorm_rightmetric'
    #outputname = 'datanormadvnonorm_rightmetric_splitloss'
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_again' #pid 16882
    #outputname = 'datanormadvnonorm_rightmetric_splitloss_sigmoid' #pid 53069
    #outputname = 'faceclassification_noval' #pid 31651 实际是se
    outputname = 'faceclassification_se_resume_resume_rightvalset' #pid  25240
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    best_val_loss_secret = 10000
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
            train_secrets = train_secrets[torch.where(labels<8631)]
            labels = labels[torch.where(labels<8631)]
            if len(labels)<1:
                continue
            train_secrets = train_secrets.to(device)

            labels = labels.to(device)
            train_covers,succ = getAdvZ(face_size,labels,len(train_secrets))
            cover_succ+= int(succ)
            print("epo:{} att z succ rate:{}".format(epoch,cover_succ/((idx+1)*batch_size)))
            #train_covers = (train_covers-torch.min(train_covers))/(torch.max(train_covers)-torch.min(train_covers))
            #train_covers = (train_covers-_mean_torch)/_std_torch
            # mean_torch = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).to(device)
            # std_torch = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).to(device)
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
                print("error.")
            
            
                mix_img = encoder(train_secrets,train_covers)
                
                #mix_img = torch.round(torch.clamp(mix_img,0,1)*255.)/255.
                recover_secret = decoder(mix_img,train_covers)

            train_loss, train_loss_secret,attloss,loss_cover = attack1_loss(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
            #show
            # train_covers_ = train_covers*_std_torch+_mean_torch
            # train_secrets_ = train_secrets*_std_torch+_mean_torch
            # train_output_ = train_output*_std_torch+_mean_torch
            # train_hidden_ = train_hidden*_std_torch+_mean_torch
            if (idx+1)%print_freq == 0:
                #toshow = torch.cat((train_secrets[:4],train_covers[:4],(train_output[:4]+1)*0.5,(train_hidden[:4]+1)*0.5),dim=0)
                toshow = torch.cat((train_secrets[:4]*_std_torch+_mean_torch,train_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
                imgg = make_grid(toshow,nrow=nrow_)
                save_image(imgg,'{}/{}_{}.png'.format(outputname,epoch,idx),normalize=False)

                plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch)
                       
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
    
            # val 
            if (idx+1)%val_freq == 0 or train_num-idx*(batch_size+1)<=batch_size or idx==1:
                val_loss, val_loss_secret,val_attloss,val_loss_cover = val(outputname,epoch)
                scheduler.step(val_loss_secret)
                is_best = False
                #is_best = True
                if val_loss_secret< best_val_loss_secret:
                    is_best = True
                if mode== 'combine':
                    modelsavepath =MODELS_PATH+'Epoch_{}_{}'.format(epoch+1,outputname)
                    save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler':scheduler.state_dict()
            },filename=modelsavepath,is_best=is_best)
                else:
                    torch.save(encoder.state_dict(), MODELS_PATH+'Epoch_N{}_{}_encoder.pkl'.format(epoch+1,outputname))
                    torch.save(decoder.state_dict(), MODELS_PATH+'Epoch_N{}_{}_decoder.pkl'.format(epoch+1,outputname))
             
        mean_train_loss = np.mean(train_losses)
    
        # Prints epoch average loss
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
        plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch) 
    if mode== 'combine':
    
        return net,mean_train_loss
    else:
        return encoder,decoder, mean_train_loss


def save_checkpoint(state, filename,is_best=False):
    checkpointname = filename+'_checkpoint.pth.tar'
    torch.save(state, checkpointname)
    if is_best:
        shutil.copyfile(checkpointname, filename+'_model_best.pth.tar')


def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

# LSB方法：
# c=10
# k=10
# st=50

# att z succ rate:1.0
# attack success rate:486/500=0.972000
# avg. l2loss_secrets:7.124824
# avg. l2loss_covers:6.553673
# avg. psnr_secrets:31.798312
# avg. ssim_secrets:0.868447
# avg. mean_pixel_errors:5.343677

# Epoch_1_faceclassification_se_resume_model_best.pth.tar
# att z succ rate:0.9977777777777778
# attack success rate:443/450=0.984444
# avg. l2loss_secrets:4.622669
# avg. l2loss_covers:8.984289
# avg. psnr_secrets:35.807160
# avg. ssim_secrets:0.961340
# avg. mean_pixel_errors:3.218821

def testTransfer():
    modelname='Epoch_1_1999_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume25_checkpoint.pth.tar'
    net.eval()
    load_checkpoint(MODELS_PATH+modelname)
    outputname=modelname.split(".")[0]+"_cifar"
    if not os.path.exists(outputname):
        os.mkdir(outputname)

    acctotal,total,cover_succ = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
    for idx, test_batch in enumerate(test_loader):
        if total>=500:
            break
        test_secrets, labels  = test_batch
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        total += len(test_secrets)
        test_covers,succ = getAdvZ(imgsize,labels,batch_size,train=False)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/total))
        
        #mix_img,recover_secret = net(test_secrets,test_covers,train=False,testSensitivity=True,labels=labels)
        mix_img,recover_secret = net(test_secrets,test_covers,train=False)
        
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,\
            lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc=\
                valmetric(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
                #valmetric_nonorm(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
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
        #toshow = torch.cat((test_secrets[:4],test_covers[:4],mix_img[:4],recover_secret[:4],diff[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)

    print("finished.")    
def test():
    #modelname='Epoch_1_faceclassification_se_resume_resume_model_best.pth.tar'
    #modelname='Epoch_1_faceclassification_se_resume_resume_rightvalset_model_best.pth.tar'
    modelname='Epoch_28_svhnagain_model_best.pth.tar'
    modelname='Epoch_1_3399_faceclassification_se_rightface_noval_uniformcover_resume2_checkpoint.pth.tar'
    #modelname='Epoch_99_svhn_noval_uniform_rightcw_nonormz_checkpoint.pth.tar'
    modelname='Epoch_1_2599_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume4_checkpoint.pth.tar'
    modelname='Epoch_1_199_face_se_rightface_noval_uniformcover_ablation_rec_advloss_addquannoise_resume10_checkpoint.pth.tar'
    modelname='Epoch_119_svhn_noval_uniform_lossrec_adv_nonormz_addquannoise_checkpoint.pth.tar'
# att z succ rate:1.0
# attack success rate:497/500=0.994000
# avg. l2loss_secrets:4.447687
# avg. l2loss_covers:8.696254
# avg. psnr_secrets:36.150030
# avg. ssim_secrets:0.964021
# avg. mean_pixel_errors:3.081834
    net.eval()
    load_checkpoint(MODELS_PATH+modelname)

    if mode == 'combine':
        #outputname=modelname.split(".")[0]+"_cifar10bysvhn_nonorm"
        #outputname=modelname.split(".")[0]+"_cifar10byvggface2_uniform"
        outputname=modelname.split(".")[0]+"_cifar10bysvhn_uniform"
    else:
        outputname=modelname.split(".")[0]
    #outputname =  'Epoch_1_faceclassification_model_best'
    if not os.path.exists(outputname):
        os.mkdir(outputname)

    acctotal,total,cover_succ = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.
    for idx, test_batch in enumerate(test_loader):
        if total>=500:
            break
        test_secrets, labels  = test_batch
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        total += len(test_secrets)
        test_covers,succ = getAdvZ(imgsize,labels,batch_size,train=False)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/total))
        
        #mix_img,recover_secret = net(test_secrets,test_covers,train=False,testSensitivity=True,labels=labels)
        mix_img,recover_secret = net(test_secrets,test_covers,train=False)
        
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,\
            lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc=\
                valmetric(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
                #valmetric_nonorm(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
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
        #toshow = torch.cat((test_secrets[:4],test_covers[:4],mix_img[:4],recover_secret[:4],diff[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)

    print("finished.")    
def testold():
    if mode == 'combine':
        modelname='Epoch_1_faceclassification_noval_model_best_31331.pth.tar'#se



# attack success rate:4/4=1.000000
# avg. l2loss_secrets:4.898735
# avg. l2loss_covers:7.180846
# avg. psnr_secrets:35.137370
# avg. ssim_secrets:0.951425
# avg. mean_pixel_errors:3.518571

# attack success rate:4/4=1.000000
# avg. l2loss_secrets:6.010477
# avg. l2loss_covers:6.791642
# avg. psnr_secrets:33.307084
# avg. ssim_secrets:0.953663
# avg. mean_pixel_errors:4.478460
        modelname='Epoch_1_faceclassification_se_resume_model_best.pth.tar'

# att z succ rate:1.0
# attack success rate:50/50=1.000000
# avg. l2loss_secrets:4.649905
# avg. l2loss_covers:8.849426
# avg. psnr_secrets:35.764385
# avg. ssim_secrets:0.959223
# avg. mean_pixel_errors:3.243926
        net.eval()
        load_checkpoint(MODELS_PATH+modelname)
        #net.load_state_dict(torch.load(MODELS_PATH+modelname)['state_dict'])
    # elif mode == 'split':
    #     enmodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_again_encoder.pkl'
    #     demodelname = 'Epoch_N1_datanormadvnonorm_rightmetric_splitloss_again_decoder.pkl'
    #     encoder.eval()
    #     decoder.eval()
    #     encoder.load_state_dict(torch.load(MODELS_PATH+enmodelname))
    #     decoder.load_state_dict(torch.load(MODELS_PATH+demodelname))
    total = 0
    acctotal = 0
    l2loss_secrets = 0 
    l2loss_covers = 0 
    psnr_secrets =0 
    ssim_secrets =0 
    cover_succ=0
    mean_pixel_errors =0 
    if mode == 'combine':
        outputname=modelname.split(".")[0]
    else:
        outputname=modelname.split(".")[0]
    #outputname =  'Epoch_1_faceclassification_model_best'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    trans = transforms.ToPILImage()
    trans_v = transforms.ToTensor()
    correct_num =0
    testdata = np.load(MODELS_PATH+'testfile_noval_noresize_se.npy',allow_pickle=True)
    faces =torch.zeros((1,3,160,160)) 
    test_coverss =torch.zeros((1,3,160,160))
    mix_faces =torch.zeros((1,3,160,160))
    rec_faces =torch.zeros((1,3,160,160))



    for i in range(len(testdata)):
        total += 1
        if total == 501:
            break
        image = Image.open(VGGface2_basepath+'train/'+testdata[i])
        face = mtcnn(image)
        #assert torch.max(face)<=1 or torch.min(face)>=-1
        label = id_label_map[testdata[i].split("/")[0]]
        label = torch.Tensor([label]).long().to(device)
        if face is None:
            face = trans_v(image)
        if face.shape[1]!=face_size or face.shape[2] !=face_size:
            face = image.resize((face_size,face_size))
            face = trans_v(face)
        face = face.unsqueeze(0).to(device)
        test_covers,succ = getAdvZ(face_size,label,1)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/(i+1)))
        mix_face,rec_face = net(face,test_covers,train=False)
        #test diff
        
        # diff = mix_face-test_covers
        # diff = (mix_face-test_covers*0.8)/0.2
        # diff = (mix_face-test_covers*0.8)/0.2
        #diff = mix_img - train_covers
        #diff = (mix_img-train_covers*0.8)/0.2
        #imgg = make_grid(diff,nrow=1)
        #save_image(imgg,'{}/{}_mixascover_facediff_coef.png'.format(outputname,i),normalize=True)
        #
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error= valmetric(rec_face,mix_face, face,test_covers,beta,label)
        acctotal += int(acc_num)
        l2loss_secrets += float(l2loss_secret)
        l2loss_covers += float(l2loss_cover)
        psnr_secrets += float(psnr_secret)
        ssim_secrets += float(ssim_secret)
        mean_pixel_errors += float(mean_pixel_error)

        print("attack success rate:{}/{}={:.6f}".format(acctotal,total,acctotal/total))
        print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets/total))
        print("avg. l2loss_covers:{:.6f}".format(l2loss_covers/total))
        print("avg. psnr_secrets:{:.6f}".format(psnr_secrets/total))
        print("avg. ssim_secrets:{:.6f}".format(ssim_secrets/total))
        print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors/total))
        #show

    
        #toshow = torch.cat((train_secrets[:4],train_covers[:4],(train_output[:4]+1)*0.5,(train_hidden[:4]+1)*0.5),dim=0)

        faces = torch.cat((faces,face.cpu()),dim=0)
        test_coverss = torch.cat((test_coverss,test_covers.cpu()),dim=0)
        mix_faces = torch.cat((mix_faces,mix_face.cpu()),dim=0)
        rec_faces = torch.cat((rec_faces,rec_face.cpu()),dim=0)
        if len(faces)==5:
            toshow = torch.cat((faces[1:]*_std_torch.cpu()+_mean_torch.cpu(),test_coverss[1:],mix_faces[1:],rec_faces[1:]),dim=0)
            imgg = make_grid(toshow,nrow=nrow_)
            #save_image(imgg,'{}/{}.png'.format(outputname,i),normalize=False)  
            faces =torch.zeros((1,3,160,160))
            test_coverss =torch.zeros((1,3,160,160))
            mix_faces =torch.zeros((1,3,160,160))
            rec_faces =torch.zeros((1,3,160,160))  

    print("finished.")

    
def testVer():
    
    total = 0
    acctotal = 0
    l2loss_secrets = 0 
    l2loss_covers = 0 
    psnr_secrets =0 
    ssim_secrets =0 
    cover_succ=0
    # if mode == 'combine':
    #     outputname=modelname.split(".")[0]
    # else:
    #     outputname=demodelname.split(".")[0]
    outputname =  'facelfwtest'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    trans = transforms.ToPILImage()
    mtcnn =MTCNN(image_size=face_size)
    correct_num =0
    total_num = 0
    for idx, train_batch in enumerate(test_loader):
        if total_num>500:
            break
        p1,p2,flag  = train_batch
        p1 = p1.to(device)
        p2 = p2.to(device)
        flag = flag.to(device)
        total_num += len(p1)
        for kk in range(len(p1)):
            if kk == 0:
                p1_cropped = mtcnn(trans(p1[kk])).unsqueeze(0)
                p2_cropped = mtcnn(trans(p2[kk])).unsqueeze(0)
            else:
                p1_cropped = torch.cat((p1_cropped,mtcnn(trans(p1[kk])).unsqueeze(0)),dim=0)
                p2_cropped = torch.cat((p2_cropped,mtcnn(trans(p2[kk])).unsqueeze(0)),dim=0)
        del p1,p2


        embed1 = target_model(p1_cropped.to(device))
        embed2 = target_model(p2_cropped.to(device))
         # from vggface2 paper FaceNet: A Unified Embedding for Face Recognition and Clustering
        diff = torch.norm(embed1-embed2,dim=1)
        pred = diff<ver_threshold
        correct_num += len(torch.where(pred==flag)[0])
        print("acc:{:.4f}/{}".format(correct_num/total_num,total_num))
        

    print("finished.") #93.65 verification acc in LFW testset within 504 total pairs


def testOpenset():
    # net.load_state_dict(torch.load(MODELS_PATH+'Epoch N4.pkl'))

    # Switch to evaluate mode
    outputname = 'Epoch_1_faceclassification_model_best'
    net.eval()
    #verattack = juncwFace.CW(target_model,c=1,kappa=0.1,steps=500,lr=0.01)
    #verattack = junMIFGSMFace.MIFGSM(target_model,eps=1,steps=2000)
    
    load_checkpoint(MODELS_PATH+'Epoch_1_faceclassification_se_resume_model_best.pth.tar')
    loss_history = []
    # Iterate over batches performing forward and backward passes
    mse_ss = []
    mse_cs = []
    psnrs = []  
    ssims = []
    # Show images
    valcnt = 0
    outputname = outputname+'_vallfw'
    acc_num_total =0
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    trans_v = transforms.ToPILImage()
    for idx, test_batch in enumerate(test_loader):
        secret_face,labels  = test_batch
        face = mtcnn(trans_v(secret_face)).unsqueeze(0).to(device)
        secret_face = secret_face.to(device)
        pred_labels = torch.argmax(target_model(secret_face),dim=1)
        true_labels = labels.to(device)

        
        valcnt += len(secret_face)
        if valcnt > valnum:
            break

        noise_adv,succ = getAdvZ(160,pred_labels,batch_size)
        
        # Creates variable from secret and cover images
        test_secrets = Variable(secret_face, requires_grad=False)
        test_covers = Variable(noise_adv, requires_grad=False)


        mix_img,recover_secret = net(test_secrets,test_covers,train=False) # to be [-1,1]
        acc_num, mse_s, mse_c,psnr,ssim = facevalmetric(recover_secret, mix_img, test_secrets,test_covers)
        acc_num_total += acc_num
        mse_ss.append(mse_s)
        mse_cs.append(mse_c)
        psnrs.append(psnr)
        ssims.append(ssim)

        #toshow = torch.cat((train_secrets[:4],train_covers[:4],(train_output[:4]+1)*0.5,(train_hidden[:4]+1)*0.5),dim=0)
        toshow = torch.cat((test_secrets[:4]*_std_torch+_mean_torch,test_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}_test.png'.format(outputname,idx),normalize=False)

        print("acc:{:.4f}".format(acc_num_total/valcnt)) 
        print("average mse_ss:{:.4f}".format(np.mean(mse_ss)))
        print("average mse_cs:{:.4f}".format(np.mean(mse_cs)))
        print("average psnrs:{:.4f}".format(np.mean(psnrs))) 
        print("average ssims:{:.4f}".format(np.mean(ssims)))    
        
            
    print("finished.")

def getmeanbit():
    img_all = torch.zeros((batch_size,3,face_size,face_size))
    for idx, train_batch in enumerate(train_loader):    
        imgs,labels = train_batch
        #imgs = imgs.to(device)
        imgs = torch.round((imgs*_std_torch.cpu()+_mean_torch.cpu())*255.)
        img_all += imgs
    #img_train_mean = torch.sum(img_all,dim=0)/((idx+1)*batch_size)
    for idx_, train_batch_ in enumerate(test_loader):    
        imgs,labels = train_batch_
        #imgs = imgs.to(device)
        imgs = torch.round((imgs*_std_torch.cpu()+_mean_torch.cpu())*255.)
        img_all += imgs

    for idx__, train_batch__ in enumerate(val_loader):    
        imgs,labels = train_batch__
        #imgs = imgs.to(device)
        imgs = torch.round((imgs*_std_torch.cpu()+_mean_torch.cpu())*255.)
        img_all += imgs

    img_vggface2_mean = torch.sum(img_all,dim=0)/((idx_+1+idx+1+idx__+1)*batch_size)
   
    np.save('img_vggface2_train_mean.npy',img_vggface2_mean)
    print("finished.")


#torch.bitwise_and
def getlsb(secret,cover):
    #secret,cover : uint8 numpy array
    svhnmean = np.load('img_vggface2_trainfiles_noval_noresize_se179008_mean.npy')
    
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
    outputname='lsb_vggface2'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    acctotal,total,cover_succ = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    for idx, test_batch in enumerate(test_loader):
        
        test_secrets, labels  = test_batch
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        total += len(test_secrets)
        test_covers,succ = getAdvZ(face_size,labels,batch_size)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/((idx+1)*batch_size)))
        test_secrets_integer = np.asarray(np.round(((test_secrets*_std_torch+_mean_torch)*255.).cpu().numpy()),dtype=np.uint8)
        test_covers_integer = np.asarray(np.round(test_covers.cpu().numpy()*255.),dtype=np.uint8)
        mix_img,recover_secret = getlsb(test_secrets_integer,test_covers_integer)
        recover_secret = torch.Tensor(recover_secret/255.).to(device)
        mix_img = torch.Tensor(mix_img/255.).to(device)
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error=valmetric(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
        acctotal += int(acc_num)
        l2loss_secrets += float(l2loss_secret)
        l2loss_covers += float(l2loss_cover)
        psnr_secrets += float(psnr_secret)
        ssim_secrets += float(ssim_secret)
        mean_pixel_errors += float(mean_pixel_error)

        print("attack success rate:{}/{}={:.6f}".format(acctotal,total,acctotal/total))
        print("avg. l2loss_secrets:{:.6f}".format(l2loss_secrets/total))
        print("avg. l2loss_covers:{:.6f}".format(l2loss_covers/total))
        print("avg. psnr_secrets:{:.6f}".format(psnr_secrets/total))
        print("avg. ssim_secrets:{:.6f}".format(ssim_secrets/total))
        print("avg. mean_pixel_errors:{:.6f}".format(mean_pixel_errors/total))

        toshow = torch.cat((test_secrets[:4]*_std_torch+_mean_torch,test_covers[:4],mix_img[:4],recover_secret[:4]),dim=0)
        imgg = make_grid(toshow,nrow=nrow_)
        save_image(imgg,'{}/{}.png'.format(outputname,idx),normalize=False)

    print("finished.")

def testModelAcc():
    
    total = 0
    acc_num_total=0
    for idx, test_batch in enumerate(test_loader):

        test_secrets, labels  = test_batch
        if total>=500:
            break
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)
        total += len(test_secrets)
        outputs = target_model((test_secrets+1)/2)
        pre = torch.argmax(outputs,dim=1)
        acc_num = len(torch.where(pre==labels)[0])   
        acc_num_total += acc_num 
    print("acc:{}".format(acc_num_total/total)) # [0,1]

testModelAcc()
#testLsb()
#getmeanbit()
test()
#testOpenset()
# #testVer()
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

import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from dataset import getCifar10
import PerceptualSimilarity.models
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch

from torchvision.utils import make_grid,save_image
import juncw

import models.CifarResnetModel as CifarResnetModel
from RIC import Net


def setSeed(seed):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size        
        torch.backends.cudnn.deterministic = True
 


def load_checkpoint(filepath):
    if os.path.isfile(filepath):
        print("=> loading checkpoint '{}'".format(filepath))
        checkpoint = torch.load(filepath)

        start_epoch = checkpoint['epoch']
        
        net.load_state_dict(checkpoint['state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint (epoch {})"
                .format(checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filepath))



def getAdvZ(img_size,labels,minibs,train=False):

    c=10
    k=5

    st=150
    succ = False
    print("getadvz c:{},k:{},st:{}".format(c,k,st))

    z0 = torch.rand((minibs,3,img_size,img_size))
   
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
    adv = cwatt(z0,labels)
    del cwatt
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)
    succ = len(torch.where(pre==labels)[0])
    return adv,succ


def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img



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


def test():

    MODELS_PATH = '../Weights/VGGFace2.pth.tar'

    net.eval()
    load_checkpoint(MODELS_PATH)
    outputname="../output/cifar10_byvggface2"
    
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
        mix_img,recover_secret = net(test_secrets,test_covers,train=False)
        
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,mean_pixel_error,\
            lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc=\
                valmetric(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
                
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

    print("finished.")    

if __name__ == "__main__":

    
    cwd = os.getcwd()
    
    batch_size = 10

    

    nrow_ = 4
    
    beta = 10
    imgsize=32

    cuda = torch.cuda.is_available()
    setSeed(1337)
    # TODO: Define train, validation and models
    
    device = torch.device("cuda:0" if cuda else "cpu")
    target_model = CifarResnetModel.ResNet18()
    cp = torch.load('../Weights/cifar10_resnet18.pth')

    target_model= torch.nn.DataParallel(target_model).cuda()
    target_model.load_state_dict(cp['net'])
    target_model.eval()
    
    kappa = 5

    modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])
    
    net = Net()
    net = torch.nn.DataParallel(net).to(device)
  
    test_loader  = getCifar10('../data/cifar-10_data/',train=False,batchsize=batch_size,resize=False,isshuffle=False)

    _mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
    _std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)

    test()

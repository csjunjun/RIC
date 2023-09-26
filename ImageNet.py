import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid,save_image
import juncw
import pandas as pd
from datetime import datetime
from ImageNetModels import Resnet50_Imagenet
from RIC import Net
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import PerceptualSimilarity.models

def setSeed(seed):
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True # enable if all images are same size        
        torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False    #现实的test阶段需要实现相同的attack的时候设置为false,否则model输出会稍有不同



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
#id_label_map = get_id_label_map(VGGface2_basepath+'label/identity_meta2.csv')
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
#VGGFace2_test_data = VGGFace2(txt_path=VGGface2_basepath+'test_list.txt',img_dir = VGGface2_basepath+'test/',transform=train_transform)

# Creates test set


def getAdvZ(img_size,labels,batch_size,train=False):
    
    c=10
    k=10
  
    st = 250
    succ = False
    print("getadvz c:{} k:{} st:{}".format(c,k,st))
   
    z0 = torch.rand((batch_size,3,img_size,img_size))
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels)
    adv = cwatt(z0,labels)
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)

    succ = len(torch.where(pre==labels)[0])
    return adv,succ



def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

def test():

    MODELS_PATH = '../Weights/VGGFace2.pth.tar'
    net.eval()
    load_checkpoint(MODELS_PATH)

    outputname="output/imagenet_byvggface2"
   
    if not os.path.exists(outputname):
        os.mkdir(outputname)

    acctotal,total,cover_succ = 0,0,0
    l2loss_secrets,l2loss_covers,psnr_secrets,ssim_secrets,mean_pixel_errors=0.,0.,0.,0.,0.
    lpips_errors,lpips_scs,mse_scs,psnr_scs,ssim_scs = 0.,0.,0.,0.,0.

    total_all = 0
    for idx, test_batch in enumerate(test_loader):
        if total>=500:
            break
        test_secrets, labels  = test_batch
        total_all += len(test_secrets)
        test_secrets = test_secrets.to(device)
        labels= labels.to(device)

        outputs = target_model((test_secrets+1)/2)
        pre = torch.argmax(outputs,dim=1)
        test_secrets = test_secrets[torch.where(pre==labels)]
        labels = labels[torch.where(pre==labels)]
        if len(test_secrets) == 0:
            continue
        total += len(test_secrets)
        test_covers,succ = getAdvZ(imgsize,labels,len(labels),train=False)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/((idx+1)*len(labels))))
        
        mix_img,recover_secret = net(test_secrets,test_covers,train=False)
        
        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,\
            mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc\
                =valmetric(recover_secret,mix_img,test_secrets,test_covers,beta,labels)
                
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
        print("attack success rate in correct images:{}/{}={:.6f}".format(acctotal,total_all,acctotal/total_all))
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


    batch_size = 5

    nrow_ = 4

    imgsize=224
    # Mean and std deviation of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
    cuda = torch.cuda.is_available()

    setSeed(1337)

    TEST_PATH ='/data/junliu/ImageNet_val'
    device = torch.device("cuda:0" if cuda else "cpu")
    target_model = Resnet50_Imagenet()
    #target_model = InceptionV3_Imagenet()
    kappa = 5

    modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    net = Net()

    net = torch.nn.DataParallel(net).to(device)
  
    test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
            TEST_PATH, 
            transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
            ])), batch_size=batch_size, num_workers=1, 
            pin_memory=True, shuffle=True, drop_last=False)

    _mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
    _std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)

    test()


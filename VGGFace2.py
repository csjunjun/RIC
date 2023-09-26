# Imports necessary libraries and modules
#from itertools import islice
#JunDeepSteganAdvEnDeFaceSE.py

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import make_grid,save_image
from facenet_pytorch import MTCNN, InceptionResnetV1
import juncw
import shutil
from RIC import Net
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import PerceptualSimilarity.models
from dataset import VGGFace2

def setSeed(seed):
    np.random.seed(seed) 
    torch.manual_seed(seed)  
    if cuda:
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False         #为了保证相同Key时生成的NAE一致，test阶段必须设置为false
        torch.backends.cudnn.deterministic = True

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
    
    #loss_all =   B*loss_secret  + loss_cover + classloss
    loss_all = B*loss_secret+classloss
    #loss_all = -loss_cover + B * loss_secret + classloss + loss_cover2

    
    return loss_all, loss_secret,classloss,loss_cover



def valmetric(S_prime, mix_img, S,C,  B,labels):
    ''' Calculates loss specified on the paper.'''
    outputs = target_model(mix_img)
    pre = torch.argmax(outputs,dim=1)
    acc_num = len(torch.where(pre==labels)[0])    
    psnr, ssim, mse_s,mse_c,mean_pixel_error = [], [], [],[],[]
    norm_S =  convert1(S*_std_torch+_mean_torch)
    norm_S_prime = convert1(S_prime)
    norm_mix_image = convert1(mix_img)
    mse_sc,psnr_sc,ssim_sc,lpips_sc,lpips_error = [], [], [],[],[]
    for i in range(len(S_prime)):
     
        mse_s.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-S_prime[i])))
        mse_c.append(float(torch.norm(C[i]-mix_img[i])))
        psnr.append(peak_signal_noise_ratio(norm_S[i],norm_S_prime[i],data_range=255))
        ssim.append(structural_similarity(norm_S[i],norm_S_prime[i],win_size=11, data_range=255.0, multichannel=True))
        mean_pixel_error.append(float(torch.sum(torch.abs(torch.round((S*_std_torch+_mean_torch)[i]*255)-torch.round(S_prime[i]*255)))/(3*face_size*face_size)))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], S_prime[i],normalize=True)
        lpips_error.append(float(tmp))
        #mix_image and secret image
        mse_sc.append(float(torch.norm((S*_std_torch+_mean_torch)[i]-mix_img[i])))
        psnr_sc.append(peak_signal_noise_ratio(norm_S[i],norm_mix_image[i],data_range=255))
        ssim_sc.append(structural_similarity(norm_S[i],norm_mix_image[i],win_size=11, data_range=255.0, multichannel=True))
        tmp = modellp.forward((S*_std_torch+_mean_torch)[i], mix_img[i],normalize=True)
        lpips_sc.append(float(tmp))
    return acc_num, np.sum(mse_s), np.sum(mse_c),np.sum(psnr),np.sum(ssim),np.sum(mean_pixel_error),\
        np.sum(lpips_error),np.sum(lpips_sc),np.sum(mse_sc),np.sum(psnr_sc),np.sum(ssim_sc)



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





def getAdvZ(img_size,labels,batch_size,train=False,seed=None):
    c=10
    k=10

    succ = False

    st = 300
    z0 = torch.rand((batch_size,3,img_size,img_size))
    cwatt = juncw.CW(target_model,c=c,kappa=k,steps=st,targeted=True,target_labels=labels,seed=seed,noise_type='uniform')
    adv = cwatt(z0,labels)
    outputs = target_model(adv)
    _,pre = torch.max(outputs,1)

    succ = len(torch.where(pre==labels)[0])
    return adv,succ


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
    
    train_loss_secret_history = []
    attloss_history = []
    loss_cover_history = []  

    outputname ='output/vggface2'
    if not os.path.exists(outputname):
        os.mkdir(outputname)
    for epoch in range(num_epochs):
        train_losses = []
        # Train one epoch
        cover_succ = 0
        for idx, train_batch in enumerate(train_loader):         
            net.train()
                      
            train_secrets, labels  = train_batch
            train_secrets = train_secrets[torch.where(labels<8631)]
            labels = labels[torch.where(labels<8631)]
            if len(labels)<1:
                continue
            train_secrets = train_secrets.to(device)

            labels = labels.to(device)
            train_covers,succ = getAdvZ(face_size,labels,len(train_secrets),train=True)
            cover_succ+= int(succ)
            print("epo:{} att z succ rate:{}".format(epoch,cover_succ/((idx+1)*batch_size)))
         
            train_secrets = Variable(train_secrets, requires_grad=False)
            train_covers = Variable(train_covers, requires_grad=False)


            optimizer.zero_grad()
            mix_img,recover_secret = net(train_secrets,train_covers) # to be [-1,1]
          

            train_loss, train_loss_secret,attloss,loss_cover = attack1_loss(recover_secret,mix_img, train_secrets,train_covers,beta,labels)
          
            if (idx+1)%print_freq == 0:
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
            print('Training: Batch {0}/{1}. Loss of {2:.4f},secret loss of {3:.4f}, attack1_loss of {4:.4f}, loss_cover {5:.5f}'\
                  .format(idx+1, len(train_loader), train_loss.data,  train_loss_secret.data,attloss.data,loss_cover))
    
            # val 
            if (idx+1)%save_freq == 0 or train_num-idx*(batch_size+1)<=batch_size or idx==1:
                #val_loss, val_loss_secret,val_attloss,val_loss_cover = val(outputname,epoch)
                #scheduler.step(val_loss_secret)
                is_best = False
                
            
                modelsavepath =outputname+'/Epoch_{}_{}_{}'.format(epoch+1,idx,outputname)
                save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'optimizer' : optimizer.state_dict(),
            #'scheduler':scheduler.state_dict()
        },filename=modelsavepath,is_best=is_best)
               
             
        mean_train_loss = np.mean(train_losses)
    
        # Prints epoch average loss
        print ('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
                epoch+1, num_epochs, mean_train_loss))
        plotloss(train_losses,train_loss_secret_history,attloss_history,loss_cover_history,outputname,epoch) 

    
    return net,mean_train_loss

    
def save_checkpoint(state, filename,is_best=False):
    checkpointname = filename+'_checkpoint.pth.tar'
    torch.save(state, checkpointname)
    if is_best:
        shutil.copyfile(checkpointname, filename+'_model_best.pth.tar')

def convert1(img):
    img = img * 255.0
    img = img.permute(0, 2, 3, 1).cpu().detach().numpy()
    return img

def test():
    MODELS_PATH = '../Weights/VGGFace2.pth.tar'
 
    
    net.eval()
    load_checkpoint(MODELS_PATH)
    outputname="vggface2_test"
   
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
   
        test_covers,succ = getAdvZ(face_size,labels,test_batch_size,train=False)
        cover_succ+= int(succ)
        print("att z succ rate:{}".format(cover_succ/total))
        
        mix_img,recover_secret = net(test_secrets,test_covers,train=False)

        acc_num, l2loss_secret,l2loss_cover,psnr_secret,ssim_secret,\
            mean_pixel_error,lpips_error,lpips_sc,mse_sc,psnr_sc,ssim_sc=\
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

    stage = "test" # or train

    # Hyper Parameters
    print_freq = 200
    save_freq = 200
    #num_epochs = 10
    num_epochs = 100
    batch_size = 32
    test_batch_size=5
    face_size = 160
    nrow_ = 4
    learning_rate = 0.0001
    beta = 10
    seed = 1


    cuda = torch.cuda.is_available()
    #setSeed(1337)
    
    setSeed(seed)
    print(seed)


    VGGface2_basepath = '../data/vggface2/'
    device = torch.device("cuda:0" if cuda else "cpu")
    target_model =InceptionResnetV1(pretrained='vggface2').eval().to(device)
    target_model.classify = True

    kappa = 5
    modellp = PerceptualSimilarity.models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    net = Net()
    net = torch.nn.DataParallel(net).to(device)
    # Save optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5,mode='min')    
    
    VGGFace2_train_data = VGGFace2(img_dir = VGGface2_basepath+'train/',mode="train")
    train_num =VGGFace2_train_data.imgnum

    
    
    _mean_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)
    _std_torch = torch.tensor((0.5, 0.5, 0.5)).view(3,1,1).to(device)


    if stage == "test":
        VGGFace2_test_data = VGGFace2(img_dir = VGGface2_basepath+'train/',mode="test")
        test_loader = torch.utils.data.DataLoader(
                VGGFace2_test_data, batch_size=test_batch_size, num_workers=1, 
                pin_memory=True, shuffle=False, drop_last=False)
        test()
    elif stage == "train":
        train_loader = torch.utils.data.DataLoader(
        VGGFace2_train_data, batch_size=batch_size, num_workers=4, 
        pin_memory=True, shuffle=True, drop_last=True)

        net, mean_train_loss = train_model(train_loader, beta, learning_rate)

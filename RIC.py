import torch.nn as nn
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import os
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())

        self.se1 = SEModule(50,2)
        self.se2 = SEModule(50,2)
        self.se3 = SEModule(50,2)

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        # from torchvision.utils import save_image
        # for i in range(0,50):
        #     save_image(p4[:,i,:,:],"features/p4_{}.jpg".format(i))
        #     save_image(p5[:,i,:,:],"features/p5_{}.jpg".format(i))
        #     save_image(p6[:,i,:,:],"features/p6_{}.jpg".format(i))
        p4 = self.se1(p4)
        p5 = self.se2(p5)
        p6 = self.se3(p6)
        # for i in range(0,50):
        #     save_image(p4[:,i,:,:],"features/p4se_{}.jpg".format(i))
        #     save_image(p5[:,i,:,:],"features/p5se_{}.jpg".format(i))
        #     save_image(p6[:,i,:,:],"features/p6se_{}.jpg".format(i))
        out = torch.cat((p4, p5, p6), 1)
        return out
#在训练的时候，由于希望分类器分类正确，最后生成的mix image会逐渐学到secret image，所以这里可以先用一个inn将原始的secret image
#转换到特征空间，然后在decode的时候逆回去
# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self,res=True,lambda_net=0.8):
        super(HidingNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        self.initialH3 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialH4 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialH5 = nn.Sequential(
            nn.Conv2d(153, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))
        
    def forward(self, h,cover):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)
        if self.res:
            out = (1-self.lambda_net)*out + self.lambda_net*cover
        return out

# Reveal Network (2 conv layers)
class RevealNetwork(nn.Module):
    def __init__(self,res=True,lambda_net =0.8):
        super(RevealNetwork, self).__init__()
        self.res = res
        self.lambda_net = lambda_net
        if self.res==True:
            in_c = 3
        else:
            in_c = 6
        self.initialR3 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.initialR4 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.initialR5 = nn.Sequential(
            nn.Conv2d(in_c, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, r,cover):
        if self.res:
            r = (r - self.lambda_net*cover)/(1-self.lambda_net)
        else:
            r = torch.cat((r,cover),dim=1)
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)
        mid = torch.cat((r1, r2, r3), 1)
        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)
        mid2 = torch.cat((r4, r5, r6), 1)
        out = self.finalR(mid2)
        return out



# Join three networks in one module


#upsampler = Upsample(size=(imgsize, imgsize), align_corners=True, mode='bilinear')

class SEModule(nn.Module):
    def __init__(self, channels, reduction, concat=False):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


# Join three networks in one module
class Net(nn.Module):
    def __init__(self,resen=True,resde=True,lambda_net=0.8):
        super(Net, self).__init__()
        self.m1 = PrepNetwork()
        self.m2 = HidingNetwork(resen,lambda_net)
        self.m3 = RevealNetwork(resde,lambda_net)
        self.act = nn.Sigmoid()
    # def forward(self, secret, cover,labels,train=True):
    #     x_1 = self.m1(secret)
    #     mid = torch.cat((x_1, cover), 1)
    #     x_2 = self.m2(mid,cover)
        
    #     x_2 = self.quan(x_2,type='noise',train=train)
    #     #x_2 = torch.clamp(x_2,0,1)
    #     random_cover,succ = getAdvZ(224,labels,batch_size)

    #     x_3 = self.m3(x_2,random_cover) #训练的时候不要用clamp 或者 sigmoid
    #     #x_3 = torch.clamp(x_3,0,1)
    #     #x_3 = self.act(x_3)
    #     return x_2, x_3

    def forward(self, secret, cover,train=True,threat_type=-1,**args):
        x_1 = self.m1(secret)
        mid = torch.cat((x_1, cover), 1)
        x_2 = self.m2(mid,cover)
        
        
        if threat_type == 0:
            x_2 = self.getnoistadv(x_2,args['intensity'])
        elif threat_type ==1:
            x_2 = self.qf(x_2,args['intensity'],args['device'])
        else:
            x_2 = self.quan(x_2,type='noise',train=train)
        #elif threat_type == 1:
        #x_2 = self.getnoistadv(x_2)
        #cover = self.getnoistadv(cover)
        if 'testSensitivity' not in args  or args['testSensitivity']==False:
            x_3 = self.m3(x_2,cover) #训练的时候不要用clamp 或者 sigmoid
        elif args['testSensitivity']==True:
            #setSeed(aseed)
            #newcover = getAdvZNoSuc(160,labels,len(secret),train=False,seed=args['aseed'],isuniform=True)
            diff = torch.norm(args['newcover']-cover)
            #diff_z = torch.norm(z-z_.cuda())
            print("diff between newcover and cover:{}".format(diff))
            
            #print("diff z :{}",format(diff_z))
            x_3 = self.m3(x_2,args['newcover'])
            del cover,args
        else:
            print("error")
        if train==False:
            x_3 = torch.clamp(x_3,0,1)
        #x_3 = self.act(x_3)
        return x_2, x_3

    def qf(self,x,qfv,device):
        now = datetime.now()
        if qfv==100:
            return x
        xr = torch.clamp((x*255).round(),0,255)
        xr = np.array(xr.detach().cpu().numpy(),dtype=np.uint8)
        res =None
        timestamp = str(datetime.timestamp(datetime.now())).split(".")[0]+ str(datetime.timestamp(datetime.now())).split(".")[1]+str(np.random.randint(0,10000))
        for xi in range(xr.shape[0]):  
            filename =  'temp{}.jpg'.format(timestamp)    
            Image.fromarray(xr[xi].transpose(1,2,0)).save(filename,quality=int(qfv))
            x_ = torch.Tensor(np.array(Image.open(filename)).transpose(2,0,1)).to(device)
            cmdstr = "rm {}".format(filename)
            os.system(cmdstr)
            x_ =x_.unsqueeze(0)/255.
            if res is None:
                res = x_
            else:
                res = torch.cat((res,x_),dim=0)
        return res

    def getnoistadv(self,x,intensity):
        img_size = x.shape[2]
        batch_size = x.shape[0]
        #z0_noise = 0.1*torch.randn((batch_size,3,img_size,img_size)).to(device)
        z0_noise = x.data.new(x.size()).normal_(0,intensity)
        #z0_noise = 0.01*torch.rand((batch_size,3,img_size,img_size)).to(device)
        x = x+z0_noise
        x = torch.clamp(x,0,1)
        return x

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
        return output/255.
# Creates net object
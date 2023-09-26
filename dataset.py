from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN
import pandas as pd
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torch.utils.data import DataLoader

class VGGFace2(Dataset):
    def __init__(self,  img_dir='data.tar', mode='train'):
        """
        Initialize data set as a list of IDs corresponding to each item of data set

        :param img_dir: path to image files as a uncompressed tar archive
        :param txt_path: a text file containing names of all of images line by line
        :param transform: apply some transforms like cropping, rotating, etc on input image
        """

        if mode == "train":
            filepaths = np.load('../data/vggface2trainfiles_rightface_1650940217.npy',allow_pickle=True)
            self.train_num = len(filepaths)
        elif mode == "val":
            filepaths = np.load('../data/vggface2valfiles_rightface_1650940217.npy',allow_pickle=True)
        elif mode == "test":
            filepaths = np.load('../data/vggface2testfile_rightface_1650940217.npy',allow_pickle=True)
        self.img_names = filepaths
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                transforms.ToTensor()
                ])
        self.imgnum = len(filepaths)
        self.face_size = 160
        self.mtcnn =MTCNN(image_size=self.face_size)
        self.id_label_map = self.get_id_label_map('../data/vggface2/identity_meta2.csv')

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)


    def get_id_label_map(self,meta_file):
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

    def __getitem__(self, index):
        """
        Generate one item of data set.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        image = Image.open(self.img_dir+self.img_names[index])
        label = self.img_names[index].split('/')[0]
        
        # if self.transform is not None:
        #     image = self.transform(image)
        
        face = self.mtcnn(image)
        # if face is None:
        #     face = self.transform(image)
        if face is None or face.shape[1]!=self.face_size or face.shape[2] !=self.face_size:
            image = image.resize((self.face_size,self.face_size))
            face = (self.transform(image)-0.5)/0.5 # 此時face是0,1區間，要把它變得和mtcnn之後的-1,1一樣
        return face,self.id_label_map[label]
    

def getCifar10(data_dir,batchsize,train=True,resize=False,isshuffle=False,resize_v=32):
    #transform_train = transforms.Compose([transforms.Resize(256), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    #transform_test  = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    std = [0.5, 0.5, 0.5]
    mean = [0.5, 0.5, 0.5]
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
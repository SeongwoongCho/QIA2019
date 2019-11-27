from utils import *
from torch.utils import data
from torchvision import transforms
from PIL import Image
from matplotlib import cm

seed_everything(42)
label2emo={'hap':0,'ang':1,'dis':2,'fea':3,'sad':4,'neu':5,'sur':6}
n_mels = 128

def get_transform(n_mels,is_train):
    return transforms.Compose([
                transforms.RandomCrop((n_mels,224)) if is_train else transforms.CenterCrop((n_mels,224)),
                transforms.Grayscale(3),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4) if is_train else transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], 
                                     [0.229, 0.224, 0.225]),])

class Dataset(data.Dataset):
    def __init__(self,file_list,root_dir,label_smooth_weight =0,is_train=True,video = False):
        self.file_list = file_list
        self.root_dir=root_dir
        self.smooth_weight = label_smooth_weight
        self.is_train=is_train
        self.transform = get_transform(n_mels,is_train)
        self.video = video
        
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self,idx):
        ## label load and preprocess
        try:
            fileName = self.file_list[idx]
            file = fileName.split('.')[0]
            label = file.split('-')[-1] if not self.video else file.split('-')[-3]
            label = label2emo[label]
            label = to_onehot(label=label)

            label = (1-self.smooth_weight)*label + self.smooth_weight*(1-label)/6
            label = label.astype('float32')
        except:
            label = 0
        ## input load and preprocess
        yS = np.load(self.root_dir+fileName)
        yS = preprocess(yS)
        if self.is_train:
            yS = augment(yS)
        yS = Image.fromarray(np.uint8(cm.gist_earth(yS)*255))
        yS = self.transform(yS)

        return yS,label
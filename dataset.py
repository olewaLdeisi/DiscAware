import torchvision.transforms as tfs
import torch.utils.data as data
from PIL import Image

root = "./data"

class Dataloader(data.Dataset):
    def __init__(self,train = True,transform=None):
        if transform:
            self.transform = tfs.Compose(transform)
        else:
            self.transform = tfs.Compose([
                tfs.ToTensor()
            ])
        if train:
            filepath = 'data/train.txt'
        else:
            filepath = 'data/test.txt'
        self.data_list,self.label_list,self.mark_list = [],[],[]

        with open(filepath, 'r') as f:
            for line in f:
                img,mark = line.strip().split('@')
                self.data_list.append('data/images/'+img.split('.')[0] + '.jpg')
                self.label_list.append('data/labels/'+img)
                self.mark_list.append(int(mark))

    def __getitem__(self, item):
        img = self.data_list[item]
        label = self.label_list[item]
        img = Image.open(img)
        label = Image.open(label)
        img = self.transform(img)
        label = self.transform(label)
        mark = self.mark_list[item]
        return img,label,mark

    def __len__(self):
        return  len(self.data_list)
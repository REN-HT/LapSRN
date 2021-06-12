from torchvision import transforms as T
from torch.utils import data
from PIL import Image
import numpy as np
import os


# 完成下采样
def target_transform(target, crop_size, upcsale_factor):
    trans=T.Compose([
        T.Resize(crop_size//upcsale_factor, interpolation=T.InterpolationMode.BICUBIC)
    ])
    return trans(target)


# cut_blur数据增强
def Cut_blur(lr, hr, alpha=0.02):
    if lr.size!=hr.size:
        raise ValueError('img size have to be matching!')
    if alpha<=0:
        return lr, hr
    cut_ratio=np.random.rand()*0.1+alpha
    h, w=hr.size
    ch, cw=np.int(cut_ratio*h), np.int(cut_ratio*w)
    cy=np.random.randint(ch,h-ch)
    cx=np.random.randint(cw,w-cw)
    box=(cx-w//2,cy-h//2,cx+w//2,cy+h//2)

    if np.random.rand()>0.5:
        new_hr=Image.new('RGB', hr.size)
        new_hr.paste(hr)
        region=lr.crop(box)
        new_hr.paste(region, box)
        lr=new_hr
    else:
        region=hr.crop(box)
        lr.paste(region, box)

    return lr, hr


class DataSet(data.Dataset):
    def __init__(self, root):
        super(DataSet, self).__init__()
        # 获取指定路径下的所有图片名称
        self.img_names=os.listdir(root)
        # 结合路径和图片名生成图片路径
        self.img_paths=[os.path.join(root, name) for name in self.img_names]
        # 随机反转图片，旋转，以及转化为tensor类型
        self.transform = T.Compose([
            # T.CenterCrop((128, 128)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(90),
            T.ToTensor(),
            T.Normalize(mean=[.463], std=[.172]),
            # T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
        ])

    def __getitem__(self, index):
        # 打开某张图片并将其转化为YCbCr颜色空间
        img=Image.open(self.img_paths[index]).convert('YCbCr')
        # 分离出Y通道
        y, _, _ = img.split()
        y = self.transform(y)

        label4x = y
        # 使用双三次插值完成下采样
        label2x = target_transform(y, 128, 2)
        # input = target_transform(y, 128, 4)
        input = target_transform(label2x, 64, 2)
        # 返回网络需要输入的图片，以及两倍放大和四倍放大结果
        return input, label2x, label4x

    def __len__(self):
        return len(self.img_names)


from PIL import Image
from torchvision import transforms as T
from model.lapSR import LapSRNet
import torch


def target_transform(target, crop_size, upcsale_factor):
    trans=T.Compose([
        T.Resize(crop_size//upcsale_factor, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
    ])
    return trans(target)


def resize(img, h, w):
    trans = T.Resize((h, w), interpolation=T.InterpolationMode.BICUBIC)
    return trans(img)

# 测试图片
def test(path):
    img = Image.open(path).convert('YCbCr')
    y, b, r = img.split()
    input = target_transform(y, 128, 4).unsqueeze(0)
    model=LapSRNet()
    model_state_dic=torch.load('D:/Program/SR/lapsr/LapSR_model_weight_best.pth')
    model.load_state_dict(model_state_dic)
    out2x, out4x = model(input)
    out4x = out4x.squeeze(0)
    unloader=T.ToPILImage()
    out4x = unloader(out4x)
    imgout = Image.merge('YCbCr', (out4x, b, r))
    imgout.show()


def test11(path):
    img = Image.open(path).convert('YCbCr')
    w, h = img.size
    y, b, r = img.split()
    b = resize(b, h*2, w*2)
    r = resize(r, h*2, w*2)
    to_tensor = T.ToTensor()
    input = to_tensor(y).unsqueeze(0)
    model=LapSRNet()
    model_state_dic=torch.load('D:/Program/SR/lapsr/LapSR_model_weight_best.pth')
    model.load_state_dict(model_state_dic)
    out2x, out4x = model(input)
    out2x = out2x.squeeze(0)
    unloader=T.ToPILImage()
    out2x = unloader(out2x)
    imgout = Image.merge('YCbCr', (out2x, b, r))
    imgout.show()


def PSNR(root, scale):
    from PIL import Image
    from torchvision import transforms as T
    import math
    import os
    img_names=os.listdir(root)
    img_paths=[os.path.join(root, name) for name in img_names]
    to_tensor = T.ToTensor()
    net=LapSRNet()
    model_state_dic=torch.load('D:/Program/SR/lapsr/LapSR_model_weight_best.pth')
    net.load_state_dict(model_state_dic)
    res=0
    for path in img_paths:
        gt=Image.open(path).convert('YCbCr')
        h, w = gt.size
        y, _, _ = gt.split()
        h, w = (h//scale)*scale, (w//scale)*scale
        y = resize(y, h, w)
        input = resize(y, h//scale, w//scale)
        input = to_tensor(input).unsqueeze(0)
        lr2x, _ = net(input)
        lr2x = lr2x.squeeze(0)
        y = to_tensor(y)
        mse = math.sqrt(((lr2x*255.0-y*255.0)**2).mean())
        res+= 100 if mse==0 else 20*math.log10(255.0/mse)

    print('PSNR:', res/len(img_paths))


if __name__=='__main__':
    path='F:/dataset/SuperResolutionDataset/Test/Mix/T1.jpg'
    test11(path)



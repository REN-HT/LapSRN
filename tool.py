import cv2
import os
from PIL import Image
from torchvision import transforms as T

# 用于裁剪图片
def clip_image(root):
    img_names = os.listdir(root)
    img_paths=[os.path.join(root, name) for name in img_names]
    count=1
    for path in img_paths:
        im = cv2.imread(path)
        row, col, _ = im.shape
        for i in range(0, row, 50):
            for j in range(0, col, 50):
                if i+128 >= row or j+128 >= col:
                    continue
                imm = im[i:i+128, j:j+128]
                img_path='F:/dataset/SuperResolutionDataset/LapSRNDIV2K_train_HR'
                save_path=os.path.join(img_path, str(count)+'.jpg')
                cv2.imwrite(save_path, imm)
                count=count+1


# 计算训练集均值和标准差,单通道
def calculate_mean_std(root):
    img_names = os.listdir(root)
    to_tensor=T.ToTensor()
    img_paths=[os.path.join(root, name) for name in img_names]
    all_mean=0
    all_std=0
    for path in img_paths:
        img=Image.open(path).convert('YCbCr')
        y, _, _=img.split()
        y=to_tensor(y)
        all_mean+=y.mean()
        all_std+=y.std()
    print("mean:", all_mean/len(img_paths), "std:", all_std/len(img_paths))


# 利用双三次插值生成等尺度低分辨率图像
def generate_low_resolution(path):
    img=Image.open(path)
    count=2
    upsample=T.Resize(128, interpolation=T.InterpolationMode.BICUBIC)
    downsample=T.Resize(64,interpolation=T.InterpolationMode.BICUBIC)
    while count>0:
        img=downsample(img)
        img=upsample(img)
        count=count-1
    # img.save("F:/dataset/SuperResolutionDataset/newimg.jpg")



if __name__=='__main__':
    root='F:/dataset/SuperResolutionDataset/Train/DIV2K_train_HR/DIV2K_train_HR'
    # clip_image(root)
    # calculate_mean_std(root)
    generate_low_resolution('F:/dataset/SuperResolutionDataset/lapsrtrainset/1.jpg')




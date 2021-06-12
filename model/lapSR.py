from torch import nn
import torch
import numpy as np

# -----------------------------------------------------
# 写在前面：整个网络结构未使用到BN层，可初始化，使用反卷积上采样
# 网络会自动生成x2,x4两种SR图像，暂未使用递归结构，也没有参数共享
# -----------------------------------------------------

# 生成size*size大小滤波器的初始值
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv_block=nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        out = self.conv_block(x)
        # # 递归网络
        # for _ in range(recur_number):
        #     out=self.conv_block(out)
        #     #加上残差结构
        #     # out=out+x
        return out


class LapSRNet(nn.Module):
    def __init__(self):
        super(LapSRNet, self).__init__()
        self.conv_input = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # conv_F1为特征提取块
        self.conv_F1 = self.make_layer(Block)
        # conv_R1为预测conv_F1特征图的子带残差
        self.conv_R1 = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        # conv_I1由输入图像通过转置卷积得到上采样结果
        self.conv_I1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        self.conv_F2 = self.make_layer(Block)
        self.conv_R2 = nn.Conv2d(64, 1, 3, 1, 1, bias=False)
        self.conv_I2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight)
                # c1, c2, h, w = m.weight.data.size()
                # weight = get_upsample_filter(h)
                # m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out=self.relu(self.conv_input(x))

        res_f1=self.conv_F1(out)
        res_i1=self.conv_I1(x)
        res_r1=self.conv_R1(res_f1)
        # 得到放大2倍的SR
        res_hr2=res_i1+res_r1

        res_f2=self.conv_F2(res_f1)
        # 利用上一次的结果res_hr2
        res_i2=self.conv_I2(res_hr2)
        res_r2=self.conv_R2(res_f2)
        # 得到放大4倍的SR
        res_hr4=res_i2+res_r2

        # 放大2倍和4倍
        return res_hr2, res_hr4


# Charbonnier损失函数
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)
        return loss

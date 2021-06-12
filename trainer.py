import os
from config import opt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from dataset.dataSet import DataSet
from model.lapSR import LapSRNet, L1_Charbonnier_loss
from torch.autograd import Variable


def train():
    train_set = DataSet(opt.root)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=opt.batch_size,
                              num_workers=opt.num_workers, drop_last=True)
    net = LapSRNet()
    criterion = L1_Charbonnier_loss()
    if opt.cuda:
        net = net.cuda()
        criterion = criterion.cuda()
    net.train()
    # 选择优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, weight_decay=1e-4)

    for epoch in range(opt.epoch):
        train_loss=0
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input, target2x, target4x = Variable(data[0]), Variable(data[1]), Variable(data[2])
            if opt.cuda:
                input = input.cuda()
                target2x = target2x.cuda()
                target4x = target4x.cuda()

            HR2x, HR4x = net(input)
            loss2x = criterion(HR2x, target2x)
            loss4x = criterion(HR4x, target4x)
            loss = loss4x.item() + loss2x.item()
            train_loss+=loss
            print('{} epoch loss:{:.2f}'.format(epoch + 1, train_loss//(i+1)))
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss2x.backward(retain_graph=True)
            loss4x.backward()
            # 梯度更新
            optimizer.step()
        # 保存模型参数
        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), 'LapSR_model_weight{}.pth'.format((epoch+1) / 10))


if __name__=='__main__':
    train()








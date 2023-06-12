
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from Models import Deeplab_v3plus
from metrics import *
import cfg
from matplotlib import pyplot as plt
import numpy as np
import os


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)


model = Deeplab_v3plus.DeepLabv3_plus(n_InputChannels=3, n_classes=cfg.DATASET[1], os=16, _print=False)
model= model.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
test_flag = True

train_loss_list = []
evl_loss_list = []
class_acc_total = []
best = [0]
running_metrics_val = runningScore(cfg.DATASET[1])

def train(model):
    model.train()
    running_metrics_val.reset()
    train_loss = 0
    prec_time = datetime.now()

        # 训练批次
    for i, sample in enumerate(train_data):
        # 载入数据
        
        img_data = Variable(sample['img'].to(device))
        img_label = Variable(sample['label'].to(device))
        # 训练
        out = model(img_data)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, img_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # 评估
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = img_label.data.cpu().numpy()
        running_metrics_val.update(true_label, pre_label)
    
    train_loss_list.append(loss.item())
    metrics = running_metrics_val.get_scores()

    for k, v in metrics[0].items():
        print(k, v)

    global train_miou
    global train_list_PA
    global train_list_mIOU
    global class_acc

    train_miou = metrics[0]['mIou: ']
    class_acc = metrics[0]['class_acc: ']
    class_acc_total.append(class_acc)
    train_list_PA = metrics[2]
    train_list_mIOU = metrics[3]

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)


def evaluate(model):
    model.eval()
    with t.no_grad():
        eval_loss = 0
        prec_time = datetime.now()

        for j, sample in enumerate(val_data):
            valImg = Variable(sample['img'].to(device))
            valLabel = Variable(sample['label'].long().to(device))

            out = model(valImg)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, valLabel)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = valLabel.data.cpu().numpy()
            running_metrics_val.update(true_label, pre_label)

        evl_loss_list.append(loss.item())
        metrics = running_metrics_val.get_scores()
        for k, v in metrics[0].items():
            print(k, v)

        global evl_list_PA
        global evl_list_mIOU
        global class_acc

        class_acc = metrics[0]['class_acc: ']
        
        evl_list_PA = metrics[2]
        evl_list_mIOU = metrics[3]

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prec_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
        print(time_str)

def main():

    # if os.path.exists(log_dir):
    #     checkpoint = t.load(log_dir)
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     start_epoch = checkpoint['epoch']
    #     print('加载 epoch {} 成功! '.format(start_epoch))
    # else:
    #     start_epoch = 0
    #     print('无保存模型，将从头开始训练')

    for epoch in range( cfg.EPOCH_NUMBER):
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        train(model)
        if max(best) <= train_miou:
            best.append(train_miou)
            # state = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
            # log_dir = './Results/weights/FCN_weight/{}.pth'.format(epoch+1)
            t.save(model.state_dict(), './Results/weights/FCN_weight/{}.pth'.format(epoch+1))
        evaluate(model)
    class_list_10 = ['Mag', 'Oli', 'Px', 'Pn', 'Cp', 'Hbl', 'Bio', 'Pl', 'Po', 'Bg']
    class_list_3 = ['Mag', 'Cp', 'Po']
    epochs = [i+1 for i in range(cfg.EPOCH_NUMBER)] 
    plt.figure(figsize=(16,10))
    plt.subplot(311)
    plt.plot(epochs, train_loss_list, label = "Train Loss", color = 'b')
    plt.plot(epochs, evl_loss_list, label = "Eval Loss", color = 'r')
    plt.ylabel("Loss Val", fontsize=12)
    plt.legend()

    plt.subplot(312)
    plt.plot(epochs, np.array(train_list_PA[::2]) * 100, label = "Train PA", color = 'b')
    plt.plot(epochs, np.array(train_list_mIOU[::2])*100, label = "Train mIOU", color = 'b', linestyle = '-.')
    plt.plot(epochs, np.array(train_list_PA[1::2])*100, label = "Eval PA", color = 'r')
    plt.plot(epochs, np.array(train_list_mIOU[1::2])*100, label = "Eval mIOU", color = 'r', linestyle = '-.')
    plt.ylabel('Accuracy %', fontsize=12)
    plt.legend()

    plt.subplot(313)
    for i in range(len(class_list_3)):
        class_ = []
        for j in range(cfg.EPOCH_NUMBER):
            class_.append(class_acc_total[j][i])
        plt.plot(epochs,  np.array(class_)*100, label = class_list_3[i])
        plt.legend(loc=2)
    plt.ylabel('Class Accuracy %', fontsize=12)
    plt.xlabel("Epoch number", fontsize=12)  

    plt.savefig("./Fig/loss.svg", dpi=600, format = 'svg') 
    plt.show()

if __name__ == "__main__":
    main()



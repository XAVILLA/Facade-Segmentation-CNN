import os
import time
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import torch.nn.functional as F
from dataset import FacadeDataset

N_CLASS=5


def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.time()
    num = 0
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num += 1
    end = time.time()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss/num, end-start))
    return running_loss/num

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    return (losses/cnt)


def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i
            gt = labels.cpu().data.numpy().squeeze(0).astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding= 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 512, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.n_class, 5, padding=2),

        )

    def forward(self, x):
        x = self.layers(x)
        # print(x.shape)
        # print(x.shape)
        # x = self.trans(x)
        return x


def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            print(images.shape)
            output = net(images).cpu().numpy()
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))

    # print(losses / cnt)
    return aps

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    train_data = FacadeDataset(flag='train', data_range=(0,905), onehot=False)
    print("finish loading train&val")
    trainS, val = torch.utils.data.random_split(train_data, [724, 181])
    # train_data = FacadeDataset(flag='train', data_range=(0, 5), onehot=False)
    # print("finish loading train&val")
    # trainS, val = torch.utils.data.random_split(train_data, [4, 1])
    train_loader = DataLoader(trainS, batch_size=4)
    eval_loader = DataLoader(val, batch_size=4)


    test_data = FacadeDataset(flag='test_dev', data_range=(0,5), onehot=False)
    test_loader = DataLoader(test_data, batch_size=1)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,5), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)

    name = 'FacadeCNN'
    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    pre_train_acc = test(train_loader, net, criterion, device)
    train_acc.append(pre_train_acc)
    pre_val_acc = test(eval_loader, net, criterion, device)
    val_acc.append(pre_val_acc)

    print('\nStart training')
    for epoch in range(1):
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        tr_acc = train(train_loader, net, criterion, optimizer, device, epoch+1)
        train_acc.append(tr_acc)


        evaluation_loader = eval_loader
        v_acc = test(evaluation_loader, net, criterion, device)
        val_acc.append(v_acc)

    print('\nFinished Training, Testing on test set')
    testloss = test(test_loader, net, criterion, device)
    print("\nLoss calculated on test set = ", testloss)
    print('\nGenerating Unlabeled Result')
    get_result(test_loader, net, device, folder='output_test')

    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))

    cal_AP(ap_loader, net, criterion, device)


if __name__ == "__main__":
    train_acc = []
    val_acc = []
    main()

from optparse import OptionParser
from collections import defaultdict
import sys
import os
import time
import pandas as pd

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchnet.meter as meter

from resnet import resnet50
from dataset import Dataset
from resnetutils import AverageMeter, save_plot, save_auc

global best_net

def train(train_loader, net, criterion, optimizer, gpu, epoch):
    print('\nTrain_net...')
    start = time.time()
    net.train()
    losses = AverageMeter()
    total_correct = 0
    mtr = meter.AUCMeter()
    for i, (imgs, marks, labels) in enumerate(train_loader):
        if gpu:
            imgs = imgs.cuda()
            labels = labels.cuda()

        outputs = net(imgs)  # 网络输出
        # _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        losses.update(loss.item(), imgs.size(0))
        smax_probs = nn.Softmax(dim=1)(outputs)  # 每个类的概率
        mtr.add(smax_probs.data[:, 1], labels.cpu())  # 计算AUC

        pred = torch.max(smax_probs, 1)[1]  # 预测类别标签
        correct = pred.eq(labels.view_as(pred)).sum()
        total_correct += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            f'Train Epoch: {epoch + 1} [{(i + 1) * len(imgs)}/{len(train_loader.dataset)} ({100 * (i + 1) / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    end = time.time()

    accuracy = 100. * float(total_correct) / float(len(train_loader.dataset))
    mtr_values = mtr.value()
    print(
        f'Time:{int((end - start)//60)}:{int((end - start)%60)}\nTrain set: Average loss: {losses.avg:.4f}, Accuracy: {total_correct}/{len(train_loader.dataset)} ({accuracy:.2f}%), AUC: {mtr_values[0]:.4f}')

    return losses.avg, accuracy, mtr_values


def test(test_loader, net, criterion, gpu):
    print('\nTest_net...')
    start = time.time()
    net.eval()
    losses = AverageMeter()
    total_correct = 0
    mtr = meter.AUCMeter()

    with torch.no_grad():
        for i, (imgs, marks, labels) in enumerate(test_loader):
            if gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()

            outputs = net(imgs)  # 网络输出
            # _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), imgs.size(0))
            smax_probs = nn.Softmax(dim=1)(outputs)  # 每个类的概率
            mtr.add(smax_probs.data[:, 1], labels.cpu())  # 计算AUC

            pred = torch.max(smax_probs, 1)[1]  # 预测类别标签
            correct = pred.eq(labels.view_as(pred)).sum()
            total_correct += correct

    accuracy = 100. * float(total_correct) / float(len(test_loader.dataset))
    mtr_values = mtr.value()
    end = time.time()
    print(
        f'Time:{int((end - start)//60)}:{int((end - start)%60)}\nTest set: Average loss: {losses.avg:.4f}, Accuracy: {total_correct}/{len(test_loader.dataset)} ({accuracy:.2f}%), AUC: {mtr_values[0]:.4f}\n')

    return losses.avg, accuracy, mtr_values


def main(net, epochs=5, batch_size=1, lr=0.1, gpu=False, img_size=224):
    global best_net
    dir_checkpoint = 'resnetcheckpoints/'

    transform = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    train_set = Dataset(train=True, transform=transform)
    test_set = Dataset(train=False, transform=transform)
    print(f'''
        Starting training:
            Epochs: {epochs}
            Batch size: {batch_size}
            Learning rate: {lr}
            Training size: {len(train_set)}
            Validation size: {len(test_set)}
            CUDA: {str(gpu)}
        ''')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss()

    loss_dict = defaultdict(list)
    accuracy_dict = defaultdict(list)

    best_net = net
    best_epoch = 0
    _,best_acc,best_auc = test(test_loader, best_net, criterion, gpu)
    best_auc = best_auc[0]
    for epoch in range(epochs):
        train_loss, train_accuarcy, train_auc = train(train_loader, net, criterion, optimizer, gpu, epoch)
        test_loss, test_accuarcy, auc_values = test(test_loader, net, criterion, gpu)

        loss_dict['train loss'].append(train_loss)
        loss_dict['test loss'].append(test_loss)
        accuracy_dict['train accuracy'].append(float(train_accuarcy))
        accuracy_dict['test accuracy'].append(float(test_accuarcy))

        # 如果有更高的准确率或者更高的AUC,则保存模型
        if test_accuarcy > best_acc or (test_accuarcy == best_acc and best_auc > auc_values[0]):
            best_net, best_acc, best_auc = net, test_accuarcy, auc_values[0]
            best_epoch = epoch + 1

    now = time.strftime("%Y-%m-%d_%H:%M:%S_", time.localtime())
    torch.save(best_net.state_dict(), dir_checkpoint + now + f'resnet50_in_{best_epoch}.pth')
    save_auc(auc_values, filename=dir_checkpoint  + now + f'{best_epoch}_AUC{auc_values[0]}.png')
    csv_dict = {**loss_dict, **accuracy_dict}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv(dir_checkpoint + f'{now}log.csv')
    save_plot(loss_dict, save_path=dir_checkpoint + now)
    save_plot(accuracy_dict, save_path=dir_checkpoint + now)


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1, type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load', default=False, help='load file model')
    (options, args) = parser.parse_args()
    return options

'''
-e 100 -g
'''

if __name__ == '__main__':
    global best_net

    args = get_args()

    if args.load:
        net = resnet50(2, pretrained=False)
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    else:
        print("Loading pretrained resnet50...")
        net = resnet50(2, pretrained=True)

    if args.gpu:
        net.cuda()

    best_net = net

    try:
        main(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, gpu=args.gpu, img_size=224)
    except :
        if best_net != None:
            torch.save(best_net.state_dict(), 'resnetcheckpoints/INTERRUPTED.pth')
            print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
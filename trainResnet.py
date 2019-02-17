import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataloader
import torch.utils.data as Data
import torchvision.transforms as tfs
from torch.autograd import Variable
import torchnet.meter as meter
from collections import defaultdict
from resnetutils import AverageMeter, save_plot, save_auc

from resnet import resnet50

def train(train_loader, model, criterion, optimizer, epoch,batch_size=2,output_size=224):
    """Train the model

    Args:
        train_loader: The DataLoader of train dataset
        model: The Model to train
        criterion: to compute the margin loss
        optimizer: optimizer for model
        epoch: the current epoch to train

    Return:
        losses.avg: The average loss of train loss
    """
    model.train()
    losses = AverageMeter()
    total_correct = 0
    mtr = meter.AUCMeter()
    input_img = torch.Tensor(batch_size, 3, output_size, output_size)
    input_target = torch.LongTensor(batch_size)

    input_img, input_target = input_img.cuda(0), input_target.cuda(0)

    for batch_idx, (data,_,target) in enumerate(train_loader):
        data = input_img.copy_(data)
        target = input_target.copy_(target)
        data_var, target_var = Variable(data), Variable(target)

        probs = model(data_var)
        # _, pred = torch.max(nn.Softmax(dim=1)(output), 1)
        loss = criterion(probs, target_var)

        losses.update(loss.item(), data.size(0))
        smax_probs = F.softmax(probs, dim=1)
        mtr.add(smax_probs.data[:, 1], target.cpu())

        pred = probs.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        total_correct += correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100 * (batch_idx + 1) / len(train_loader), loss.item()))

    accuracy = 100. * float(total_correct) / float(len(train_loader.dataset))
    mtr_values = mtr.value()
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), AUC: {:.4f}'.format(
        losses.avg, total_correct, len(train_loader.dataset), accuracy, mtr_values[0]))

    return losses.avg, accuracy, mtr_values

def test(test_loader, model, criterion,batch_size=1,output_size=224):
    """Test the model

    Args:
        test_loader: The DataLoader of test dataset
        model: The model to test
        criterion: to compute the margin loss

    Return:
        losses.avg: The average loss of test
        accuracy: The accuracy of test, computed by total_correct / len(test_loader.datasets)
    """
    model.eval()
    losses = AverageMeter()
    total_correct = 0
    mtr = meter.AUCMeter()
    input_img = torch.Tensor(batch_size, 3, output_size, output_size)
    input_target = torch.LongTensor(batch_size)

    input_img, input_target = input_img.cuda(0), input_target.cuda(0)

    for data,_, target in test_loader:
        data = input_img.copy_(data)
        target = input_target.copy_(target)
        with torch.no_grad():
            data_var = Variable(data)
        target_var = Variable(target)

        probs = model(data_var)
        # _, pred = torch.max(nn.Softmax(dim=1)(output), 1)
        loss = criterion(probs, target_var)

        smax_probs = F.softmax(probs, dim=1)
        mtr.add(smax_probs.data[:, 1], target.cpu())

        pred = probs.data.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum()
        total_correct += correct
        losses.update(loss.item(), data.size(0))
    accuracy = 100. * float(total_correct )/ float(len(test_loader.dataset))
    mtr_values = mtr.value()
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), AUC: {:.4f}\n'.format(
        losses.avg, total_correct, len(test_loader.dataset), accuracy, mtr_values[0]))

    return losses.avg, accuracy, mtr_values

if __name__  == '__main__':
    transform = [tfs.Resize((224,224)),tfs.ToTensor()]
    train_set = Dataloader(transform=transform)
    test_set = Dataloader(train=False,transform=transform)
    train_loader = Data.DataLoader(train_set, batch_size=5, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)
    model = resnet50(pretrained=False)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0005)

    model.cuda(0)
    criterion.cuda(0)

    loss_dict = defaultdict(list)
    accuracy_dict = defaultdict(list)

    save_path = '.'
    for epoch in range(1, 51):
        train_loss, train_accuarcy, train_auc = train(train_loader, model, criterion, optimizer, epoch,batch_size=5)
        test_loss, test_accuarcy, auc_values = test(test_loader, model, criterion,batch_size=1)

        if test_accuarcy >= 83:
        # if ((test_accuarcy >= 84 and auc_values[0] >= 0.75) or auc_values[0] >= 0.868) or (test_accuarcy >= 80 and auc_values[0] >= 0.75):
            torch.save(model.state_dict(),
                       save_path + '/' + '{:03d}_model_acc{:.2f}_auc{:.4f}.pth'.format(epoch,  test_accuarcy, auc_values[0]))
            save_auc(auc_values, filename=save_path + '/' + '{:03d}_AUC{}.png'.format(epoch, auc_values[0]))

        loss_dict['train loss'].append(train_loss)
        loss_dict['test loss'].append(test_loss)
        accuracy_dict['train accuracy'].append(float(train_accuarcy))
        accuracy_dict['test accuracy'].append(float(test_accuarcy))

    csv_dict = {**loss_dict, **accuracy_dict}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv(save_path + '/' + '{}_log.csv'.format(200))
    save_plot(loss_dict,save_path=save_path)
    save_plot(accuracy_dict,save_path=save_path)
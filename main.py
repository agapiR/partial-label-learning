import os
import os.path
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import argparse
import numpy as np

from loss2 import partial_loss
from model_linear import Linearnet
from model_mlp import Mlp
from model_cnn import Cnn
from model_resnet import Resnet
from dataset_mnist import Mnist
from dataset_fmnist import FashionMnist
from dataset_kmnist import KuzushijiMnist
from dataset_cifar10 import Cifar10

import time

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--partial_type',
                    type=str,
                    help='binomial, pair',
                    default='binomial')
parser.add_argument('--partial_rate', type=float, default=0.1)
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--n_epoch', type=int, default=501)

parser.add_argument('--dataset',
                    type=str,
                    help='mnist, fashionmnist, kuzushijimnist or cifar10',
                    default='mnist')
parser.add_argument('--model',
                    type=str,
                    help='linear, mlp, cnn or resnet',
                    default='linear')

parser.add_argument('--weight_update_start', type=int, default=0)

parser.add_argument('--decay_step', type=int, default=500)
parser.add_argument('--decay_rate', type=float, default=1)

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--result_dir', type=str, default='results/')
parser.add_argument('--seed', type=int, default=10000)
parser.add_argument('--tradeoff', type=float, default=1)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

decay_coef = 1
epoch_coef = 500

# load dataset
if args.dataset == 'mnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = Mnist(root='/home/u2018000735/datasets/mnist/MNIST',
                          download=False,
                          train_or_not=True,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.1307, ), (0.3081, ))
                          ]),
                          partial_type=args.partial_type,
                          partial_rate=args.partial_rate)
    test_dataset = Mnist(root='/home/u2018000735/datasets/mnist/MNIST',
                         download=False,
                         train_or_not=False,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307, ), (0.3081, ))
                         ]),
                         partial_type=args.partial_type,
                         partial_rate=args.partial_rate)

if args.dataset == 'fashionmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = FashionMnist(
        root='/home/u2018000735/datasets/fashion_mnist/FashionMNIST',
        download=False,
        train_or_not=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)
    test_dataset = FashionMnist(
        root='/home/u2018000735/datasets/fashion_mnist/FashionMNIST',
        download=False,
        train_or_not=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]),
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)

if args.dataset == 'kmnist':
    num_features = 28 * 28
    num_classes = 10
    num_training = 60000
    train_dataset = KuzushijiMnist(
        root='/home/u2018000735/datasets/kmnist/KMNIST',
        download=False,
        train_or_not=True,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))]),
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)
    test_dataset = KuzushijiMnist(
        root='/home/u2018000735/datasets/kmnist/KMNIST',
        download=False,
        train_or_not=False,
        transform=transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, ), (0.5, ))]),
        partial_type=args.partial_type,
        partial_rate=args.partial_rate)

if args.dataset == 'cifar10':
    input_channels = 3
    num_classes = 10
    dropout_rate = 0.25
    num_training = 50000
    train_dataset = Cifar10(root='/home/u2018000735/datasets/cifar10/',
                            download=False,
                            train_or_not=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                     (0.2023, 0.1994, 0.2010)),
                            ]),
                            partial_type=args.partial_type,
                            partial_rate=args.partial_rate)
    test_dataset = Cifar10(root='/home/u2018000735/datasets/cifar10/',
                           download=False,
                           train_or_not=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                    (0.2023, 0.1994, 0.2010)),
                           ]),
                           partial_type=args.partial_type,
                           partial_rate=args.partial_rate)

learningrate = args.lr
lr_plan = [learningrate] * args.n_epoch
for i in range(0, args.n_epoch):
    lr_plan[i] = learningrate * args.decay_rate**(i // args.decay_step)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


# result dir
method = "ours0905"
save_dir = './' + args.result_dir
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(args.dataset, args.model,
                                           args.partial_type,
                                           args.partial_rate, args.lr,
                                           args.weight_decay, method, args.tradeoff, args.seed, args.decay_rate, args.decay_step,  decay_coef, epoch_coef)
save_file = os.path.join(save_dir, save_name)


# Evaluate the Model
def evaluate(loader, model):
    model.eval()
    correct = 0
    total = 0
    for images, _, _, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)
        output1 = model(images)
        output = F.softmax(output1, dim=1)
        _, pred = torch.max(output.data, 1)
        total += images.size(0)
        correct += (pred == labels).sum().item()
    acc = 100 * float(correct) / float(total)
    return acc


def main():
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batchsize,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batchsize,
                                              num_workers=args.num_workers,
                                              drop_last=False,
                                              shuffle=False)

    print('building model...')
    if args.model == 'linear':
        net = Linearnet(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'mlp':
        net = Mlp(n_inputs=num_features, n_outputs=num_classes)
    elif args.model == 'cnn':
        net = Cnn(input_channels=input_channels,
                  n_outputs=num_classes,
                  dropout_rate=dropout_rate)
    elif args.model == 'resnet':
        net = Resnet(depth=32, n_outputs=num_classes)
    net.to(device)
    print(net.parameters)

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=learningrate,
                                weight_decay=args.weight_decay,
                                momentum=0.9)

    test_acc = evaluate(test_loader, net)
    with open(save_file, 'a') as file:
        file.write("epoch,train_acc,test_acc\n")
        file.write("0,0,{}\n".format(test_acc))

    for epoch in range(1, args.n_epoch):
        print ('training..., epoch {}'.format(epoch))
        net.train()
        adjust_learning_rate(optimizer, epoch)

        for i, (images, weight, labels, trues,
                indexes) in enumerate(train_loader):
            images = Variable(images).to(device)
            weight = Variable(weight).to(device)
            labels = Variable(labels).to(device)
            trues = trues.to(device)
            output = net(images)

            loss, new_weight = partial_loss(output, labels, weight, trues, epoch, args.tradeoff, decay_coef, epoch_coef)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch > args.weight_update_start:
                for j, k in enumerate(indexes):
                    train_loader.dataset.weight[k, :] = new_weight[
                        j, :].detach()

        print('evaluating model...')
        train_acc = evaluate(train_loader, net)
        test_acc = evaluate(test_loader, net)
        print(loss,train_acc, test_acc)

        with open(save_file, 'a') as file:
            file.write("{},{},{},{}\n".format(epoch, loss, train_acc, test_acc))


if __name__ == '__main__':
    main()

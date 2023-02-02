import os
import numpy as np
import random
import torch
import torch.utils.data
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from scipy.io import loadmat
from scipy.special import comb
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


import re

from utils.gen_index_dataset import gen_index_dataset


class RealDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert X.shape[0] == y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (X, y)


class RealIdxDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert X.shape[0] == y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (X, y, idx)


def generate_real_dataloader(dataname, datadir, batch_size, seed):
    datapath = os.path.join(datadir, "{}.mat".format(dataname))
    dt = loadmat(datapath)

    X = dt['features']
    partial_y = dt['p_labels']
    y = dt['logitlabels']

    X = np.float32(X)
    partial_y = np.float32(partial_y)
    y = np.float32(np.argmax(y, axis=1))
    

    print("random_state is {}".format(seed))
    train_X, test_X, train_y, test_y, train_partial_y, test_partial_y = train_test_split(X,
                                                                                        y,
                                                                                        partial_y,
                                                                                        train_size=0.8,
                                                                                        test_size=0.2,
                                                                                        stratify=y,
                                                                                        random_state=seed)
    train_X, valid_X, train_y, valid_y, train_partial_y, valid_partial_y = train_test_split(train_X,
                                                                                            train_y,
                                                                                            train_partial_y,
                                                                                            train_size=7 / 8,
                                                                                            test_size=1 / 8,
                                                                                            stratify=train_y,
                                                                                            random_state=seed)

    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)

    print(train_X.shape[0], valid_X.shape[0], test_X.shape[0])

    ordinary_train_dataset = RealDataset(train_X, train_y)
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_train_dataset,
        batch_size=len(ordinary_train_dataset),
        shuffle=False,
        num_workers=0)

    # train_dataset = RealIdxDataset(train_X, train_partial_y)
    train_dataset = gen_index_dataset(train_X, train_partial_y, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)

    ordinary_valid_dataset = RealDataset(valid_X, valid_y)
    valid_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_valid_dataset,
        batch_size=len(ordinary_valid_dataset),
        shuffle=False,
        num_workers=0)

    ordinary_test_dataset = RealDataset(test_X, test_y)
    test_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_test_dataset,
        batch_size=len(ordinary_test_dataset),
        shuffle=False,
        num_workers=0)

    num_features = X.shape[1]
    num_classes = partial_y.shape[1]

    return (train_loader, train_eval_loader, valid_eval_loader,
            test_eval_loader, train_partial_y, num_features, num_classes)



def generate_uniform_cv_candidate_labels(dataname, train_labels, partial_type):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    if partial_type == "01":
        assert K == 10    
        transition_matrix = [
        [1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0.5, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0.5, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0.5, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0.5, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0.5, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0.5, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0.5, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.5],
        [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    elif partial_type == "02":
        assert K == 10    
        q_adj = 0.3
        transition_matrix = [
        [1, q_adj, 0, 0, 0, 0, 0, 0, 0, q_adj],
        [q_adj, 1, q_adj, 0, 0, 0, 0, 0, 0, 0],
        [0, q_adj, 1, q_adj, 0, 0, 0, 0, 0, 0],
        [0, 0, q_adj, 1, q_adj, 0, 0, 0, 0, 0],
        [0, 0, 0, q_adj, 1, q_adj, 0, 0, 0, 0],
        [0, 0, 0, 0, q_adj, 1, q_adj, 0, 0, 0],
        [0, 0, 0, 0, 0, q_adj, 1, q_adj, 0, 0],
        [0, 0, 0, 0, 0, 0, q_adj, 1, q_adj, 0],
        [0, 0, 0, 0, 0, 0, 0, q_adj, 1, q_adj],
        [q_adj, 0, 0, 0, 0, 0, 0, 0, q_adj, 1],
        ]      
    elif partial_type == "03":
        assert K == 10    
        p_1, p_2, p_3, p_4 = 0.2, 0.8, 0.4, 0.2
        transition_matrix = [
        [1,   p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1],
        [p_1,   1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1],
        [p_1, p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4],
        [p_4, p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3],
        [p_3, p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3],
        [p_3, p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2],
        [p_2, p_3,  p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2],
        [p_2, p_2,  p_3,  p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2],
        [p_2, p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1,  1,    p_1],
        [p_1, p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1,    1],
        ]
    elif partial_type == "04":
        assert K == 10    
        q_1, q_2, q_3  = 0.5, 0.3, 0.1
        transition_matrix =  [
        [1, q_1, q_2, q_3, 0, 0, 0, q_3, q_2, q_1],
        [q_1, 1, q_1, q_2, q_3, 0, 0, 0, q_3, q_2],
        [q_2, q_1, 1, q_1, q_2, q_3, 0, 0, 0, q_3],
        [q_3, q_2, q_1, 1, q_1, q_2, q_3, 0, 0, 0],
        [0, q_3, q_2, q_1, 1, q_1, q_2, q_3, 0, 0],
        [0, 0, q_3, q_2, q_1, 1, q_1, q_2, q_3, 0],
        [0, 0, 0, q_3, q_2, q_1, 1, q_1, q_2, q_3],
        [q_3, 0, 0, 0, q_3, q_2, q_1, 1, q_1, q_2],
        [q_2, q_3, 0, 0, 0, q_3, q_2, q_1, 1, q_1],
        [q_1, q_2, q_3, 0, 0, 0, q_3, q_2, q_1, 1],
        ]   
    elif partial_type == "10":
        assert K == 10    
        p_1, p_2, p_3, p_4 = 0.9, 0.8, 0.7, 0.6
        transition_matrix = [
        [1,   p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1],
        [p_1,   1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1],
        [p_1, p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3,  p_4],
        [p_4, p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3,  p_3],
        [p_3, p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2,  p_3],
        [p_3, p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2,  p_2],
        [p_2, p_3,  p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2,  p_2],
        [p_2, p_2,  p_3,  p_3,  p_4,  p_1,  p_1,    1,  p_1,  p_2],
        [p_2, p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1,  1,    p_1],
        [p_1, p_2,  p_2,  p_2,  p_3,  p_3,  p_4,  p_1,  p_1,    1],
        ]
    else:
        match = re.match("uniform_(.*)", partial_type)
        if match:
            p = float(match.groups()[0])
            transition_matrix = np.ones((K,K)) * p
            for i in range(K):
                transition_matrix[i,i] = 1.0
        else:
            match = re.match("random_(.*)", partial_type)
            if match:
                p = float(match.groups()[0])
                transition_matrix = np.random.rand(K,K) * p
                for i in range(10):
                    transition_matrix[i,i] = 1.0
            else:
                match = re.match("huniform_(.*)", partial_type)
                if match:
                    assert dataname == "cifar100", dataname
                    p = float(match.groups()[0])
                    transition_matrix = np.ones((K,K)) * p
                    for i in range(K):
                        transition_matrix[i,i] = 1.0
                    labels = np.arange(K)
                    coarse_labels = cifar100_sparse2coarse(labels)
                    different_coarse_label = np.expand_dims(coarse_labels,0) != np.expand_dims(coarse_labels,1)
                    transition_matrix[different_coarse_label] = 0.0                
                
                
    transition_matrix = np.array(transition_matrix)

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def prepare_cv_datasets(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='~/datasets/mnist',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
        test_dataset = dsets.MNIST(root='~/datasets/mnist',
                                   train=False,
                                   transform=transforms.ToTensor())
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='~/datasets/kmnist',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)
        test_dataset = dsets.KMNIST(root='~/datasets/kmnist',
                                    train=False,
                                    transform=transforms.ToTensor())
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(
            root='~/datasets/fashion_mnist',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
        test_dataset = dsets.FashionMNIST(root='~/datasets/fashion_mnist',
                                          train=False,
                                          transform=transforms.ToTensor())
    elif dataname == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(
            ),  # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        ordinary_train_dataset = dsets.CIFAR10(
            root='~/datasets/cifar10',
            train=True,
            transform=train_transform,
            download=True)
        test_dataset = dsets.CIFAR10(root='~/datasets/cifar10',
                                     train=False,
                                     transform=test_transform)
    elif dataname == 'cifar100':
        train_transform = transforms.Compose([
            transforms.ToTensor(
            ),  # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        ordinary_train_dataset = dsets.CIFAR100(
            root='~/datasets/cifar100',
            train=True,
            transform=train_transform,
            download=True)
        test_dataset = dsets.CIFAR100(root='~/datasets/cifar100',
                                     train=False,
                                     transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=ordinary_train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=0)
    full_train_loader = torch.utils.data.DataLoader(
        dataset=ordinary_train_dataset,
        batch_size=len(ordinary_train_dataset.data),
        shuffle=True,
        num_workers=0)
    if dataname == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10
    return (full_train_loader, train_loader, test_loader,
            ordinary_train_dataset, test_dataset, num_classes)


def prepare_cv_datasets_hyper(dataname, batch_size):
    if dataname == 'mnist':
        ordinary_train_dataset = dsets.MNIST(root='~/datasets/mnist',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
    elif dataname == 'kmnist':
        ordinary_train_dataset = dsets.KMNIST(root='~/datasets/kmnist',
                                              train=True,
                                              transform=transforms.ToTensor(),
                                              download=True)
    elif dataname == 'fashion':
        ordinary_train_dataset = dsets.FashionMNIST(
            root='~/datasets/fashion_mnist',
            train=True,
            transform=transforms.ToTensor(),
            download=True)
    elif dataname == 'cifar10':
        train_transform = transforms.Compose([
            transforms.ToTensor(
            ),  # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        ordinary_train_dataset = dsets.CIFAR10(root='~/datasets/cifar10',
                                               train=True,
                                               transform=train_transform,
                                               download=True)
    elif dataname == 'cifar100':
        train_transform = transforms.Compose([
            transforms.ToTensor(
            ),  # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.247, 0.243, 0.261))
        ])
        ordinary_train_dataset = dsets.CIFAR100(root='~/datasets/cifar100',
                                               train=True,
                                               transform=train_transform,
                                               download=True)

    dataset_size = len(ordinary_train_dataset)
    valid_proportion = 0.1
    valid_size = int(np.floor(valid_proportion * dataset_size))
    train_size = dataset_size - valid_size

    trainingset, validationset = torch.utils.data.random_split(
        ordinary_train_dataset, [train_size, valid_size])

    train_loader = torch.utils.data.DataLoader(dataset=trainingset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=validationset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               drop_last=False,
                                               num_workers=0)
    train_eval_loader = torch.utils.data.DataLoader(dataset=trainingset,
                                                    batch_size=train_size,
                                                    shuffle=False,
                                                    drop_last=False,
                                                    num_workers=0)
    num_classes = 10
    return (train_eval_loader, train_loader, valid_loader, trainingset,
            validationset, num_classes)


def prepare_train_loaders_for_uniform_cv_candidate_labels(
        dataname, full_train_loader, batch_size, partial_type):
    for i, (data, labels) in enumerate(full_train_loader):
        K = torch.max(
            labels
        ) + 1  # K is number of classes, full_train_loader is full batch
    partialY = generate_uniform_cv_candidate_labels(dataname, labels, partial_type)
    partial_matrix_dataset = gen_index_dataset(data, partialY.float(),
                                               partialY.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    dim = int(data.reshape(-1).shape[0] / data.shape[0])
    return partial_matrix_train_loader, data, partialY, dim

"""
First attempt for clustering-based Partial Label Generation
"""
def generate_cluster_based_candidate_labels(data, train_labels):
    if torch.min(train_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(train_labels) == 1:
        train_labels = train_labels - 1

    K = torch.max(train_labels) - torch.min(train_labels) + 1
    assert K == 10    
    n = train_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), train_labels] = 1.0

    # Clustering:
    _,c,dim1,dim2 = data.shape
    flattened_data = data.reshape((n, c*dim1*dim2))
    X = flattened_data.numpy()
    num_clusters = K
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1).fit(X)

    sample_size = int(n*0.01) # 1% 
    sample = random.sample(list(range(n)), sample_size)	
    confusion_labels = np.eye(K)
    for i,cluster_i in enumerate(kmeans.labels_[sample]):
        for j,cluster_j in enumerate(kmeans.labels_):
            if cluster_i==cluster_j:
                true_label_i = train_labels[i].item()
                true_label_j = train_labels[j].item()
                if true_label_i!=true_label_j:
                    confusion_labels[true_label_i, true_label_j] += 1
                    confusion_labels[true_label_j, true_label_i] += 1

    # normalize to get probs
    confusion_labels = normalize(confusion_labels, axis=1, norm='l1')
    np.fill_diagonal(confusion_labels, 1.0)
    print("Ambiguity degree: ", confusion_labels[confusion_labels<1.0].max())

    transition_matrix = confusion_labels

    random_n = np.random.uniform(0, 1, size=(n, K))

    for j in range(n):  # for each instance
        for jj in range(K): # for each class 
            if jj == train_labels[j]: # except true class
                continue
            if random_n[j, jj] < transition_matrix[train_labels[j], jj]:
                partialY[j, jj] = 1.0

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


def prepare_train_loaders_for_cluster_based_candidate_labels(
        dataname, full_train_loader, batch_size, partial_type):
    for i, (data, labels) in enumerate(full_train_loader):
        K = torch.max(
            labels
        ) + 1  # K is number of classes, full_train_loader is full batch
    partialY = generate_cluster_based_candidate_labels(data, labels)
    partial_matrix_dataset = gen_index_dataset(data, partialY.float(),
                                               partialY.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(
        dataset=partial_matrix_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0)
    dim = int(data.reshape(-1).shape[0] / data.shape[0])
    return partial_matrix_train_loader, data, partialY, dim


def cifar100_sparse2coarse(targets):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    # copied from https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]

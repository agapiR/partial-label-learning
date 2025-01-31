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
from sklearn.metrics.pairwise import cosine_distances as cdist
from sklearn.metrics.pairwise import euclidean_distances as edist
from synthetic_classification_generator import make_classification



import re

from utils.gen_index_dataset import gen_index_dataset


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = y
        self.X = X
        self.y = y
        assert X.shape[0] == y.shape[0]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]
        return (X, y)

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

    print("Loading Dataset {} - Random state is {}".format(dataname, seed))

    datapath = os.path.join(datadir, "{}.mat".format(dataname))
    dt = loadmat(datapath)

    X = dt['features']
    partial_y = dt['p_labels']
    y = dt['logitlabels']

    X = np.float32(X)
    partial_y = np.float32(partial_y)
    y = np.float32(np.argmax(y, axis=1))
    
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

    # Train datasets: Partial and Ordinary
    train_dataset = gen_index_dataset(train_X, train_partial_y, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)
    ordinary_train_dataset = RealDataset(train_X, train_y)
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_train_dataset,
        batch_size=len(ordinary_train_dataset),
        shuffle=False,
        num_workers=0)

    # Validation datasets: Partial and Ordinary
    valid_dataset = gen_index_dataset(valid_X, valid_partial_y, valid_y)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
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

    # Test datasets: Partial and Ordinary
    test_dataset = gen_index_dataset(test_X, test_partial_y, test_y)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0)
    ordinary_test_dataset = RealDataset(test_X, test_y)
    test_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_test_dataset,
        batch_size=len(ordinary_test_dataset),
        shuffle=False,
        num_workers=0)

    num_features = X.shape[1]
    num_classes = partial_y.shape[1]

    return (train_loader, train_eval_loader, 
            valid_loader, valid_eval_loader,
            test_loader, test_eval_loader, 
            torch.FloatTensor(train_partial_y), torch.FloatTensor(valid_partial_y), torch.FloatTensor(test_partial_y), 
            num_features, num_classes)



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
                match = re.match("h([0-9]*)uniform_(.*)", partial_type)
                if match:
                    assert dataname == "cifar100", dataname
                    p = float(match.groups()[1])
                    transition_matrix = np.ones((K,K)) * p
                    for i in range(K):
                        transition_matrix[i,i] = 1.0
                    labels = np.arange(K)

                    num_groups = match.groups()[0]
                    if num_groups == "":
                        num_groups = 20
                    else:
                        num_groups = int(num_groups)
                        
                    coarse_labels = cifar100_sparse2coarse(labels, num_groups)
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


def generate_cv_dataloader(dataname, batch_size, partial_rate, partial_type, noise_model="uniform", num_groups=10, distractionbased_ratio="0.0"):
    (full_train_loader, full_test_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes) = prepare_cv_datasets(dataname, batch_size)
    partial_matrix_train_loader, train_partial_Y, dim = prepare_loaders_for_cv_candidate_labels(dataname, full_train_loader, batch_size, partial_rate,
                                                                                                partial_type=partial_type,
                                                                                                noise_model=noise_model,
                                                                                                num_groups=num_groups,
                                                                                                distractionbased_ratio=distractionbased_ratio)
    partial_matrix_test_loader, test_partial_Y, dim = prepare_loaders_for_cv_candidate_labels(dataname, full_test_loader, batch_size, partial_rate,
                                                                                              partial_type=partial_type,
                                                                                              noise_model=noise_model,
                                                                                              num_groups=num_groups,
                                                                                              distractionbased_ratio=distractionbased_ratio)                                                                                              

    # TODO
    partial_matrix_valid_loader = partial_matrix_test_loader
    valid_loader = test_loader
    valid_partial_Y = test_partial_Y

    return (partial_matrix_train_loader, train_loader,
            partial_matrix_valid_loader, valid_loader,
            partial_matrix_test_loader, test_loader,
            train_partial_Y, valid_partial_Y, test_partial_Y,
            dim, num_classes
    )
    


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
    elif dataname.startswith('shierarchy32'):
        dset_args = dataname.split('-')
        dset = dset_args[0]
        csep = float(dset_args[1])
        seed = int(dset_args[2])
        # seed = 12
        # csep = 1.0
        levels = 6
        d = 120
        n = 100
        # ordinary_train_dataset, test_dataset = generate_synthetic_hierarchical_data(seed)
        ordinary_train_dataset, test_dataset = generate_synthetic_hierarchical_data(seed, 
                                                num_samples_per_class=n,
                                                feature_dim=d,
                                                num_levels=levels,
                                                class_sep=csep #1.0, prev 1.5
                                                )

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
    full_test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset.data),
        shuffle=True,
        num_workers=0)
    
    if dataname == 'cifar100':
        num_classes = 100
    elif dataname.startswith('shierarchy32'):
        num_classes = 32
    else:
        num_classes = 10
        
    return (full_train_loader, full_test_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, num_classes)


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


def prepare_loaders_for_cv_candidate_labels(dataname, full_loader, batch_size, partial_rate,
                                            partial_type="01",
                                            noise_model="uniform",
                                            num_groups=10,
                                            distractionbased_ratio=0.0):
    for i, (data, labels) in enumerate(full_loader):

        if noise_model == "distractionbased":
            num_classes = (torch.max(labels) - torch.min(labels) + 1).numpy(force=True)
            partialY = add_noise_distractionbased(data, labels, partial_rate, num_classes, distractionbased_ratio)
        elif noise_model == "cluster1":
            partialY = generate_cluster_based_candidate_labels(data, labels)
        elif noise_model == "cluster2":
            partialY = generate_cluster_based_candidate_labels2(data, labels, partial_rate)
        elif noise_model == "cluster3":
            partialY = generate_cluster_based_candidate_labels3(data, labels, partial_rate)
        elif noise_model == "instancebased":
            partialY = generate_instance_based_candidate_labels(data, labels, partial_rate, dataname, num_groups=num_groups)
        elif noise_model == "uniform":
            partialY = generate_uniform_cv_candidate_labels(dataname, labels, partial_type)
        else:
            assert False, "Unknown noise model: " + noise_model
            
        # partial_matrix_dataset = gen_index_dataset(data, partialY.float(), partialY.float())
        partial_matrix_dataset = gen_index_dataset(data, partialY.float(), labels.float())
        partial_matrix_train_loader = torch.utils.data.DataLoader(
            dataset=partial_matrix_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0)
    dim = int(data.reshape(-1).shape[0] / data.shape[0])
    return partial_matrix_train_loader, partialY, dim



"""
Clustering-based Partial Label Generation for real classification datasets. 
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

def generate_cluster_based_candidate_labels2(data, true_labels, partial_rate):
    if torch.min(true_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(true_labels) == 1:
        true_labels = true_labels - 1

    K = torch.max(true_labels) - torch.min(true_labels) + 1
    n = true_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), true_labels] = 1.0

    # Clustering (1. Find centroid of each class as the average of the samples belonging in the class. 3. Select distractors as the labels of the closest centroids.)
    #              Note, each label corresponds to exactly one centroid, and vice versa. 
    _,c,dim1,dim2 = data.shape
    flattened_data = data.reshape((n, c*dim1*dim2))
    X = flattened_data.numpy()

    centroids = []
    for i in range(K):
        # find the centroid of the cluster
        X_curr = X[true_labels == i]
        centroid = np.mean(X_curr, axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)
    sample_centroid_distances = cdist(X, Y=centroids)

    num_distractors = int(partial_rate*K)-1
    for x in range(n):
        candidate_distractors_sorted = list(np.argsort(sample_centroid_distances[x]))
        distractors = candidate_distractors_sorted[:num_distractors]
        if true_labels[x] in distractors:
            distractors.append(candidate_distractors_sorted[num_distractors])            
        partialY[x, distractors] = 1

    print("Finished Generating Candidate Label Sets!\n")
    return partialY


def generate_cluster_based_candidate_labels3(data, true_labels, partial_rate, cluster_per_class=2):
    if torch.min(true_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(true_labels) == 1:
        true_labels = true_labels - 1

    K = torch.max(true_labels) - torch.min(true_labels) + 1
    # assert K == 10    
    n = true_labels.shape[0]

    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), true_labels] = 1.0

    # Clustering (1. Apply kmeans 2. Label cluster centroids with the dominant label in their cluster. 3. Select distractors as the labels of the closest centroids.)
    #              Note, each label may be associated with more than one centroids. Each centroid is assigned exactly one label. 
    _,c,dim1,dim2 = data.shape
    flattened_data = data.reshape((n, c*dim1*dim2))
    X = flattened_data.numpy()
    num_clusters = cluster_per_class*K 
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=1).fit(X)

    # Get Centroids and Centroids' labels:
    centroids = kmeans.cluster_centers_
    cluster_membership = kmeans.labels_
    y_centroids = np.empty(num_clusters)
    for c in range(num_clusters):
        # find the indices of the cluster members
        cluster_members = np.where(cluster_membership == c)[0]
        # find the true labels of the cluster members
        cluster_members_labels = true_labels[cluster_members]
        # find the domimant label(s)
        cluster_dominant_label = np.bincount(cluster_members_labels).argmax()
        # assign the dominant label as the label of the centroid
        y_centroids[c] = cluster_dominant_label


    # Generate Partial Labels:
    sample_centroid_distances = cdist(X, Y=centroids)
    num_distractors = int(partial_rate*K)-1
    for x in range(n):
        candidate_distractors_sorted = list(np.argsort(sample_centroid_distances[x]))
        distractors = candidate_distractors_sorted[:num_distractors]
        # the distractor labels could contain duplicate labels and/or the true label
        distractor_labels = y_centroids[distractors]         
        partialY[x, distractor_labels] = 1

    print("Finished Generating Candidate Label Sets!\n")
    return partialY

def generate_instance_based_candidate_labels(data, true_labels, partial_rate, dataname, num_groups=10):
    if torch.min(true_labels) > 1:
        raise RuntimeError('testError')
    elif torch.min(true_labels) == 1:
        true_labels = true_labels - 1

    K = torch.max(true_labels) - torch.min(true_labels) + 1
    n = true_labels.shape[0]

    # n = 500
    # true_labels = true_labels[:n]
    # data=data[:n]


    # Generate Partial Labels:
    partialY = torch.zeros(n, K)
    partialY[torch.arange(n), true_labels] = 1.0

    if partial_rate == 0:
        print("No distractors")
        return partialY



    # find distances to nearest instances from each class d_1...d_k
    # compute normalised probability vector <alpha/d_i>
    # sample partial_rate * K distractors
    if len(data.shape)==2:
        flattened_data = data
    elif len(data.shape)==3:
        _,dim1,dim2 = data.shape
        flattened_data = data.reshape((n, dim1*dim2))
    else:
        _,c,dim1,dim2 = data.shape
        flattened_data = data.reshape((n, c*dim1*dim2))
    X = flattened_data.numpy()

    
    dist_matrix = torch.zeros(n, K)
    for i in range(K):
        X_k = X[true_labels==i]
        ds = edist(X,X_k)
        ds = ds.min(axis=1)
        dist_matrix[:,i] = torch.tensor(ds)
    prob_matrix = 1 / (dist_matrix+1e-5)
    prob_matrix[torch.arange(n), true_labels] = 0.0

    if dataname == "cifar100":
        labels = np.arange(K)
        coarse_labels = cifar100_sparse2coarse(labels, num_groups)
        different_coarse_label = np.expand_dims(coarse_labels,0) != np.expand_dims(coarse_labels,1)
        for i in range(n):
            prob_matrix[i, different_coarse_label[true_labels[i]]] = 0
    elif dataname.startswith('shierarchy32'):
        labels = np.arange(K)
        coarse_labels = shierarchy32_sparse2coarse(labels, num_groups)
        different_coarse_label = np.expand_dims(coarse_labels,0) != np.expand_dims(coarse_labels,1)
        for i in range(n):
            prob_matrix[i, different_coarse_label[true_labels[i]]] = 0

    # we set the highest probability to partial_rate
    prob_matrix *= partial_rate / torch.max(prob_matrix, dim=1, keepdim=True).values
    distractors = torch.distributions.Binomial(1, prob_matrix).sample().long()

    # we sample partial_rate * K distractors
    # prob_matrix /= prob_matrix.sum(dim=1, keepdim=True)
    # distractors = torch.multinomial(prob_matrix, int(partial_rate * K), replacement=False) #  TODO maybe with replacement?


    
    for i in range(n):
        partialY[i, torch.nonzero(distractors[i])] = 1

    print("Avg number of distractors: ", torch.mean(distractors.sum(dim=1).float()))
    print("Finished Generating Instance based Candidate Label Sets!\n")
    return partialY


"""
End
"""


def cifar100_sparse2coarse(targets, groups):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.
    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    # 0: aquatic mammals
    # 1: fish	
    # 2: flowers
    # 3: food containers
    # 4: fruit and vegetables
    # 5: household electrical devices
    # 6: household furniture	
    # 7: insects
    # 8: large carnivores
    # 9: large man-made outdoor things
    # 10:large natural outdoor scenes
    # 11:large omnivores and herbivores	
    # 12:medium-sized mammals
    # 13:non-insect invertebrates
    # 14:people
    # 15:reptiles	
    # 16:small mammals	
    # 17:trees	
    # 18:vehicles 1
    # 19:vehicles 2

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
    new_targets = coarse_labels[targets]
    if groups == 20:
        return new_targets
    elif groups == 50:
        coarse_labels = np.repeat(np.arange(50), 2)
        return coarse_labels[targets]
    else:
        if groups==10:
            coarser_labels = [0, 0, 1, 2, 1, 2, 3, 7, 4, 3, 8, 4, 5, 7, 5, 6, 6, 8, 9, 9]
        elif groups==5:
            coarser_labels = [1, 1, 0, 1, 0, 1, 4, 3, 2, 4, 0, 2, 2, 3, 2, 3, 3, 0, 4, 4]
        elif groups==4:
            coarser_labels = [0, 0, 1, 2, 1, 2, 2, 0, 3, 1, 1, 3, 3, 0, 3, 0, 3, 1, 2, 2]
        elif groups==2: # animals vs the rest
            coarser_labels = [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        elif groups==1: # no hierarchy
            coarser_labels = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        coarser_labels = np.array(coarser_labels)
        return coarser_labels[new_targets]


def shierarchy32_sparse2coarse(targets, groups):

    # "groups" must belong in the list of valid groups, given the hierarchy structure.
    valid_groups = [2, 4, 6, 8, 16, 32]

    def closest_element(num, lst):
        return min(lst, key=lambda x: abs(x - num))

    groups = closest_element(groups, valid_groups)
    assert groups in valid_groups

    C = 32
    num_classes_per_group = C // groups
    coarse_labels = np.array([i for labels in [[c]*num_classes_per_group for c in range(groups)] for i in labels])

    return coarse_labels[targets]
        

## Synthetic Hypercube Dataset PLL generation
def generate_synthetic_hypercube_dataloader(partial_rate, batch_size, seed, num_classes=5,
                                            num_samples=1000,
                                            feature_dim=5,
                                            class_sep=0.1,
                                            noise_model="distancebased",
                                            distractionbased_ratio=1.0,
                                            use_cache=True):
    
    cachepath = "cache/pr-{}_classes-{}_samples-{}_features-{}_sep-{}_noise-{}_dratio-{}_seed-{}.npz".format(partial_rate, num_classes, num_samples, feature_dim, class_sep, noise_model, distractionbased_ratio, seed)

    if use_cache and os.path.isfile(cachepath):
        print("Loading dataset from cache", cachepath)
        npzfile = np.load(cachepath)
        train_X = npzfile['arr_0']
        valid_X = npzfile['arr_1']
        test_X = npzfile['arr_2']
        train_y = npzfile['arr_3']
        valid_y = npzfile['arr_4']
        test_y = npzfile['arr_5']
        train_partial_y = npzfile['arr_6']
        valid_partial_y = npzfile['arr_7']
        test_partial_y = npzfile['arr_8']
    else:        
        train_X, valid_X, test_X, train_y, valid_y, test_y, train_partial_y, valid_partial_y, test_partial_y = generate_synthetic_hypercube_data(partial_rate, seed, num_classes, num_samples, feature_dim, class_sep, noise_model, distractionbased_ratio)
        os.makedirs(os.path.dirname(cachepath), exist_ok=True)
        np.savez(cachepath, train_X, valid_X, test_X, train_y, valid_y, test_y, train_partial_y, valid_partial_y, test_partial_y)


    print(train_X.shape[0], valid_X.shape[0], test_X.shape[0])

    ordinary_train_dataset = RealDataset(train_X, train_y)
    train_eval_loader = torch.utils.data.DataLoader(
        dataset=ordinary_train_dataset,
        batch_size=len(ordinary_train_dataset),
        shuffle=False,
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

    # train_dataset = RealIdxDataset(train_X, train_partial_y)
    train_dataset = gen_index_dataset(train_X, train_partial_y, train_y)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    valid_dataset = gen_index_dataset(valid_X, valid_partial_y, valid_y)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)
    test_dataset = gen_index_dataset(test_X, test_partial_y, test_y)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)


    num_features = train_X.shape[1]
    num_classes = train_partial_y.shape[1]

    return (train_loader, train_eval_loader,
            valid_loader, valid_eval_loader,
            test_loader, test_eval_loader,
            train_partial_y, valid_partial_y, test_partial_y, num_features, num_classes)

        
        
def generate_synthetic_hypercube_data(partial_rate, seed, 
                                      num_classes=5,
                                      num_samples=1000,
                                      feature_dim=5,
                                      class_sep=0.1,
                                      noise_model="distancebased",
                                      distractionbased_ratio=1.0):

    ## Generate Samples
    X, y, centroids, y_centroids = make_classification( n_samples=num_samples,
                                                        # n_features = informative + redundant 
                                                        #               + repeated + random/useless
                                                        n_features=feature_dim, 
                                                        # all features are informative
                                                        n_informative=feature_dim, 
                                                        n_redundant=0,
                                                        n_repeated=0,
                                                        n_classes=num_classes,
                                                        # each class is associated with a single cluster
                                                        n_clusters_per_class=1,
                                                        flip_y=0.0,     # default = 0.01
                                                        class_sep=class_sep,
                                                        hypercube=True,
                                                        shift=0.0,
                                                        scale=1.0,
                                                        shuffle=True,
                                                        random_state=seed,
                                                        return_centroids=True)

    ## Generate Partial Labels
    if noise_model == "distancebased":
        partial_y = add_noise_distancebased(X, y, partial_rate, num_classes, centroids)
    elif noise_model == "distractionbased":
        partial_y = add_noise_distractionbased(X, y, partial_rate, num_classes, distractionbased_ratio)
    else:
        assert False, "Unknown noise model" + noise_model
    
    ## Create Splits
    X = np.float32(X)
    y = np.float32(y)
    partial_y = np.float32(partial_y)

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

    return train_X, valid_X, test_X, train_y, valid_y, test_y, train_partial_y, valid_partial_y, test_partial_y


def add_noise_distancebased(X, y, partial_rate, num_classes, centroids):
    num_classes = torch.max(y) - torch.min(y) + 1
    num_samples = y.shape[0]
    partial_y = torch.zeros(num_samples, num_classes)
    partial_y[torch.arange(num_samples), y] = 1
    
    num_distractors = int(partial_rate*num_classes)
    sample_centroid_distances = cdist(X, Y=centroids)
    for x in range(num_samples):
        candidate_distractors_sorted = list(np.argsort(sample_centroid_distances[x]))
        distractors = candidate_distractors_sorted[:num_distractors]
        # if the true label is selected among the distractors, replace with additional distractor
        if y[x] in distractors:
            distractors.append(candidate_distractors_sorted[num_distractors-1])
        partial_y[x, distractors] = 1
    return partial_y

def add_noise_distractionbased(X, y, partial_rate, num_classes, distractionbased_ratio):
    num_classes = torch.max(y) - torch.min(y) + 1
    num_samples = y.shape[0]
    partial_y = torch.zeros(num_samples, num_classes)
    partial_y[torch.arange(num_samples), y] = 1
    
    for class_curr in range(num_classes):
        indices = np.where(y==class_curr)[0]
        y_curr = y[indices]
        X_curr = X[indices]
        distractor_count = int(partial_rate * len(indices))
        # we only have distractors from int(num_classes * distractionbased_ratio) classes
        distractor_classes = np.random.choice(range(num_classes), int(num_classes * distractionbased_ratio), replace=False)
        for class_distractor in distractor_classes:
            if class_distractor== class_curr:
                continue
            distractors = np.random.choice(indices, distractor_count, replace=False)
            partial_y[distractors, class_distractor] = 1
    return partial_y



## Synthetic Hierarchical Dataset PLL generation


def generate_synthetic_hierarchical_data(seed, 
                                        num_samples_per_class=100,
                                        feature_dim=120,
                                        num_levels=6,
                                        class_sep=1.5,
                                        use_cache=False
                                        ):

    num_classes = 2**(num_levels-1)
    num_samples = num_samples_per_class * num_classes
    cachepath = "cache/shierarchy_classes-{}_levels-{}_samples-{}_features-{}_sep-{}_seed-{}.npz".format(num_classes, num_levels, num_samples, feature_dim, class_sep, seed)

    if use_cache and os.path.isfile(cachepath):
        print("Loading dataset from cache", cachepath)
        npzfile = np.load(cachepath)
        X = npzfile['arr_0']
        y = npzfile['arr_1']
    else:        
        ## Generate Samples
        features_dim_per_level = feature_dim//num_levels
        X = np.empty((num_samples, features_dim_per_level*num_levels))
        for l in range(num_levels):
            num_classes_per_level = 2**l
            # Xl shape: (num_samples * features_dim_per_level)
            Xl, _, _, _ = make_classification(n_samples=num_samples,
                                            n_features=features_dim_per_level, 
                                            n_informative=features_dim_per_level, 
                                            n_redundant=0,
                                            n_repeated=0,
                                            n_classes=num_classes_per_level,
                                            n_clusters_per_class=1,
                                            flip_y=0.0,
                                            class_sep=class_sep,
                                            hypercube=True,
                                            shift=0.0,
                                            scale=1.0,
                                            shuffle=False,
                                            random_state=seed,
                                            return_centroids=True)

            X[:, (l*features_dim_per_level):((l+1)*features_dim_per_level)] = Xl

        # TODO: shuffle the features! 
        y = np.array([i for labels in [[c]*num_samples_per_class for c in range(num_classes)] for i in labels])

        ## Save Cache
        os.makedirs(os.path.dirname(cachepath), exist_ok=True)
        np.savez(cachepath, X, y)


    ## Create Splits
    X = np.float32(X)
    y = np.longlong(y)

    print("random_state is {}".format(seed))
    train_X, test_X, train_y, test_y = train_test_split(X,
                                                        y,
                                                        train_size=0.8,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=seed)
    # train_X, valid_X, train_y, valid_y = train_test_split(train_X,
    #                                                     train_y,
    #                                                     train_size=7 / 8,
    #                                                     test_size=1 / 8,
    #                                                     stratify=train_y,
    #                                                     random_state=seed)

    
    ## Create Datasets
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    # valid_X = scaler.transform(valid_X)
    test_X = scaler.transform(test_X)

    ordinary_train_dataset = CustomDataset(train_X, train_y)
    ordinary_test_dataset = CustomDataset(test_X, test_y)

    return ordinary_train_dataset, ordinary_test_dataset

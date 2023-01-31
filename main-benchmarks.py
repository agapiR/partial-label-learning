import os
import argparse
import numpy as np
import torch
from models.model_linear import Linearnet
from models.model_mlp import Mlp
from models.model_cnn import Cnn
from models.model_resnet import Resnet
from utils.utils_data import generate_real_dataloader
from utils.utils_data import prepare_cv_datasets
from utils.utils_data import prepare_train_loaders_for_uniform_cv_candidate_labels, prepare_train_loaders_for_cluster_based_candidate_labels
from utils.utils_algo import accuracy_check, confidence_update, confidence_update_lw, prob_check, ratio_check
from utils.utils_loss import (rc_loss, cc_loss, lws_loss, 
                              log_prp_Loss as prp_loss, 
                              h_prp_Loss as h_prp_loss, 
                              joint_prp_on_logits as ll_loss,
                              bi_prp_loss, bi_prp_nll_loss, nll_loss, democracy_loss)

# TODO: read as argument
CLUSTER_PLL = True


parser = argparse.ArgumentParser()

parser.add_argument('-ds',
                    help='specify a dataset',
                    default='birdac',
                    type=str,
                    required=False)  # birdac, lost, LYN, MSRCv2, spd
parser.add_argument('-pr', help='partial_type', default="01", type=str)
parser.add_argument(
    '-mo',
    help='model name',
    default='mlp',
    choices=['linear', 'mlp', 'cnn', 'resnet', 'densenet', 'lenet'],
    type=str,
    required=False)
parser.add_argument('-lo',
                    help='specify a loss function',
                    default='rc',
                    type=str,
                    choices=['rc', 'cc', 'lws', 'prp', 'hprp', 'll', 'bi_prp', 'nll', 'bi_prp_nll', 'democracy'],
                    required=False)
parser.add_argument('-lw',
                    help='lw sigmoid loss weight',
                    default=0,
                    type=float,
                    required=False)
parser.add_argument('-lw0',
                    help='lw of first term',
                    default=1,
                    type=float,
                    required=False)
parser.add_argument('-lr',
                    help='optimizer\'s learning rate',
                    default=1e-3,
                    type=float)
parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
parser.add_argument('-ldr',
                    help='learning rate decay rate',
                    default=0.5,
                    type=float)
parser.add_argument('-lds',
                    help='learning rate decay step',
                    default=50,
                    type=int)
parser.add_argument('-bs',
                    help='batch_size of ordinary labels.',
                    default=256,
                    type=int)
parser.add_argument('-ep', help='number of epochs', type=int, default=250)
parser.add_argument('-seed',
                    help='Random seed',
                    default=0,
                    type=int,
                    required=False)
parser.add_argument('-gpu',
                    help='used gpu id',
                    default='0',
                    type=str,
                    required=False)
parser.add_argument('-alpha',
                    help='alpha-coefficient for gradient.',
                    default=1.0, # 1 is equivalent to prp
                    type=float,
                    required=False)

args = parser.parse_args()

save_dir = "./results_cv_best"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
if args.lo in ['rc', 'cc', 'prp', 'bi_prp', 'bi_prp_nll', 'nll', 'democracy']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed)
elif args.lo in ['hprp']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed, args.alpha)
elif args.lo in ['ll']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed)
elif args.lo in ['lws']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        args.ds, args.mo, args.lo, args.lw0, args.lw, args.lr, args.wd,
        args.ldr, args.lds, args.ep, args.bs, args.seed)
save_path = os.path.join(save_dir, save_name)
with open(save_path, 'w') as f:
    f.writelines("epoch,train_acc,test_acc,train_pos_prob\n") # TODO: if file already exists, erase previous, DONE

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:" +
                      args.gpu if torch.cuda.is_available() else "cpu")

## TODO: Replace the prepare_cv_datasets with loading benchmark datasets
# (full_train_loader, train_loader, test_loader, ordinary_train_dataset,
#  test_dataset, K) = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
 
# (partial_matrix_train_loader, train_data, train_givenY,
#  dim) = prepare_train_loaders_for_uniform_cv_candidate_labels(
#      dataname=args.ds,
#      full_train_loader=full_train_loader,
#      batch_size=args.bs,
#      partial_type=args.pr)

# (train_loader, train_eval_loader, valid_eval_loader,
# test_eval_loader, train_partial_y, num_features, num_classes)

if args.ds in ['birdac', 'lost']:
    (partial_matrix_train_loader, train_loader, eval_loader, test_loader, partialY, dim, K) = generate_real_dataloader(args.ds, './data/realworld/', args.bs, 42)
    train_givenY = partialY
    train_givenY = torch.tensor(train_givenY)

elif args.ds in ['mnist', 'kmnist', 'fashion', 'cifar10']:
    (full_train_loader, train_loader, test_loader, ordinary_train_dataset, test_dataset, K) = prepare_cv_datasets(dataname=args.ds, batch_size=args.bs)
    if CLUSTER_PLL:
        (partial_matrix_train_loader, train_data, train_givenY, dim) = prepare_train_loaders_for_cluster_based_candidate_labels(
        dataname=args.ds,
        full_train_loader=full_train_loader,
        batch_size=args.bs,
        partial_type=args.pr)
    else:
        (partial_matrix_train_loader, train_data, train_givenY, dim) = prepare_train_loaders_for_uniform_cv_candidate_labels(
        dataname=args.ds,
        full_train_loader=full_train_loader,
        batch_size=args.bs,
        partial_type=args.pr)



if args.lo == 'rc':
    tempY = train_givenY.sum(dim=1).unsqueeze(1).repeat(
        1, train_givenY.shape[1])
    confidence = train_givenY.float() / tempY
    confidence = confidence.to(device)
    loss_fn = rc_loss
elif args.lo == 'cc':
    loss_fn = cc_loss
elif args.lo == 'lws':
    n, c = train_givenY.shape[0], train_givenY.shape[1]
    confidence = torch.ones(n, c) / c
    confidence = confidence.to(device)
    loss_fn = lws_loss
elif args.lo == 'prp':
    # softmax = torch.nn.Softmax(dim=1)
    loss_fn = prp_loss()
elif args.lo == 'hprp':
    loss_fn = h_prp_loss(h=args.alpha)
elif args.lo == 'll':
    loss_fn = ll_loss()
elif args.lo == 'bi_prp':
    loss_fn = bi_prp_loss()
elif args.lo == 'bi_prp_nll':
    loss_fn = bi_prp_nll_loss()
elif args.lo == 'nll':
    loss_fn = nll_loss()
elif args.lo == 'democracy':
    loss_fn = democracy_loss()

if args.mo == 'mlp':
    model = Mlp(n_inputs=dim, n_outputs=K)
elif args.mo == 'linear':
    model = Linearnet(n_inputs=dim, n_outputs=K)
elif args.mo == 'cnn':
    input_channels = 3
    dropout_rate = 0.25
    model = Cnn(input_channels=input_channels,
                n_outputs=K,
                dropout_rate=dropout_rate)
elif args.mo == "resnet":
    model = Resnet(depth=32, n_outputs=K)

model = model.to(device)
print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
train_pos_prob = prob_check(loader=partial_matrix_train_loader, model=model, device=device)
# train_ratios = ratio_check(loader=partial_matrix_train_loader, model=model, device=device)
train_ratios = 0.0 # TODO

print('Epoch: 0. Tr Acc: {:.6f}. Te Acc: {:.6f}. Tr Pos Prob {:.6f}. Tr Ratios {}.'.format(
        0, train_accuracy, test_accuracy, train_pos_prob, np.around(train_ratios,6)))
with open(save_path, "a") as f:
    f.writelines("{},{:.6f},{:.6f},{:.6f}\n".format(0, train_accuracy, test_accuracy, train_pos_prob))

lr_plan = [args.lr] * args.ep
for i in range(0, args.ep):
    lr_plan[i] = args.lr * args.ldr**(i / args.lds)


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_plan[epoch]


test_acc_list = []
train_acc_list = []

for epoch in range(args.ep):
    model.train()
    adjust_learning_rate(optimizer, epoch)
    for i, (images, labels, true_labels,
            index) in enumerate(partial_matrix_train_loader):
        X, Y, index = images.to(device), labels.to(device), index.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        if args.lo == 'rc':
            average_loss = loss_fn(outputs, confidence, index)
        elif args.lo == 'cc':
            average_loss = loss_fn(outputs, Y.float())
        elif args.lo == 'lws':
            average_loss, _, _ = loss_fn(outputs, Y.float(), confidence, index,
                                         args.lw, args.lw0, None)
        elif args.lo in ['prp', 'hprp', 'bi_prp', 'bi_prp_nll']:
            # average_loss = loss_fn(softmax(outputs), Y.float())
            average_loss = loss_fn(outputs, Y.float())
        elif args.lo == 'll':
            average_loss = loss_fn(outputs, Y.float())
        elif args.lo == 'nll':
            average_loss, _ = loss_fn(outputs, Y.float())

        average_loss.backward()
        optimizer.step()
        if args.lo == 'rc':
            confidence = confidence_update(model, confidence, X, Y, index)
        elif args.lo == 'lws':
            confidence = confidence_update_lw(model, confidence, X, Y, index)
        
    model.eval()
    train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
    test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
    train_pos_prob = prob_check(loader=partial_matrix_train_loader, model=model, device=device)
    # train_ratios = ratio_check(loader=partial_matrix_train_loader, model=model, device=device)
    train_ratios = 0.0 # TODO

    print('Epoch: {}. Tr Acc: {:.6f}. Te Acc: {:.6f}. Tr Pos Prob {:.6f}. Tr Ratios {}.'.format(
        epoch + 1, train_accuracy, test_accuracy, train_pos_prob, np.around(train_ratios,6)))
    with open(save_path, "a") as f:
        f.writelines("{},{:.6f},{:.6f},{:.6f}\n".format(epoch + 1, train_accuracy,
                                                 test_accuracy, train_pos_prob))

    if epoch >= (args.ep - 10):
        test_acc_list.extend([test_accuracy])
        train_acc_list.extend([train_accuracy])

avg_test_acc = np.mean(test_acc_list)
avg_train_acc = np.mean(train_acc_list)

print("Learning Rate:", args.lr, "Weight Decay:", args.wd)
print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
print("Average Training Accuracy over Last 10 Epochs:", avg_train_acc,
      "\n\n\n")

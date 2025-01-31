import os
import argparse
import numpy as np
import torch
import sys
import time
from models.model_linear import Linearnet
from models.model_mlp import Mlp
from models.model_cnn import Cnn
from models.model_resnet import Resnet
from pytorch_cifar100.models.resnet import resnet18, resnet50, resnet101, resnet34
from utils.utils_data import generate_real_dataloader, generate_synthetic_hypercube_dataloader
from utils.utils_data import generate_cv_dataloader
#from utils.utils_data import prepare_train_loaders_for_uniform_cv_candidate_labels, prepare_train_loaders_for_cluster_based_candidate_labels
from utils.utils_algo import accuracy_check, confidence_update, confidence_update_lw, prob_check, ratio_check
from utils.utils_loss import (rc_loss, cc_loss, lws_loss, 
                              log_prp_Loss as prp_loss, 
                              h_prp_Loss as h_prp_loss, 
                              joint_prp_on_logits as ll_loss,
                              bi_prp_loss, bi_prp_nll_loss, nll_loss, democracy_loss, meritocratic_loss)

# TODO: read as argument
NO_IMPROVEMENT_TOLERANCE=20

print(sys.argv[1:], file=sys.stderr)


parser = argparse.ArgumentParser()

parser.add_argument('-ds',
                    help='specify a dataset',
                    default='birdac',
                    type=str,
                    required=False)  # birdac, lost, LYN, MSRCv2, spd
parser.add_argument('-pr', help='partial_type', default="01", type=str)
parser.add_argument('-mo',
                    help='model name',
                    default='mlp',
                    choices=['linear', 'mlp', 'cnn', 'resnet', 'densenet', 'lenet', "resnet18", "resnet50", "resnet34", "resnet101"],
                    type=str,
                    required=False)
parser.add_argument('-lo',
                    help='specify a loss function',
                    default='rc',
                    type=str,
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
parser.add_argument('-res',
                    help='result directory.',
                    default="./results_cv_best",
                    type=str,
                    required=False)
parser.add_argument('-cluster',
                    help='whether to do classwise clustering: 0/1/2/3',
                    type=int,
                    default=0,
                    required=False)
parser.add_argument('-num_groups',
                    help='size of the label set partition for drawing distractors',
                    type=int,
                    default=10,
                    required=False)
parser.add_argument('-clip', help='gradient clipping', type=float, default=None, required=False)
parser.add_argument('-logit_decay', help='L2 loss on the logits', type=float, default=0.0, required=False)
parser.add_argument('-beta', help='Beta parameter for beta-meritocratic loss.', type=float, default=0.5, required=False)

## Synthetic data hyperparameters
parser.add_argument('-prt', help='partial rate.', default=0.1, type=float, required=False)                  
parser.add_argument('-nc', help='number of classes.', default=10, type=int, required=False)                  
parser.add_argument('-ns', help='number of samples.', default=1000, type=int, required=False)                  
parser.add_argument('-nf', help='number of features.', default=5, type=int, required=False)                  
parser.add_argument('-csep', help='class separation.', default=0.1, type=float, required=False)        
parser.add_argument('-dseed', help='Random seed for data generation.', default=42, type=int, required=False)
parser.add_argument('-noise_model', help='Noise model', type=str, required=True,
                    choices=['distancebased', 'distractionbased', "cluster1", "cluster2", "cluster3", "instancebased", "uniform", "real"])
parser.add_argument('-distractionbased_ratio', help='ratio of classes that are distractors for a given class.', default=1.0, type=float, required=False)        
args = parser.parse_args()

## Output directory
save_dir = args.res
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## Dataset name
if args.ds.startswith('synthetic'):
    dname = f'synthetic_{args.ns}_{args.nc}_{args.nf}_{args.prt}_{args.csep}_{args.noise_model}_{args.dseed}'
elif args.ds.startswith('shierarchy'):
    dname = f'{args.ds}-{args.csep}-{args.dseed}'
else:
    dname = args.ds

if args.lo in ['hprp']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        dname, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed, args.alpha)
elif args.lo in ['ll']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        dname, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed)
elif args.lo in ['lws']:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        dname, args.mo, args.lo, args.lw0, args.lw, args.lr, args.wd,
        args.ldr, args.lds, args.ep, args.bs, args.seed)
else:
    save_name = "Res-sgd_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
        dname, args.mo, args.lo, args.lr, args.wd, args.ldr,
        args.lds, args.ep, args.bs, args.seed)
save_path = os.path.join(save_dir, save_name)
with open(save_path, 'w') as f:
    f.writelines("epoch,train_acc,test_acc,train_pos_prob\n")

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

if args.ds in ['birdac', 'lost', 'MSRCv2', 'LYN', 'spd']:
    # (partial_matrix_train_loader, train_loader, eval_loader, test_loader, train_partial_Y, dim, K) = generate_real_dataloader(args.ds, './data/realworld/', args.bs, 42)
    (partial_matrix_train_loader, train_loader,
     partial_matrix_valid_loader, valid_loader,
     partial_matrix_test_loader, test_loader,
     train_partial_Y, valid_partial_Y, test_partial_Y,
     dim, K) = generate_real_dataloader(args.ds, './data/realworld/', args.bs, args.dseed)
    train_givenY = train_partial_Y

elif args.ds in ['mnist', 'kmnist', 'fashion', 'cifar10', 'cifar100']:
    (partial_matrix_train_loader, train_loader,
     partial_matrix_valid_loader, valid_loader,
     partial_matrix_test_loader, test_loader,
     train_partial_Y, valid_partial_Y, test_partial_Y,
     dim, K) = generate_cv_dataloader(dataname=args.ds, batch_size=args.bs, partial_rate = args.prt, partial_type=args.pr,
                                      noise_model=args.noise_model,
                                      num_groups=args.num_groups,
                                      distractionbased_ratio=args.distractionbased_ratio)
    train_givenY = train_partial_Y

elif args.ds in ['shierarchy32']:
    (partial_matrix_train_loader, train_loader,
     partial_matrix_valid_loader, valid_loader,
     partial_matrix_test_loader, test_loader,
     train_partial_Y, valid_partial_Y, test_partial_Y,
     dim, K) = generate_cv_dataloader(dataname=dname, batch_size=args.bs, partial_rate = args.prt, partial_type=args.pr,
                                      noise_model=args.noise_model,
                                      num_groups=args.num_groups,
                                      distractionbased_ratio=args.distractionbased_ratio)
    train_givenY = train_partial_Y
        
elif args.ds.startswith('synthetic'):
    (partial_matrix_train_loader, train_loader,
     partial_matrix_valid_loader, valid_loader,
     partial_matrix_test_loader, test_loader,
     train_partial_Y, valid_partial_Y, test_partial_Y,
     dim, K) = generate_synthetic_hypercube_dataloader(args.prt, args.bs, args.dseed, num_classes=args.nc, num_samples=args.ns, feature_dim=args.nf, class_sep=args.csep, noise_model=args.noise_model, distractionbased_ratio=args.distractionbased_ratio)
    train_givenY = train_partial_Y
    train_givenY = torch.tensor(train_givenY)

if args.lo == 'rc':
    print(train_givenY.shape)
    print(train_givenY.type)
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
    loss_fn = prp_loss()
elif args.lo == 'prp_basic':
    loss_fn = prp_loss(use_weighting=False)
elif args.lo == 'hprp':
    loss_fn = h_prp_loss(h=args.alpha)
elif args.lo == 'll':
    loss_fn = ll_loss()
elif args.lo == 'bi_prp':
    loss_fn = bi_prp_loss(from_logits=True, logit_decay=args.logit_decay)
elif args.lo == 'bi_prp2':
    loss_fn = bi_prp_loss(from_logits=False, logit_decay=args.logit_decay)
elif args.lo == 'bi_prp_nll':
    loss_fn = bi_prp_nll_loss()
elif args.lo == 'nll':
    loss_fn = nll_loss()
elif args.lo == 'democracy':
    loss_fn = democracy_loss()
elif args.lo == 'merit':
    loss_fn = meritocratic_loss(args.beta)

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
elif args.mo == "resnet50":
    assert args.ds == "cifar100"
    model = resnet50()
elif args.mo == "resnet18":
    assert args.ds == "cifar100"
    model = resnet18()

model = model.to(device)
print(model)
print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.3*args.ep),int(0.6*args.ep), int(0.8*args.ep)], gamma=0.2) #learning rate decay

train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
train_pos_prob, _ = prob_check(loader=partial_matrix_train_loader, model=model, device=device)

print('Epoch: 0. Tr Acc: {:.6f}. Te Acc: {:.6f}. Tr Pos Prob {:.6f}.'.format(
        0, train_accuracy, test_accuracy, train_pos_prob))
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
test_prob_pll_list = []
test_prob_list = []
train_prob_pll_list = []
train_prob_list = []

best_valid_pos_prob = 0.0
tolerance = NO_IMPROVEMENT_TOLERANCE
for epoch in range(args.ep):
    T0 = time.time()
    model.train()
    # adjust_learning_rate(optimizer, epoch)
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
        elif args.lo == 'nll':
            average_loss, _ = loss_fn(outputs, Y.float())
        else:
            average_loss = loss_fn(outputs, Y.float())

        average_loss.backward()

        # for layer in model.parameters():
        #     print("Gradients: ", torch.sqrt(torch.square(layer.grad).sum()))

        if args.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        
        optimizer.step()
        if args.lo == 'rc':
            confidence = confidence_update(model, confidence, X, Y, index)
        elif args.lo == 'lws':
            confidence = confidence_update_lw(model, confidence, X, Y, index)
        
    train_scheduler.step()
    model.eval()
    train_accuracy = accuracy_check(loader=train_loader, model=model, device=device)
    train_pos_prob, train_pos_prob_true = prob_check(loader=partial_matrix_train_loader, model=model, device=device)
    valid_pos_prob, _ = prob_check(loader=partial_matrix_valid_loader, model=model, device=device)    
    valid_accuracy = accuracy_check(loader=valid_loader, model=model, device=device)
    test_accuracy = accuracy_check(loader=test_loader, model=model, device=device)
    test_pos_prob, test_pos_prob_true = prob_check(loader=partial_matrix_test_loader, model=model, device=device)
    
    # train_ratios = ratio_check(loader=partial_matrix_train_loader, model=model, device=device)

    T1 = time.time()
    print('Epoch: {}. Sec: {}. Tr Acc: {:.6f}. Tr Pos Prob {:.6f}. Va Acc: {:.6f}. Va Pos Prob {:.6f}.'.format(
        epoch + 1, T1-T0, train_accuracy, train_pos_prob, valid_accuracy, valid_pos_prob))
    sys.stdout.flush()
    with open(save_path, "a") as f:
        f.writelines("{},{:.6f},{:.6f},{:.6f}\n".format(epoch + 1, train_accuracy,
                                                        test_accuracy, train_pos_prob))

    # if epoch >= (args.ep - 10):
    #     test_acc_list.extend([test_accuracy])
    #     train_acc_list.extend([train_accuracy])

    test_acc_list.extend([test_accuracy])
    train_acc_list.extend([train_accuracy])
    test_prob_pll_list.extend([test_pos_prob])
    test_prob_list.extend([test_pos_prob_true])
    train_prob_pll_list.extend([train_pos_prob])
    train_prob_list.extend([train_pos_prob_true])

    if valid_pos_prob > best_valid_pos_prob:
        best_valid_pos_prob = valid_pos_prob
        tolerance = NO_IMPROVEMENT_TOLERANCE
    else:
        tolerance -= 1
    if tolerance <= 0:
        break

avg_test_acc = np.mean(test_acc_list[-10:])
best_test_acc = np.max(test_acc_list)
avg_train_acc = np.mean(train_acc_list[-10:])
avg_test_prob = np.mean(test_prob_list[-10:])
avg_test_prob_pll = np.mean(test_prob_pll_list[-10:])
avg_train_prob = np.mean(train_prob_list[-10:])
avg_train_prob_pll = np.mean(train_prob_pll_list[-10:])

print("Learning Rate:", args.lr, "Weight Decay:", args.wd)
print("Best Test Accuracy: ", best_test_acc)
print("Average Test Accuracy over Last 10 Epochs:", avg_test_acc)
print("Average Test Probability over Last 10 Epochs:", avg_test_prob)
print("Average Test PLL Probability over Last 10 Epochs:", avg_test_prob_pll)
print("Average Train Probability over Last 10 Epochs:", avg_train_prob)
print("Average Train PLL Probability over Last 10 Epochs:", avg_train_prob_pll)
print("Average Training Accuracy over Last 10 Epochs:", avg_train_acc,
      "\n\n\n")

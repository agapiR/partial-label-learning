import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn

from utils.models import mlp_model, deep_linear_model
from utils.utils_loss import (log_prp_Loss as prp_loss, 
                              nll_loss)

"""
Visualizing the optimization trajectories for the first 3 outputs.
"""

## From: https://cs.stackexchange.com/questions/3227/uniform-sampling-from-a-simplex
def rand_simplex(k):
    return tuple(np.random.dirichlet((1,)*k))


OUT = 3
TOL = 1e-6

colormap = {
    0: "r",
    1: "b",
    2: "g"
    }

def regression(model, input, target_output, epochs=100, lr=0.01, weight_decay=0.1):
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    mse = nn.MSELoss()

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(input)
        loss = mse(out, target_output)
        loss.backward()
        optimizer.step()
        accuracy_train = loss.item()
        
    # print("Regression result: {} -> {}".format(input.detach().numpy(force=True), loss.detach().numpy(force=False)))
    return model, accuracy_train


def get_loss(losstype, ys, logits):
    if losstype == "prp":
        loss_fn = prp_loss()
        loss = loss_fn(logits, ys)
    elif losstype == "nll":
        loss_fn = nll_loss()
        loss, _ = loss_fn(logits, ys)
    else:
        assert False, "Unknown loss type."
    return loss


def train(model, input, target, losstype, epochs=100, lr=0.01, weight_decay=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    softmax = nn.Softmax(dim=1)
    model.train()
    probs = []

    # Training Loop
    converged = False
    for _ in range(epochs):
        # Logging
        with torch.no_grad():
            logits = model(input)
            probabilities = softmax(logits).squeeze().tolist()
            probs.append(probabilities) 
            # Check Convergence:
            if len(probs)>1:
                change = np.linalg.norm(np.array(probabilities) - np.array(probs[-2]), ord=1)
                if change < TOL:
                    converged = True

        if converged:
            print("Optimization converged!")
            break
        else:
            # Optimizing
            optimizer.zero_grad()
            out = model(input)
            loss = get_loss(losstype, target, out)
            loss.backward()
            optimizer.step()
    
    training_trajectories = np.array(probs)

    return model, training_trajectories.transpose()



def show_multiple_trajectories(losstypes, outfile, ys_list, in_dim=100, 
                                                            hid_dim=10, 
                                                            out_dim=3, 
                                                            modeltype='linear', 
                                                            epochs=200, 
                                                            learning_rate=0.01, 
                                                            trials=5,
                                                            target_out_probs=None, 
                                                            rand_seeds=None,
                                                            pretrain_trials=3):
    
    if rand_seeds is None:
        rand_seeds = list(np.random.randint(1, size=num_trials))
        #list(range(num_trials))
    if target_out_probs is None:
        target_out_probs = []
        for seed in rand_seeds:
            out_probs = list(rand_simplex(OUT))
            free_probs = list(rand_simplex(out_dim-OUT))
            target_out_probs.append([i / (sum(out_probs) + sum(free_probs)) for i in out_probs + free_probs])
    # print("target_out_probs: {}".format(target_out_probs))

    for i,seed in enumerate(rand_seeds):
        print("Init {}: {} \n".format(i+1, target_out_probs[i]))

    assert trials==len(target_out_probs)
    assert trials==len(rand_seeds)
    
    Y = np.array(ys_list)   # (n_batch * OUT)
    target_full = np.concatenate((Y, np.zeros((Y.shape[0],out_dim-3))), axis=1)     # (n_batch * n_outputs)

    fig, axs = plt.subplots(len(losstypes), 1, figsize=(40,40*len(losstypes)), subplot_kw={"projection": "3d"})

    for i, losstype in enumerate(losstypes):

        probs = dict()

        for j, seed in enumerate(rand_seeds):

            probs[seed] = dict()
            probs[seed].update([(i,[]) for i in range(OUT)])
        
            ## Set random seeds to begin new trial
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            ## Sample random input
            input = np.random.rand(1,in_dim)    # (1 * n_inputs)

            ## Initialize model
            if modeltype=='linear':
                model = deep_linear_model(in_dim, hidden_dim=hid_dim, output_dim=out_dim)
            else:
                model = mlp_model(in_dim, hidden_dim=hid_dim, output_dim=out_dim)

            ## Pretrain to target A/B/C ratio
            # # retrieve logits for target init probabilities
            target_probs = np.array(target_out_probs[j])
            # target_logits = np.log(np.maximum(target_probs,1e-15))
            logits_full = np.log(np.maximum(target_probs,1e-15))
            # # assign random values to the free logits
            # free_logit_sum = target_logits.max()                                    # free logits sum to the min target logit
            # free_logits = np.random.rand(out_dim-3) + 1e-8                          # create random numbers
            # free_logits = free_logits/free_logits.sum() * free_logit_sum            # force them to sum to target_sum
            # logits_full = np.concatenate((target_logits, free_logits), axis=None)   # append free logits to target logits
            # find the best model weights that correspond to the target logits
            mse_best = float('inf')
            model_best = None
            for _ in range(pretrain_trials):
                model, mse = regression(model, torch.FloatTensor(input), torch.FloatTensor(logits_full).unsqueeze(0))
                if mse < mse_best:
                    mse_best = mse
                    model_best = model

            ## Get the optimization trajectory
            _, training_trajectories = train(model_best, torch.FloatTensor(input), torch.FloatTensor(target_full), losstype, epochs=epochs, lr=learning_rate)

            for c in range(OUT):
                probs[seed][c] = training_trajectories[c]
            
               
        # Plot optimization trajectories 2-D (prob / epoch training curves)
        # for c in range(OUT):
        #     for seed in rand_seeds:
        #         axs[i].plot(list(range(1,epochs+1)), probs[seed][c], color=colormap[c])
        # axs[i].set_title(losstype)
        # axs[i].title.set_fontsize(48)

        ## Plot optimization trajectories 3-D
        # cm = plt.get_cmap("RdYlGn")
        for seed in rand_seeds:
            # c = [int(10*(x+y+z)) for x, y, z in zip(probs[seed][0], probs[seed][1], probs[seed][2])]
            # c = np.arange(len(probs[seed][0]))
            axs[i].scatter(probs[seed][0][-1], probs[seed][1][-1], probs[seed][2][-1], marker='o', s=100, c='r')
            axs[i].scatter(probs[seed][0][0], probs[seed][1][0], probs[seed][2][0], marker='*', s=60)
            axs[i].plot(probs[seed][0], probs[seed][1], probs[seed][2])
        
        axs[i].set_title(losstype)
        axs[i].set_xlabel('A', labelpad=20, fontsize=52)
        axs[i].set_ylabel('B', labelpad=20, fontsize=52)
        axs[i].set_zlabel('C', labelpad=20, fontsize=52)
        axs[i].title.set_fontsize(52)

    plt.savefig(outfile)


save_dir = "plots/multi-trajectories"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

    
losstypes=["prp", "nll"]

ys_list = [[1,1,0],[1,0,1]]         #[[1,1,0],[1,0,1],[1,1,0]]
modeltypes = ["mlp", "linear"]
in_dims = [100]                      #[5, 20, 100]
hid_dims = [10]                      #[2, 10, 100]
out_dims = [3, 10]
num_trials = 40
num_epochs = 300
lr = 0.01
for modeltype in modeltypes:
    for in_dim in in_dims:
        for hid_dim in hid_dims:
            for out_dim in out_dims:
                outfile = "{}/multi_trajectories_3d_1AB_1AC_{}_{}_{}_{}_trials-{}_epochs-{}_lr-{}.png".format(save_dir, modeltype, in_dim, hid_dim, out_dim, num_trials, num_epochs, lr)
                print("\n---------------")
                print(outfile)
                show_multiple_trajectories(losstypes, outfile , ys_list=ys_list, in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim, modeltype=modeltype, epochs=num_epochs, learning_rate=lr, trials=num_trials, rand_seeds=list(range(num_trials)))
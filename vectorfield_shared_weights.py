import numpy as np
import matplotlib.pyplot as plt
from egtsimplex import egtsimplex
import re
import torch
import torch.nn as nn

from utils.models import linear_model, mlp_model, deep_linear_model
from utils.utils_loss import (log_prp_Loss as prp_loss, 
                              nll_loss)


nodemap={0:"A", 1:"B", 2:"C"}
colormap = {
    "prp": "red",
    "nll": "b",
    "prp1":"g",
    "prp2": "m",
}

def regression(model, input, target_output, epochs=100, lr=0.01, weight_decay=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(input)
        loss = mse(out, target_output)
        loss.backward()
        optimizer.step()
        accuracy_train = loss.item()

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

def update(losstype, model, input, ys_list, lr=0.1):
    logits = model(input)
    e = np.exp(logits.detach().squeeze().numpy())
    probs = e / np.sum(e)

    loss = 0
    for ys in ys_list:
        loss += get_loss(losstype, ys.unsqueeze(0), logits)
        
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    logits2 = model(input)
    
    e2 = np.exp(logits2.detach().squeeze().numpy())
    probs2 = e2 / np.sum(e2)

    probgrad = (probs2 - probs)

    # print("probs: ", probs)
    # print("probs2: ", probs2)
    # print("probgrad: ", probgrad)
    # print("----------")

    return logits2, probgrad
    
def show_vector_field(losstypes, outfile, ys_list, in_dim=100, hid_dim=10, out_dim=3, modeltype='linear', fit_trials=1, free_logit_trials=1):
    
    # 1. out_dim has to be at least 3
    # 2. if out_dim=3 we do not have free logits 
    if out_dim <= 3:
        out_dim = 3
        free_logit_trials = 1

    input = np.random.rand(1,in_dim)
    Y = np.array(ys_list)   # (n_batch * n_outputs)

    plt.autoscale(False)
    fig, axs = plt.subplots(len(losstypes), 1, figsize=(20,20*len(losstypes)))

    for i, losstype in enumerate(losstypes):

        def f(x,t):
            if np.min(x) <=0:
                return [0.0, 0.0, 0.0]
            
            probs = np.array(x)
            logits = np.log(np.maximum(probs,1e-15))
            if modeltype=='linear':
                model = deep_linear_model(in_dim, hidden_dim=hid_dim, output_dim=out_dim)
            else:
                model = mlp_model(in_dim, hidden_dim=hid_dim, output_dim=out_dim)

            # if out_dim > 3, we need to randomly assign values to other logits, repeat for several trials and average the probgrad
            probgrad_l = []
            
            for _ in range(free_logit_trials):
                # assign random values to the free logits
                target_sum = logits.min()                                       # free logits sum to the min target logit
                free_logits = np.random.rand(out_dim-3) + 1e-8                # create random numbers
                free_logits = free_logits/free_logits.sum() * target_sum        # force them to sum to target_sum
                logits_full = np.concatenate((logits, free_logits), axis=None)  # append free logits to target logits
                target_full = np.concatenate((Y, np.zeros((Y.shape[0],out_dim-3))), axis=1)

                # find the best model weights that correspond to the target logits
                mse_best = float('inf')
                model_best = None
                for _ in range(fit_trials):
                    model, mse = regression(model, torch.FloatTensor(input), torch.FloatTensor(logits_full).unsqueeze(0))
                    if mse < mse_best:
                        mse_best = mse
                        model_best = model

                # get the gradient in probability space
                _, probgrad = update(losstype, model_best, torch.FloatTensor(input), torch.FloatTensor(target_full))
                probgrad_l.append(probgrad)

            # calculate the probability gradient, as average of prob gradients for different free logit assignments
            probgrad = np.average(probgrad_l, axis=0)
            
            return probgrad[:3]
        
        #initialize simplex_dynamics object with function
        dynamics=egtsimplex.simplex_dynamics(f)

        #plot the simplex dynamics
        dynamics.plot_simplex(axs[i])
        axs[i].set_title(losstype)
        axs[i].title.set_fontsize(48)

    plt.savefig(outfile)

    
losstypes=["prp", "nll"]


# TODO: 
# (*) Previously, model dimensions were: 5 - 2 - 3, now default is 100 - 10 - 3
# (*) Previously, we used a vector of ones as input, now we sample a random vector


# ## Single Sample Dataset

# show_vector_field(losstypes,"plots/vectorfields_shared_weights_1AB.png", ys_list=[[1,1,0]])

# ## Consistent Datasets

# show_vector_field(losstypes,"plots/vectorfields_shared_weights_1AB_1AC.png", ys_list=[[1,1,0],[1,0,1]])
# show_vector_field(losstypes,"plots/vectorfields_shared_weights_2AB_1AC.png", ys_list=[[1,1,0],[1,1,0],[1,0,1]])

# ## Inconsistent Datasets

# show_vector_field(losstypes,"plots/vectorfields_shared_weights_3AB_1AC_1BC.png", ys_list=[[1,1,0],[1,1,0],[1,1,0],
#                                                                             [1,0,1],
#                                                                             [0,1,1]])


# show_vector_field(losstypes,"plots/vectorfields_shared_weights_3AB_2AC_1BC.png", ys_list=[[1,1,0],[1,1,0],[1,1,0],
#                                                                             [1,0,1],[1,0,1],
#                                                                             [0,1,1]])


# ## Experiments with more outputs

# show_vector_field(losstypes,
#                 "plots/vectorfields_shared_weights_1AB_1AC_mlp_10_out.png", 
#                 ys_list=[[1,1,0],[1,0,1]],
#                 in_dim=100, hid_dim=10, out_dim=10, modeltype='mlp', 
#                 fit_trials=5, free_logit_trials=5)

# show_vector_field(losstypes,
#                 "plots/vectorfields_shared_weights_1AB_1AC_mlp_3_out.png", 
#                 ys_list=[[1,1,0],[1,0,1]],
#                 in_dim=100, hid_dim=10, out_dim=3, modeltype='mlp', 
#                 fit_trials=5, free_logit_trials=1)

show_vector_field(losstypes,
                "plots/vectorfields_shared_weights_1AB_1AC_mlp_10_out_1_fit_trial.png", 
                ys_list=[[1,1,0],[1,0,1]],
                in_dim=100, hid_dim=10, out_dim=10, modeltype='mlp', 
                fit_trials=2, free_logit_trials=5)

# ## Experiments with more trials + different models + inconsistent datasets

# for model in ['mlp', 'linear']:
#     show_vector_field(losstypes,f"plots/vectorfields_shared_weights_3AB_1AC_1BC_{model}.png", 
#                                 ys_list=[[1,1,0],[1,1,0],[1,1,0],
#                                         [1,0,1],
#                                         [0,1,1]],
#                                 in_dim=100, hid_dim=10, out_dim=3, modeltype=model, 
#                                 fit_trials=5, free_logit_trials=1)

#     show_vector_field(losstypes,f"plots/vectorfields_shared_weights_3AB_2AC_1BC_{model}.png", 
#                                 ys_list=[[1,1,0],[1,1,0],[1,1,0],
#                                 [1,0,1],[1,0,1],
#                                 [0,1,1]],
#                                 in_dim=100, hid_dim=10, out_dim=3, modeltype=model, 
#                                 fit_trials=5, free_logit_trials=1)
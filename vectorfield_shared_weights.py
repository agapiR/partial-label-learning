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
    
def show_vector_field(losstypes, outfile, ys_list, in_dim=5, modeltype='linear'):
    
    plt.autoscale(False)

    input = np.ones((1,in_dim))
    Y = np.array(ys_list)   # (n_batch * n_outputs)

    fig, axs = plt.subplots(len(losstypes), 1, figsize=(20,20*len(losstypes)))

    for i, losstype in enumerate(losstypes):

        def f(x,t):
            if np.min(x) <=0:
                return [0.0, 0.0, 0.0]
            
            probs = np.array(x)
            logits = np.log(np.maximum(probs,1e-15))
            if modeltype=='linear':
                model = deep_linear_model(in_dim, hidden_dim=2, output_dim=3)
            else:
                model = mlp_model(in_dim, hidden_dim=2, output_dim=3)

            # find model weights that correspond to the target logits
            model, mse = regression(model, torch.FloatTensor(input), torch.FloatTensor(logits).unsqueeze(0))
            # print("MSE: ", mse)

            _, probgrad = update(losstype, model, torch.FloatTensor(input), torch.FloatTensor(Y))
            return probgrad
        
        #initialize simplex_dynamics object with function
        dynamics=egtsimplex.simplex_dynamics(f)

        #plot the simplex dynamics
        dynamics.plot_simplex(axs[i])
        axs[i].set_title(losstype)
        axs[i].title.set_fontsize(48)

    plt.savefig(outfile)

    
losstypes=["prp", "nll"]


# Single Sample Dataset

show_vector_field(losstypes,"plots/vectorfields_shared_weights_1AB.png", ys_list=[[1,1,0]])

# Consistent Datasets

show_vector_field(losstypes,"plots/vectorfields_shared_weights_1AB_1AC.png", ys_list=[[1,1,0],[1,0,1]])
show_vector_field(losstypes,"plots/vectorfields_shared_weights_2AB_1AC.png", ys_list=[[1,1,0],[1,1,0],[1,0,1]])

# Inconsistent Datasets

show_vector_field(losstypes,"plots/vectorfields_shared_weights_3AB_1AC_1BC.png", ys_list=[[1,1,0],[1,1,0],[1,1,0],
                                                                            [1,0,1],
                                                                            [0,1,1]])


show_vector_field(losstypes,"plots/vectorfields_shared_weights_3AB_2AC_1BC.png", ys_list=[[1,1,0],[1,1,0],[1,1,0],
                                                                            [1,0,1],[1,0,1],
                                                                            [0,1,1]])

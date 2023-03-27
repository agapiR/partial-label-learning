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
        loss_np = loss.item()
        if loss_np < 0.001:
            break
        
    # print("Regression result: {} -> {}".format(input.detach().numpy(force=True), loss.detach().numpy(force=False)))
    return model, loss_np



def get_loss(losstype, ys, logits):
    if losstype == "prp":
        loss_fn = prp_loss()
        loss = loss_fn(logits, ys)
    elif losstype == "nll":
        loss_fn = nll_loss()
        loss, _ = loss_fn(logits, ys)
    elif losstype == "prp_basic":
        loss_fn = prp_loss(use_weighting=False)
        loss = loss_fn(logits, ys)
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

    return logits2, probgrad, probs, probs2
    
def show_vector_field(losstypes, outfile, ys_list, in_dim=5, hidden_dim=2, modeltype='linear', output_dim=3, samples=100, min_total=0.1, max_total=0.9, hidden_layers=1):

    if output_dim > 3:
        ys_list = [np.pad(ys, (0, output_dim - 3)) for ys in ys_list]

    
    plt.autoscale(False)

    input = np.ones((1,in_dim))
    Y = np.array(ys_list)   # (n_batch * n_outputs)

    fig, axs = plt.subplots(len(losstypes), 1, figsize=(20,20*len(losstypes)))

    for i, losstype in enumerate(losstypes):

        def f(x,t):
            if np.min(x) <= 1e-5:
                return [0.0, 0.0, 0.0]

            if modeltype=='linear':
                model = deep_linear_model(in_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            elif modeltype=='logit':
                model = linear_model(in_dim, output_dim=output_dim)
            else:
                model = mlp_model(in_dim, hidden_dim=hidden_dim, output_dim=output_dim, hidden_layers=hidden_layers)
                
                                  
            if output_dim == 3:
                probs_list = [np.array(x)]
            else: # there are other, unused dimensions and we take a sample
                assert output_dim > 3
                total_probs = np.expand_dims(np.linspace(min_total, max_total, num = samples), 1)
                main_outputs = np.array(x) * total_probs
                other_outputs = np.random.rand(samples, output_dim - 3)
                other_outputs = other_outputs / np.sum(other_outputs, axis=1, keepdims=True) * (1 - total_probs)
                probs_list = np.concatenate([main_outputs, other_outputs], axis=1)
                

            probgrad_all = torch.FloatTensor([0,0,0])
            for probs in probs_list:
                logits = np.log(np.maximum(probs,1e-15))

                # find model weights that correspond to the target logits
                model, mse = regression(model, torch.FloatTensor(input), torch.FloatTensor(logits).unsqueeze(0))
                # print("MSE: {} -> {}".format(probs, mse))

                _, probgrad, probs_before, probs_after = update(losstype, model, torch.FloatTensor(input), torch.FloatTensor(Y))
                probs_before = probs_before[:3]
                probs_after = probs_after[:3]
                probs_before = probs_before / probs_before.sum()
                probs_after = probs_after / probs_after.sum()
                probgrad_all += (probs_after - probs_before)
            probgrad_all /= len(probs_list)
            # print("{} -> {} -> {} -> {}".format(x, probs, probs_before, probs_after))
            return probgrad_all
        
        #initialize simplex_dynamics object with function
        dynamics=egtsimplex.simplex_dynamics(f)

        #plot the simplex dynamics

        if len(losstypes) == 1:
            a = axs
        else:
            a = axs[i]
            
        dynamics.plot_simplex(a)
        # a.set_title(losstype)
        a.title.set_fontsize(48)

    plt.savefig(outfile)


ys_map={
    "1AB": [[1,1,0]],
    "1AB-1AC": [[1,1,0],[1,0,1]],
    }

losstypes=["nll","prp"]
labels = ["1AB-1AC", "1AB"]
modeltypes = ["mlp"]
in_dims = [1]
hidden_dims = [10]
hidden_layers = [1,2,5,10]
output_dims = [4]
samples=1
minmaxes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for loss in losstypes:
    for label in labels:
        ys_list = ys_map[label]
        for modeltype in modeltypes:
            for in_dim in in_dims:
                for hidden_dim in hidden_dims:
                    for hidden_layer in hidden_layers:
                        for output_dim in output_dims:
                            for minmax in minmaxes:
                                outfile = "plots/tmp/vectorfields_{}_{}_{}_{}_{}_{}_{}_{}.png".format(label, modeltype, in_dim, hidden_dim, output_dim, minmax, loss, hidden_layer)
                                print("\n---------------")
                                print(outfile)
                                show_vector_field([loss], outfile , ys_list=ys_list,
                                                  in_dim=in_dim, hidden_dim=hidden_dim,
                                                  modeltype=modeltype, output_dim=output_dim,
                                                  samples=samples, min_total=minmax, max_total=minmax, hidden_layers=hidden_layer)
                            


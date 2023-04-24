import numpy as np
import matplotlib.pyplot as plt
from egtsimplex import egtsimplex
import re
import torch
import torch.nn as nn
import os

from utils.models import linear_model, mlp_model, deep_linear_model
from utils.utils_loss import (log_prp_Loss as prp_loss, 
                              nll_loss, democracy_loss, bi_prp_loss)


nodemap={0:"A", 1:"B", 2:"C"}
colormap = {
    "prp": "red",
    "prp_basic": "coral",
    "nll": "b",
    "uniform": "cyan",
    "bi_prp": "lightsalmon",
    # "prp1":"g",
    # "prp2": "m",
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



def get_loss_fn(losstype):
    if losstype == "prp":
        # loss_fn = prp_loss(use_weighting=True)
        loss_fn = prp_loss(use_weighting=False)
    elif losstype == "nll":
        loss_fn = nll_loss()
    elif losstype == "prp_basic":
        loss_fn = prp_loss(use_weighting=False)
    elif losstype == "uniform":
        loss_fn = democracy_loss()
    elif losstype == "bi_prp":
        # loss_fn = bi_prp_loss()
        loss_fn = bi_prp_loss(use_weighting=False)
    elif losstype == "bi_prp_basic":
        loss_fn = bi_prp_loss(use_weighting=False)
    else:
        assert False, "Unknown loss type." + losstype
    return loss_fn

def get_loss(losstype, ys, logits):
    if losstype == "nll":
        loss_fn = nll_loss()
        loss, _ = loss_fn(logits, ys)
    elif losstype == "prp":
        # loss_fn = prp_loss(use_weighting=True)
        loss_fn = prp_loss(use_weighting=False)
    elif losstype == "nll":
        loss_fn = nll_loss()
    elif losstype == "prp_basic":
        loss_fn = prp_loss(use_weighting=False)
    elif losstype == "uniform":
        loss_fn = democracy_loss()
    elif losstype == "bi_prp":
        # loss_fn = bi_prp_loss()
        loss_fn = bi_prp_loss(use_weighting=False)
    elif losstype == "bi_prp_basic":
        loss_fn = bi_prp_loss(use_weighting=False)
    else:
        assert False, "Unknown loss type." + losstype
    loss = loss_fn(logits, ys)
    return loss

def update(losstype, model, input, ys_list, lr=0.1):
    logits = model(input)
    e = np.exp(logits.detach().squeeze().numpy())
    probs = e / np.sum(e)

    loss = 0
    for ys in ys_list:
        curr_loss = get_loss(losstype, ys.unsqueeze(0), logits)
        # curr_loss = loss_fn(ys.unsqueeze(0), logits)
        if type(curr_loss) is tuple:
            loss +=  curr_loss[0]
        else:
            loss += curr_loss
        
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

    fig, axs = plt.subplots(len(losstypes), 1, figsize=(13*len(losstypes), 13))

    for i, losstype in enumerate(losstypes):

        # loss_fn = get_loss_fn(losstype)

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
        a.set_title(losstype)
        a.title.set_fontsize(30)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    plt.savefig(outfile)


ys_map={
    "1AB": [[1,1,0]],
    "1AB-1AC": [[1,1,0],[1,0,1]],
    }

def run(config):
    for loss in config["losstypes"]:
        for label in config["labels"]:
            ys_list = ys_map[label]
            for modeltype in config["modeltypes"]:
                for in_dim in config["in_dims"]:
                    for hidden_dim in config["hidden_dims"]:
                        for hidden_layer in config["hidden_layers"]:
                            for output_dim in config["output_dims"]:
                                for minmax in config["minmaxes"]:
                                    exp = config["exp"]
                                    samples = config["samples"]
                                    outfile = "plots/vectorfields/exp{}/vectorfields_{}_{}_{}_{}_{}_{}_{}_{}.png".format(exp, label, modeltype, in_dim, hidden_dim, output_dim, minmax, loss, hidden_layer)
                                print("\n---------------")
                                print(outfile)
                                show_vector_field([loss], outfile , ys_list=ys_list,
                                                  in_dim=in_dim, hidden_dim=hidden_dim,
                                                  modeltype=modeltype, output_dim=output_dim,
                                                  samples=samples, min_total=minmax, max_total=minmax, hidden_layers=hidden_layer)
                            


config1 = {
    "exp": 1,
    "losstypes": ["prp", "bi_prp", "nll","uniform"],
    "labels": ["1AB-1AC"],
    "modeltypes": ["logit"],
    "in_dims": [1],
    "hidden_dims": [0],
    "hidden_layers": [0],
    "output_dims": [3],
    "samples":1,
    "minmaxes":[0.1],
}

config2 = {
    "exp": 2,
    "losstypes": ["nll","uniform", "prp", "bi_prp"],
    "labels": ["1AB"],
    "modeltypes": ["mlp"],
    "in_dims": [1],
    "hidden_dims": [100],
    "hidden_layers": [0,1,3,5,10],
    "output_dims": [3],
    "samples":1,
    "minmaxes":[0.1],
}

config3 = {
    "exp": 3,
    "losstypes": ["prp", "bi_prp", "nll","uniform"],
    "labels": ["1AB-1AC"],
    "modeltypes": ["mlp"],
    "in_dims": [1],
    "hidden_dims": [100],
    "hidden_layers": [1,3,5,10],
    "output_dims": [3],
    "samples":1,
    "minmaxes":[0.1],
}

run(config1)
run(config2)
run(config3)


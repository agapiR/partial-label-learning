import numpy as np
import matplotlib.pyplot as plt

# three outputs, first two are allowed

def get_grads(losstype, ys, probs):
    k = np.sum(ys)
    sumprobs = np.sum(probs * ys)
    if losstype == "prp":
        grads = np.ones(len(ys))
        pos_indices = np.where(ys)
        neg_indices = np.where(1-ys)
        grads[pos_indices] = -1 / k
        grads[neg_indices] = probs[neg_indices] / (1-sumprobs)
    elif losstype =="prp1":
        grads = np.zeros(len(ys))
        for j in range(len(ys)):
            for i in range(len(ys)):
                grads[j] += - 1/(1-sumprobs) * ys[i] * probs[i] * ((i==j) - probs[j])               
    elif losstype =="prp2":
        grads = np.zeros(len(ys))
        for j in range(len(ys)):
            for i in range(len(ys)):
                grads[j] += - 1/k * ys[i] * ((i==j) - probs[j])
    elif losstype == "nll":
        grads = np.zeros(len(ys))
        for j in range(len(ys)):
            for i in range(len(ys)):
                grads[j] += - 1/sumprobs * ys[i] * probs[i] * ((i==j) - probs[j])

    return grads

def update(losstype, logits, ys, lr=0.1):
    e = np.exp(logits)
    probs = e / np.sum(e)

    grads = get_grads(losstype, ys, probs)

    logits2 = logits - lr * grads
    e2 = np.exp(logits2)
    probs2 = e2 / np.sum(e2)

    probgrad = (probs2 - probs)
    # probgrad /= np.sum(np.abs(probgrad))
    print("grads: ", grads)
    print("probs: ", probs)
    # print("probs2: ", probs2)
    print("probgrad: ", probgrad)
    print("----------")

    return logits2, probgrad
    
def logits_on_horizontal_line(probC):
    logitA = 0
    logitBs = np.linspace(-10, 10, 100)
    eA = np.exp(logitA)
    eBs = np.exp(logitBs)
    eCs = probC / (1-probC) * (eA + eBs)
    logitCs = np.log(eCs)
    return [(logitA, logitB, logitC) for logitB, logitC in zip(logitBs, logitCs)]

def show_horizontal_line(losstypes, probC, outfile):
    ys = np.array([1,1,0])
    logits_list = logits_on_horizontal_line(probC)
    result = {}
    for losstype in losstypes:
        result[losstype] = [[],[],[]]
    for logits in logits_list:
        for losstype in losstypes:
            print("--------{}----------".format(losstype))
            _, probgrad = update(losstype, logits, ys)
            result[losstype][0].append(probgrad[0])
            result[losstype][1].append(probgrad[1])
            result[losstype][2].append(probgrad[2])

    fig, axs = plt.subplots(3, 1, figsize=(20,60))
    nodemap={0:"A", 1:"B", 2:"C"}
    colormap = {
        "prp": "red",
        "nll": "b",
        "prp1":"g",
        "prp2": "m",
    }

    for i in range(3):
        for losstype in losstypes:
            color = colormap[losstype]
            axs[i].plot(result[losstype][i], label="{}_{}".format(losstype, nodemap[i]), color=color, marker='o')
        axs[i].set_title("Prob C is fixed ({}), moving probability from A towards B".format(probC))
        axs[i].legend()
        axs[i].set_xlabel("Horzontal line in the triangle")
        axs[i].set_ylabel("Update magnitude in the probability of {}".format(nodemap[i]))
        axs[i].title.set_fontsize(48)
        axs[i].xaxis.label.set_fontsize(28)
        axs[i].yaxis.label.set_fontsize(28)

    plt.savefig(outfile)

def show vector_field(losstypes, outfile):
    ys = np.array([1,1,0])

logits = np.array([1, 0.5, 10])
ys = np.array([1,1,0])
losstypes=["prp1", "prp2", "prp", "nll"]

# for i in range(100):
#     logits = update(losstype, logits, ys)

for probC in [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    outfile = "plots/horizontal_line_probC-{}.png".format(probC)
    show_horizontal_line(losstypes, probC, outfile)

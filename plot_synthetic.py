import re
import os
import matplotlib.pyplot as plt
from statistics import median
from itertools import product


directory = "results_synthetic"
prefices = ["synthetic"]

# "${OUT_PATH} / 
# synthetic_${sample_num}_${class_num}_${feature_num}_${prt}_${csep}_${dseed}_${model}_${loss}_${seed}.out"
# result[sample_num][class_num][feature_num][prt][csep][dseed][model][loss][seed]


## - Parse Results
result = {}
config = {}
args = ['sample_num', 'class_num', 'feature_num', 'partial_rate', 'class_sep', 'dseed', 'model', 'loss', 'seed']
config.update([(arg, set()) for arg in args])
for prefix in prefices:
    for filename in os.listdir(directory):
        match = re.match("{}_([0-9]+)_([0-9]+)_([0-9]+)_([0-9].[0-9]+)_([0-9].[0-9]+)_([0-9]+)_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_([0-9]+).out".format(prefix), filename)
        if match:
            g = match.groups()
            # dataset arguments:
            sample_num = g[0]
            class_num = g[1]
            feature_num = g[2]
            partial_rate = g[3]
            class_sep = g[4]
            dseed = g[5]
            # print("Dataset: ", sample_num, class_num, feature_num, partial_rate, class_sep, dseed)
            # experiment arguments
            model = g[6]
            loss = g[7]
            seed = g[8]
            # print("Experiment: ", model, loss, seed)

            for i, arg in enumerate(args):
                config[arg].add(g[i])

            f = os.path.join(directory, filename)
            with open(f) as fff:
                for line in fff:
                    match = re.match("^Average Training Accuracy over Last 10 Epochs: (.*)$", line)
                    # result[sample_num][class_num][feature_num][prt][csep][dseed][model][loss][seed]
                    if match:
                        test_accuracy = match.groups()[0]
                        key = f'{sample_num}_{class_num}_{feature_num}_{partial_rate}_{class_sep}_{dseed}_{model}_{loss}_{seed}'
                        result[key] = test_accuracy
                        # print("Accuracy: ", test_accuracy)

## - Make Plots

# Hyperparams:
sample_num = 1000
class_num = 10
feature_num = 5
dseed_default = 42
seed_default = 5

partial_rates = sorted([float(prt) for prt in config['partial_rate'] if float(prt)>0])
# partial_rates = sorted([float(prt) for prt in config['partial_rate']])


# Accuracy Comparison for random Dataset Samples
cmap = plt.get_cmap("tab10")
for model, class_sep in product(config['model'], config['class_sep']):
    outfile = "figs/train_accuracy_comparison_for_random_datasets_{}_{}_{}_{}_{}_{}.png".format(sample_num, class_num, feature_num, class_sep, model, seed_default)
    for i,loss in enumerate(config['loss']):
        median_accuracies = []
        max_accuracies = []
        for prt in partial_rates:
            accuracies = []
            for dseed in config['dseed']:
                key = f'{sample_num}_{class_num}_{feature_num}_{prt}_{class_sep}_{dseed}_{model}_{loss}_{seed_default}'
                accuracies.append(float(result[key]))
            median_accuracies.append(median(accuracies))
            max_accuracies.append(max(accuracies))
        plt.plot(partial_rates, median_accuracies, label=loss+"-median", marker='o', color=cmap(i))
        plt.plot(partial_rates, max_accuracies, label=loss+"-max", marker='x', color=cmap(i), linestyle='dashed')
    plt.legend()
    plt.savefig(outfile)
    plt.clf()

# Accuracy Comparison for random initializations
cmap = plt.get_cmap("tab10")
for model, class_sep in product(config['model'], config['class_sep']):
    outfile = "figs/train_accuracy_comparison_for_random_init_{}_{}_{}_{}_{}_{}.png".format(sample_num, class_num, feature_num, class_sep, dseed_default, model)
    for i,loss in enumerate(config['loss']):
        median_accuracies = []
        max_accuracies = []
        for prt in partial_rates:
            accuracies = []
            for seed in config['seed']:
                key = f'{sample_num}_{class_num}_{feature_num}_{prt}_{class_sep}_{dseed_default}_{model}_{loss}_{seed}'
                accuracies.append(float(result[key]))
            median_accuracies.append(median(accuracies))
            max_accuracies.append(max(accuracies))
        plt.plot(partial_rates, median_accuracies, label=loss+"-median", marker='o', color=cmap(i))
        plt.plot(partial_rates, max_accuracies, label=loss+"-max", marker='x', color=cmap(i), linestyle='dashed')
    plt.legend()
    plt.savefig(outfile)
    plt.clf()

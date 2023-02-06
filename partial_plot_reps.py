import re
import os
import matplotlib.pyplot as plt
from statistics import median


directory = "out"
prefices = ["hypercube"]

result = {}
max_rep = 1
for prefix in prefices:
    for filename in os.listdir(directory):
        match = re.match("{}_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(.+)_([0-9].[0-9]+)_rep_([0-9]+).out".format(prefix), filename)
        if match:
            g = match.groups()
            model = g[0]
            dataset = g[1]
            loss = g[2]
            partial_rate = g[3]
            repetition = int(g[4])
            if repetition>max_rep:
                max_rep=repetition
            # print(filename, model, dataset, loss, partial_rate, repetition)

            f = os.path.join(directory, filename)
            with open(f) as fff:
                for line in fff:
                    match = re.match("^Average Test Accuracy over Last 10 Epochs: (.*)$", line)
                    if match:
                        test_accuracy = match.groups()[0]
                        if prefix not in result:
                            result[prefix] = {}
                        if dataset not in result[prefix]:
                            result[prefix][dataset] = {}
                        if model not in result[prefix][dataset]:
                            result[prefix][dataset][model] = {}
                        if loss not in result[prefix][dataset][model]:
                            result[prefix][dataset][model][loss] = {}
                        if partial_rate not in result[prefix][dataset][model][loss]:
                            result[prefix][dataset][model][loss][partial_rate] = [test_accuracy]
                        else:
                            result[prefix][dataset][model][loss][partial_rate].append(test_accuracy)


# plot test accuracies
cmap = plt.get_cmap("tab10")
for prefix in result.keys():
    d1 = result[prefix]
    for dataset in d1.keys():
        d2 = d1[dataset]
        for model in d2.keys():
            print("\n", prefix, dataset, model)
            d3 = d2[model]
            outfile = "plots/{}_{}-0.2_{}_{}-reps.png".format(prefix, dataset, model, max_rep)
            for i,loss in enumerate(d3.keys()):
                d4 = d3[loss]
                partial_rates = d4.keys()
                accuracies = d4.values()
                sorted_pairs = sorted(zip(partial_rates, accuracies))
                partial_rates = [round(float(x),2) for x, _ in sorted_pairs]
                accuracies_sorted = [[float(acc) for acc in x] for _, x in sorted_pairs]
                median_accuracies = [round(median(x),2) for x in accuracies_sorted]
                max_accuracies = [round(max(x),2) for x in accuracies_sorted]
                print("   ", loss, partial_rates, median_accuracies, max_accuracies)
                plt.plot(partial_rates, median_accuracies, label=loss+"-median", marker='o', color=cmap(i))
                plt.plot(partial_rates, max_accuracies, label=loss+"-max", marker='x', color=cmap(i), linestyle='dashed')
            plt.legend()
            plt.savefig(outfile)
            plt.clf()

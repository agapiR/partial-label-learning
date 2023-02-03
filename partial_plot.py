import re
import os
import matplotlib.pyplot as plt

directory = "out"
prefices = ["uniform", "random", "huniform", "h10uniform", "h5uniform", "h4uniform"]

result = {}
for prefix in prefices:
    for filename in os.listdir(directory):
        match = re.match("{}_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)_(.+)_([0-9].[0-9]+).out".format(prefix), filename)
        if match:
            g = match.groups()
            model = g[0]
            dataset = g[1]
            loss = g[2]
            partial_rate = g[3]
            # print(filename, model, dataset, loss, partial_rate)

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
                            result[prefix][dataset][model][loss][partial_rate] = test_accuracy


# plot test accuracies
for prefix in result.keys():
    d1 = result[prefix]
    for dataset in d1.keys():
        d2 = d1[dataset]
        for model in d2.keys():
            print("\n", prefix, dataset, model)
            d3 = d2[model]
            outfile = "plots/{}_{}_{}.png".format(prefix, dataset, model)
            for loss in d3.keys():
                d4 = d3[loss]
                partial_rates = d4.keys()
                accuracies = d4.values()
                sorted_pairs = sorted(zip(partial_rates, accuracies))
                partial_rates = [round(float(x),2) for x, _ in sorted_pairs]
                accuracies = [round(float(x),2) for _, x in sorted_pairs]
                print("   ", loss, partial_rates, accuracies)
                plt.plot(partial_rates, accuracies, label=loss, marker='o')
            plt.legend()
            plt.savefig(outfile)
            plt.clf()

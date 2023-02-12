import re
import os
import matplotlib.pyplot as plt
from statistics import median
import math

metrics=["Test Accuracy", "Test Prob", "Test Prob PLL", "Train Prob", "Train Prob PLL"]

metric_map={
    "Test Accuracy": "Average Test Accuracy over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob": "Average Test Probability over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob PLL": "Average Test PLL Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob": "Average Train Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob PLL": "Average Train PLL Probability over Last 10 Epochs: (\d+\.\d*)",
}

color_map = {
    "prp": "red",
    "nll": "b",
    "cc":"b",
    "lws": "g",
    "democracy":"c",
    "bi_prp":"y",
    "bi_prp2":"y",
    "rc":"k",
    "bi_prp_nll":"m"
    }

def plot(directory, series, x_axis, outdir, metrics):
    result = {}
    for m in metrics:
        result[m] = {}
    for filename in os.listdir(directory):
        match = re.match("experiment_(.*).out", filename)
        if match:
            args = match.groups()[0]
            args = args.split('-')[1:]
            argdict = {}
            for i in range(0, len(args), 2):
                argdict[args[i]] = args[i+1].rstrip('_')
            result_s = argdict[series]
            result_x = float(argdict[x_axis])

            for m in metrics:
                if result_s not in result[m]:
                    result[m][result_s] = {}
                if result_x not in result[m][result_s]:
                    result[m][result_s][result_x] = []


            f = os.path.join(directory, filename)
            with open(f) as fff:
                line = fff.read()
            for m in metrics:
                match = re.findall(metric_map[m], line)
                if match:
                    value = round(float(match[-1]), 2)
                    result[m][result_s][result_x].append(value)


    # create plot
    fig, axs = plt.subplots(3, math.ceil(len(metrics)/3), figsize=(20, 20))
    for i, m in enumerate(metrics):
        for s in result[m].keys():
            color = color_map[s]
            xs = result[m][s].keys()
            ys = result[m][s].values()
            sorted_pairs = sorted(zip(xs, ys))
            xs = [x for x, _ in sorted_pairs]
            ys  = [x for _, x in sorted_pairs]
        
            ys_max = [max(y) for y in ys]
            ys_med = [round(median(y),2) for y in ys]
        
            print("   ", s, xs, ys_max, ys_med, color)
            
            axs[i%3, i//3].plot(xs, ys_med, color=color, label=s+"-median", marker='o')
            axs[i%3, i//3].plot(xs, ys_max, color=color, label=s+"-max", marker='o', linestyle='dashed')
            axs[i%3, i//3].set_title(m)
            axs[i%3, i//3].title.set_fontsize(48)
            axs[i%3, i//3].xaxis.label.set_fontsize(28)
            axs[i%3, i//3].yaxis.label.set_fontsize(28)
            axs[i%3, i//3].legend()
    outfile = "{}/plot_{}_{}.png".format(outdir, series, x_axis)
    plt.savefig(outfile)
    # plt.clf()


directory = "out/zs1"
series = "lo"
x_axis = "prt"
outdir="plots"
plot(directory, series, x_axis, outdir, metrics)

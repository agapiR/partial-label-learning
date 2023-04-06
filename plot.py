import re
import os
import matplotlib.pyplot as plt
from statistics import median
import math

metrics=["Test Prob", "Test Prob PLL", "Test Accuracy", "Train Prob", "Train Prob PLL", "Train Accuracy"]

metric_map={
    "Test Accuracy": "Average Test Accuracy over Last 10 Epochs: (\d+\.\d*)",
    "Train Accuracy": "Average Training Accuracy over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob": "Average Test Probability over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob PLL": "Average Test PLL Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob": "Average Train Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob PLL": "Average Train PLL Probability over Last 10 Epochs: (\d+\.\d*)",
}

is_float = {
    "prt": True,
    "bs": True,
    "nc": True,
    "mo": False,
    "nf": True,
    "ns": True,
    "csep": True,
    "lr": True,
    "num_groups": True,
}

color_map = {
    "prp": "red",
    "prp_basic": "c",
    "nll": "b",
    "cc":"b",
    "lws": "g",
    "democracy":"c",
    "bi_prp":"y",
    "bi_prp2":"y",
    "rc":"k",
    "bi_prp_nll":"m"
    }

def plot(directory, series, x_axis, outdir, metrics, filtermap={}, prefix=""):
    result = {}
    for m in metrics:
        result[m] = {}
    for filename in os.listdir(directory):
        match = re.match("experiment_(.*).out", filename)
        if match:
            args = match.groups()[0]
            args = args.split('-')[1:]
            argdict = {}
            skipfile = False
            for i in range(0, len(args), 2):
                argdict[args[i]] = args[i+1].rstrip('_')
                if args[i] in filtermap and argdict[args[i]] != filtermap[args[i]]:
                    # print(args[i], argdict[args[i]], filtermap[args[i]])
                    skipfile=True
            if skipfile:
                continue

            result_s = argdict[series]
            result_x = argdict[x_axis]
            if is_float[x_axis]:
                result_x = float(result_x)

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

            print("   {} {}\n          -> {}".format(s, xs, ys))
            ys_max = [max(y, default=0.0) for y in ys]
            ys_med = [round(median(y),2) if len(y) > 0 else 0.0 for y in ys]
        
            print("   ", s, xs, ys_max, ys_med, color)
            
            axs[i%3, i//3].plot(xs, ys_med, color=color, label=s+"-median", marker='o')
            axs[i%3, i//3].plot(xs, ys_max, color=color, label=s+"-max", marker='o', linestyle='dashed')
            axs[i%3, i//3].set_title(m)
            axs[i%3, i//3].title.set_fontsize(48)
            axs[i%3, i//3].xaxis.label.set_fontsize(28)
            axs[i%3, i//3].yaxis.label.set_fontsize(28)
            axs[i%3, i//3].legend()
    
    outfile = "{}/plot_{}{}_{}.png".format(outdir, prefix, series, x_axis)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outfile)
    # plt.clf()


def exp14():
    directory = "out/zs14"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/cifar100_groups"
    for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.0"]:        
        filtermap = {
            "prt":prt,
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    filtermap = {
    }
    x_axes = ["prt"]
    outdir="plots/cifar100_groups"
    for num_groups in ["1", "2", "4", "5", "10", "20"]:        
        filtermap = {
            "num_groups":num_groups,
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))

def exp15():
    directory = "out/zs15"
    series = "lo"
    x_axes = ["lr"]
    outdir="plots/cifar100_biprp_lr"
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)


def exp16():
    directory = "out/zs16"
    series = "lo"
    x_axes = ["prt"]
    outdir="plots/zs16_cifar10"
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)

def exp17():
    directory = "out/zs17"
    outdir="plots/cifar100_resnet"
    series = "lo"
    
    x_axes = ["num_groups"]
    for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.0"]:        
        filtermap = {
            "prt":prt,
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    x_axes = ["prt"]
    for num_groups in ["1", "2", "4", "5", "10", "20"]:        
        filtermap = {
            "num_groups":num_groups,
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))

def exp19():
    directory = "out/zs19"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs19_cifar100_resnet18"
    for prt in ["0.0", "0.05", "0.1", "0.3", "0.6"]:        
        filtermap = {
            "prt":prt,
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))

def exp20():
    directory = "out/zs20"
    series = "lo"
    x_axes = ["lr"]
    outdir="plots/zs20_cifar100_resnet50_lr"
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)

exp19()

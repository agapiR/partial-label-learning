import re
import os
import matplotlib.pyplot as plt
from statistics import median
import numpy as np
import math
import re
import itertools

metrics=["Test Prob", "Test Prob PLL", "Test Accuracy", "Train Prob", "Train Prob PLL", "Train Accuracy"]

metric_map={
    "Test Accuracy old": "Average Test Accuracy over Last 10 Epochs: (\d+\.\d*)",
    "Train Accuracy": "Average Training Accuracy over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob": "Average Test Probability over Last 10 Epochs: (\d+\.\d*)",
    "Test Prob PLL": "Average Test PLL Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob": "Average Train Probability over Last 10 Epochs: (\d+\.\d*)",
    "Train Prob PLL": "Average Train PLL Probability over Last 10 Epochs: (\d+\.\d*)",
    "Test Accuracy": "Best Test Accuracy:  (\d+\.\d*)",
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
    "pr": False,
    "seed": True,
    "dseed": True,
    "distractionbased_ratio": True,
    "logit_decay": True,
    "case": False,
}

def pr_map(pr):
    result = {}
    
    huniform_match = re.match("h([0-9]*)uniform_(.*)", pr)
    uniform_match = re.match("uniform_(.*)", pr)

    if pr == "01":
        result["case"] = "Case 1"
    elif pr == "02":
        result["case"] = "Case 2"
    elif pr == "04":
        result["case"] = "Case 3"
    elif pr == "03":
        result["case"] = "Case 4"
    elif pr == "10":
        result["case"] = "Case 5"
    elif huniform_match:
        result["prt"] = float(huniform_match.groups()[1])
        result["num_groups"] = float(huniform_match.groups()[0])
    else:
        assert False, "Unhandled pr value: " + pr

    return result


color_map = {
    "prp": "red",
    "prp_basic": "coral",
    "bi_prp":"lightsalmon",
    "bi_prp2":"lightsalmon",
    "nll": "b",
    "cc":"b",
    "lws": "g",
    "democracy":"cyan",
    "rc":"k",
    "bi_prp_nll":"m",
    "ll":"lime",
    "merit":"blue",
    }

name_map = {
    "prp": r'Libra',
    "prp_basic": r'Libra',
    "bi_prp": r'Sag',
    "bi_prp2": r'Sag',
    "nll": r'NLL',
    "cc": r'NLL',
    "lws": r'lws',
    "democracy": r'uniform',
    "rc":r'RC',
    "bi_prp_nll":r'Sag-NLL',
    "ll":r'll',
    "merit":r'merit',
    }

marker_map = {
    "prp": 'o',
    "prp_basic": 'o',
    "bi_prp": 'v',
    "bi_prp2": 'v',
    "nll": 's',
    "cc": 's',
    "lws": 'p',
    "democracy": '1',
    "rc":'P',
    "bi_prp_nll":'v',
    "ll":'x',
    "merit": '1',
    }

def order_keys(keys):
    ranking = {
        "prp": 1,
        "prp_basic": 2,
        "bi_prp": 5,
        "bi_prp2": 6,
        "nll": 8,
        "cc": 9,
        "lws": 7,
        "democracy": 3, 
        "rc": 4, 
        "bi_prp_nll": 10,
        "ll": 11,
        "merit": 12,
    }
    keys = [x for _,x in  sorted([(ranking[k], k) for k in keys])]
    return keys

def plot(directory, series, x_axis, outdir, metrics, filtermap={}, prefix="", title="", xlabel="",show_max=True, return_metrics=False):
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
                key = args[i]
                value = args[i+1].rstrip('_')
                
                argdict[key] = value
                if key in filtermap and argdict[key] not in filtermap[key]:
                    skipfile=True
                if key == "pr":
                    pr_dict = pr_map(value)
                    for k in pr_dict:
                        argdict[k] = pr_dict[k]
                        if k in filtermap and k not in filtermap[k]:
                            skipfile=True
                            
                    # prt, num_groups = pr_map(value)
                    # argdict["prt"] = prt
                    # argdict["num_groups"] = num_groups
                    # if "prt" in filtermap and prt not in float(filtermap["prt"]):
                    #     skipfile=True
                    # if "num_groups" in filtermap and num_groups not in float(filtermap["num_groups"]):
                    #     skipfile=True
                    
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
    size = math.ceil(len(metrics)/3) * 10
    if len(metrics) >= 3:
        fig, axs = plt.subplots(3, math.ceil(len(metrics)/3), figsize=(size, size))
    else:
        fig, axs = plt.subplots(1, len(metrics), figsize=(size, size))
    for i, m in enumerate(metrics):
        keys = result[m].keys()
        keys = order_keys(keys)
        for s in keys:
            if series == "lo":
                color = color_map[s]
                label = name_map[s]
                marker = marker_map[s]
            else:
                color = None
                label = s
                marker='o'
            xs = result[m][s].keys()
            ys = result[m][s].values()
            sorted_pairs = sorted(zip(xs, ys))
            xs = [x for x, _ in sorted_pairs]
            ys  = [x for _, x in sorted_pairs]

            print("   {} {}\n          -> {}".format(s, xs, ys))
            ys_mean = [np.mean(y) for y in ys]
            ys_errors = [np.array(y).std() for y in ys]
            ys_max = [max(y, default=0.0) for y in ys]
            ys_med = [round(median(y),2) if len(y) > 0 else 0.0 for y in ys]
        
            print("   ", s, xs, ys_max, ys_med, color)

            if len(metrics) == 1:
                axis = axs
            elif len(metrics) <= 3:
                axis = axs[i]
            else:
                axis= axs[i%3, i//3]

            if show_max:
                axis.plot(xs, ys_med, color=color, label=label+"-median", marker='o')
                axis.plot(xs, ys_max, color=color, label=label+"-max", marker='o', linestyle='dashed')
            else:
                print(label)
                axis.plot(xs, ys_mean, color=color, label=label, marker=marker)
                
                
            # axs[i%3, i//3].errorbar(xs, ys_mean, yerr=ys_errors, color=color, elinewidth=3, label=s, marker='o')
        axis.set_title(title)
        axis.title.set_fontsize(40)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(m)
        axis.xaxis.label.set_fontsize(40)
        axis.yaxis.label.set_fontsize(40)
        axis.legend(prop={'size': 20})
        axis.grid(color='grey', linestyle='-', linewidth=0.3)
    
    outfile = "{}/plot_{}{}_{}.pdf".format(outdir, prefix, series, x_axis)
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(outfile)
    # plt.clf()

    if return_metrics:
        return result


def exp14():
    directory = "out/zs14"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs14_cifar100_groups"
    for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.0"]:        
        filtermap = {
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    x_axes = ["prt"]
    for num_groups in ["1", "2", "4", "5", "10", "20"]:        
        filtermap = {
            "num_groups":[num_groups],
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
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    x_axes = ["prt"]
    for num_groups in ["1", "2", "4", "5", "10", "20"]:        
        filtermap = {
            "num_groups":[num_groups],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))

def exp18():
    directory = "out/zs18"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs18_cifar100_adam"
    for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1", "0.0"]:        
        filtermap = {
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))

def exp19():
    directory = "out/zs19"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs19_cifar100_resnet18"
    for prt in ["0.0", "0.05", "0.1", "0.3", "0.6"]:        
        filtermap = {
            "prt":[prt],
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

def exp21():
    directory = "out/zs21"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs21_shierarchy32"
    for prt in ["0.9", "0.7", "0.5", "0.3", "0.2", "0.1", "0.0"]:        
        filtermap = {
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    filtermap = {
    }
    x_axes = ["prt"]
    outdir="plots/zs21_shierarchy32"
    for num_groups in ["1", "2", "4", "8", "16", "32"]:        
        filtermap = {
            "num_groups":[num_groups],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))


def exp22():
    directory = "out/zs22"
    series = "lo"
    for csep in ["0.5", "1.0", "1.5", "2.0"]:        
        outdir="plots/zs22_shierarchy32_csep_{}".format(csep)
        x_axes = ["num_groups"]
        for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.4", "0.3", "0.2", "0.1"]:
            filtermap = {
                "prt":[prt],
            }
            for x_axis in x_axes:
                plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))    
        filtermap = {
        }
        x_axes = ["prt"]
        for num_groups in ["1", "2", "4", "8", "16"]:        
            filtermap = {
                "num_groups":[num_groups],
            }
            for x_axis in x_axes:
                plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))


def exp23():
    directory = "out/zs23"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs23_cifar100"
    for prt in ["0.9", "0.6", "0.3", "0.0"]:        
        filtermap = {
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    x_axes = ["prt"]
    for num_groups in ["1", "2", "4", "5", "10", "20"]:        
        filtermap = {
            "num_groups":[num_groups],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))

def exp24():
    directory = "out/zs24"
    series = "lo"
    x_axes = ["num_groups"]
    outdir="plots/zs24_cifar100_uniform"

    for prt in ["0.2", "0.1", "0.05"]:        
        filtermap = {
            "prt":[prt],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt))
        
    x_axes = ["prt"]
    for num_groups in ["1", "2", "4", "5", "10", "20", "50"]:        
        filtermap = {
            "num_groups":[num_groups],
        }
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="groups-{}_".format(num_groups))

def exp25():
    directory = "out/zs25"
    series = "lo"
    outdir="plots/zs25_synthetic_distractionbased"
    filtermap = {
        "lr":["0.001"],
    }
    x_axes = ["prt", "lr", "seed", "dseed"]
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)


def exp26():
    directory = "out/zs26"
    series = "lo"
    outdir="plots/zs26_synthetic_distractionbased"
    x_axes = ["prt"]
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)

def exp27():
    directory = "out/zs27"
    series = "lo"
    outdir="plots/zs27_synthetic_distractionbased"
    metrics=["Test Accuracy"]
    x_axes = ["distractionbased_ratio"]
    for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.3", "0.2", "0.1", "0.05"]:        
        filtermap = {
            "lo": ["prp", "nll", "lws", "rc", "democracy"],
            "distractionbased_ratio": ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'],
            "prt":[prt],
        }
        title = "p-rate = {}".format(prt)
        xlabel = "s-rate"
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt), title=title, xlabel=xlabel)

def exp28():
    directory = "out/zs28"
    series = "lo"
    outdir="plots/zs28_biprp"
    x_axes = ["logit_decay"]
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics)

def exp29():
    directory = "out/zs29"
    series = "lo"
    outdir="plots/zs29_synthetic_distractionbased"
    metrics=["Test Accuracy"]
    # x_axes = ["distractionbased_ratio"]
    # for prt in ["0.9", "0.8", "0.7", "0.6", "0.5", "0.3", "0.2", "0.1", "0.05"]:        
    #     filtermap = {
    #         "lo": ["prp", "nll", "lws", "rc", "democracy", "bi_prp"],
    #         "distractionbased_ratio": ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'],
    #         "prt":[prt],
    #     }
    #     title = r'$r_{{Dpool}} = {}$'.format(prt)
    #     xlabel = r'$r_{{Docc}}$'
    #     for x_axis in x_axes:
    #         plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt), title=title, xlabel=xlabel)

    x_axes = ["prt"]
    for distractionbased_ratio in ["0.1", "0.3", "0.5", "0.7", "0.9"]:        
        filtermap = {
            "lo": ["prp", "nll", "lws", "rc", "democracy", "bi_prp"],
            "distractionbased_ratio":[distractionbased_ratio],
        }
        title = r'$r_{{Docc}} = {}$'.format(distractionbased_ratio)
        xlabel = r'$r_{{Dpool}}$'
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="dist-{}_".format(distractionbased_ratio), title=title, xlabel=xlabel, show_max=False)

def exp30():
    directory = "out/zs30"
    series = "lo"
    outdir="plots/zs30_cifar100"
    metrics=["Test Accuracy"]
    # x_axes = ["distractionbased_ratio"]
    # for prt in ["0.9", "0.7", "0.5", "0.3", "0.1"]:        
    #     filtermap = {
    #         "lo": ["prp", "nll", "lws", "rc", "democracy", "bi_prp"],
    #         "distractionbased_ratio": ['0.9', '0.7', '0.5', '0.3', '0.1'],
    #         "prt":[prt],
    #     }
    #     title = "p-rate = {}".format(prt)
    #     xlabel = "s-rate"
    #     for x_axis in x_axes:
    #         plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="prt-{}_".format(prt), title=title, xlabel=xlabel)

    x_axes = ["prt"]
    for distractionbased_ratio in ["0.1", "0.3", "0.5", "0.7", "0.9"]:        
        filtermap = {
            "lo": ["prp", "nll", "lws", "rc", "democracy", "bi_prp"],
            "distractionbased_ratio":[distractionbased_ratio],
        }
        title = r'$r_{{Docc}} = {}$'.format(distractionbased_ratio)
        xlabel = r'$r_{{Dpool}}$'
        for x_axis in x_axes:
            plot(directory, series, x_axis, outdir, metrics, filtermap, prefix="dist-{}_".format(distractionbased_ratio), title=title, xlabel=xlabel, show_max=False)

def exp31():
    directory = "out/zs31"
    series = "lo"
    outdir="plots/zs31_cifar10"
    metrics=["Test Accuracy"]
    x_axes = ["case"]
    xlabel = "Noise Model"
    title = "Cifar10 with various noise models"
    for x_axis in x_axes:
        plot(directory, series, x_axis, outdir, metrics, xlabel=xlabel, title=title, show_max=False)


def exp32():
    directory = "out/zs32"
    series = "lo"
    outdir="plots/zs32"
    metrics=["Test Accuracy", "Test Prob PLL"]
    x_axes = ["mo"]
    xlabel = "Classification Model"
    result = {}
    for ds in ['birdac', 'lost', 'MSRCv2']:        
        filtermap = {
            "lo": ['bi_prp', "prp", 'nll', 'lws', 'rc', 'democracy'],
            "ds":[ds],
        }
        # title = "{} - with various models".format(ds)
        result[ds] = plot(directory, series, x_axes[0], outdir, metrics, filtermap, prefix="dataset-{}_".format(ds), title="", xlabel=xlabel, return_metrics=True)

    for ds, mo in [('birdac', 'mlp'), ('lost', 'linear'), ('MSRCv2', 'mlp')]:    
        res = result[ds]
        print("-- Dataset {}".format(ds))
        for metric in ["Test Accuracy"]:
            loss_funs = res[metric].keys()
            for lo in loss_funs:
                # models = res[metric][lo].keys()
                # for mo in ["mlp"]:
                print("{} - loss {} - model {}: {} +/- {} %".format(metric, lo, mo, np.around(100*np.mean(res[metric][lo][mo]),2), np.around(100*np.std(res[metric][lo][mo]),2)))


def exp33():
    directory = "out/zs33"
    series = "lo"
    outdir="plots/zs33"
    metrics=["Test Accuracy", "Test Prob PLL"]
    x_axes = ["mo"]
    xlabel = "Classification Model"
    result = {}
    dataset_list = ['birdac', 'lost', 'MSRCv2', 'LYN', 'spd']
    model_list = ['mlp', 'linear']
    for ds in dataset_list:        
        filtermap = {
            "lo": ['nll', 'democracy', 'merit'],
            "ds":[ds],
            "beta":['0.5']
        }
        # title = "{} - with various models".format(ds)
        result[ds] = plot(directory, series, x_axes[0], outdir, metrics, filtermap, prefix="dataset-{}_".format(ds), title="", xlabel=xlabel, return_metrics=True)

    for ds, mo in itertools.product(dataset_list, model_list):    
        res = result[ds]
        print("-- Dataset {}".format(ds))
        for metric in ["Test Accuracy"]:
            loss_funs = res[metric].keys()
            for lo in loss_funs:
                print("{} - loss {} - model {}: {} +/- {} %".format(metric, lo, mo, np.around(100*np.mean(res[metric][lo][mo]),2), np.around(100*np.std(res[metric][lo][mo]),2)))

exp33()

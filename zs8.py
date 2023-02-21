# effect of changing distance between centroids
from run import run_all

experiment=8
gpus=[0,2,3,4,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar100'],
    "-mo": ['cnn'],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["0", "1"],
    "-dseed": ["0"],
    "-cluster":["3"],
    "-prt": ["0.01", "0.02", "0.03", "0.04"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

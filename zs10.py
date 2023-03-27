# effect of changing distance between centroids
from run import run_all

experiment=10
gpus=[0,2,3,4,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar10'],
    "-mo": ['cnn'],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["0"],
    "-cluster":["4"],
    "-prt": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

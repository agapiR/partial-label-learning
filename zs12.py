# effect of changing distance between centroids
from run import run_all

experiment=12
gpus=[0,2,3,4,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar100'],
    "-mo": ['cnn'],
    "-lo": ['prp', 'prp_basic', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["0"],
    "-cluster":["4"],
    "-prt": ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3", "0.35", "0.4"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

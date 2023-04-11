# effect of changing distance between centroids
from run import run_all

experiment=18
gpus=[0,3,4,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar100'],
    "-mo": ['cnn'],
    "-lo": ['prp', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["0"],
    "-cluster":["4"],
    "-prt": ["0.0", "0.1"],
    "-num_groups":["1", "2", "4", "5", "10", "20"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

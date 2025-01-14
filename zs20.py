# effect of changing distance between centroids
from run import run_all

experiment=20
gpus=[1,2,3,4,5,6]
jobs_per_gpu=1
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar100'],
    "-mo": ['resnet18'],
    "-lo": ['prp', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.005", "0.01", "0.1"],
    "-wd": ["0.0005"],
    "-bs": ["128"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["0"],
    "-cluster":["4"],
    "-prt": ["0.0"],
    "-num_groups":["1"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

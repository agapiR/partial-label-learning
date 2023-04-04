# effect of changing distance between centroids
from run import run_all

experiment=21
gpus=[0,3,4,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['shierarchy32'],
    "-mo": ['cnn'],
    "-lo": ['prp', 'prp_basic', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["1","2","3","4"],
    "-cluster":["4"],
    "-prt": ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

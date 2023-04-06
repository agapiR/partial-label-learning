# effect of changing distance between centroids
from run import run_all

experiment=22
gpus=[0,1,2,3]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['shierarchy32'],
    "-mo": ['mlp'],
    "-lo": ['prp', 'prp_basic', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.01"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["1"],
    "-dseed": ["0"],
    "-cluster":["4"],
    "-prt": ["0.2"],
    "-num_groups":["1", "2", "4", "8", "16", "32"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

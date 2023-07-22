# effect of changing distance between centroids
from run import run_all

experiment=21
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
    "-seed": ["1","2","3"],
    "-dseed": ["0"],
    "-noise_model":["instancebased"],
    "-prt": ["0.9", "0.7", "0.5", "0.3", "0.2", "0.1", "0.0"],
    "-num_groups":["1", "2", "4", "8", "16", "32"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

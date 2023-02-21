# effect of changing distance between centroids
from run import run_all

experiment=7
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
    "-seed": ["1", "2", "3"],
    "-dseed": ["1", "2", "3"],
    "-cluster":["1"],
    "-prt": [str(i/10) for i in range(11)],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

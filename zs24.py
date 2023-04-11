# effect of changing distance between centroids
from run import run_all

experiment=24
gpus=[0,3,4,5,6]
jobs_per_gpu=5
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['cifar100'],
    "-mo": ['resnet18'],
    "-lo": ['prp', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.01"],
    "-wd": ["0.0005"],
    "-bs": ["128"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-dseed": ["0"],
    "-cluster":["0"],
    "-pr": ["h1uniform_0.2", "h2uniform_0.2", "h4uniform_0.2", "h5uniform_0.2", "h10uniform_0.2", "h20uniform_0.2", "h50uniform_0.2", "h1uniform_0.1", "h2uniform_0.1", "h4uniform_0.1", "h5uniform_0.1", "h10uniform_0.1", "h20uniform_0.1", "h50uniform_0.1", "h1uniform_0.05", "h2uniform_0.05", "h4uniform_0.05", "h5uniform_0.05", "h10uniform_0.05", "h20uniform_0.05", "h50uniform_0.05"]
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

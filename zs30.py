from run import run_all

experiment=30
gpus=[0,3,4,5,6]
jobs_per_gpu=5
outdir = "out/zs{}".format(experiment)

args = {
    "-seed": ["0"],
    "-dseed": ["0"],
    "-ds": ['cifar100'],
    "-mo": ['resnet18'],
    "-lw": ['1'],
    "-lr": ["0.1"],
    "-wd": ["0.0005"],
    "-bs": ["128"],
    "-ep": ["200"],
    "-prt": ["0.1", "0.3", "0.5", "0.7", "0.9"],
    "-lo": ['bi_prp', "prp", 'nll', 'lws', 'rc', 'democracy'],
    "-distractionbased_ratio": ['0.9', '0.7', '0.5', '0.3', '0.1'],
    "-noise_model": ["distractionbased"],
    "-logit_decay": ["0.03"],
}

run_all(args, outdir, gpus, jobs_per_gpu)

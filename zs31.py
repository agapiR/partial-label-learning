from run import run_all

experiment=31
gpus=[1,2,3,5,6]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-seed": ["0"],
    "-dseed": ["0"],
    "-ds": ['cifar10'],
    "-mo": ['cnn'],
    "-lw": ['1'],
    "-lr": ["0.05"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-pr": ["01", "02", "04"],
    "-lo": ['bi_prp', "prp", 'nll', 'lws', 'rc', 'democracy'],
    "-noise_model": ["uniform"],
    "-logit_decay": ["0.03"],
}

run_all(args, outdir, gpus, jobs_per_gpu)

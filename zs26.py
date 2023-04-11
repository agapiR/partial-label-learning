from run import run_all

experiment=26
gpus=[0,3,4,5,6]
jobs_per_gpu=12
outdir = "out/zs{}".format(experiment)

args = {
    "-seed": ["0", "1", "2"],
    "-dseed": ["0", "1", "2"],
    "-ds": ['synthetic'],
    "-mo": ['mlp'],
    "-lw": ['1'],
    "-lr": ["0.01"],
    "-wd": ["0.001"],
    "-bs": ["100"],
    "-ep": ["100"],
    "-nc": ["100"],
    "-ns": ["100000"],
    "-nf": ["100"],
    "-csep": ["5"],
    "-prt": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp', "democracy", "prp_basic"],
    "-noise_model": ["distractionbased"],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

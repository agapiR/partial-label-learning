from run import run_all

experiment=25
gpus=[0,2,3,4,5,6]
jobs_per_gpu=12
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['synthetic'],
    "-mo": ['mlp'],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp', "democracy", "prp_basic"],
    "-lw": ['1'],
    "-lr": ["0.001"],
    "-wd": ["0.001"],
    "-bs": ["100"],
    "-ep": ["100"],
    "-seed": ["0"],
    "-prt": ["0.3", "0.6", "0.7", "0.8", "0.9"],
    "-nc": ["100"],
    "-ns": ["50000"],
    "-nf": ["100"],
    "-csep": ["5"],
    "-dseed": ["2", "3", "4"],
    "-noise_model": ["distractionbased"],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

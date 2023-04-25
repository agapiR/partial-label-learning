from run import run_all

# parent: 27
# tuning bi_prp
experiment=28
gpus=[0,3,4,5,6]
jobs_per_gpu=12
outdir = "out/zs{}".format(experiment)

args = {
    "-seed": ["0"],
    "-dseed": ["0"],
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
    "-prt": ["0.9"],
    "-lo": ['bi_prp'],
    "-distractionbased_ratio": ['0.5'],
    "-noise_model": ["distractionbased"],
    "-logit_decay": ["0.01", "0.02", "0.03", "0.04", "0.05", "0.06", "0.07", "0.08", "0.09", "0.1"],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

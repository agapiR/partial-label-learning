from run import run_all

experiment=25
gpus=[0,2,3,5,6]
jobs_per_gpu=12
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['synthetic'],
    "-mo": ['mlp'],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp', "democracy",],
    "-lw": ['1'],
    "-lr": ["0.01", "0.001"],
    "-wd": ["0.001"],
    "-bs": ["100"],
    "-ep": ["200"],
    "-seed": [str(i) for i in range(5)],
    # "-alpha":["1.0"],
    "-prt": ["0.3", "0.6", "0.7", "0.8", "0.9"],
    "-nc": ["100"],
    "-ns": ["100000"],
    "-nf": ["100"],
    "-csep": ["5"],
    "-dseed": [str(i) for i in range(5)],
    "-noisemodel": ["distractionbased"],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

# effect of changing the dataset size
from run import run_all

experiment=5
gpus=[0,2,3,4,5,6]
jobs_per_gpu=10
outdir = "out/zs{}".format(experiment)

args = {
    "-ds": ['synthetic'],
    # "-pr": ['01'],
    "-mo": ['mlp'],
    "-lo": ['prp', 'lws', 'nll', 'rc', 'bi_prp'],
    "-lw": ['1'],
    "-lr": ["0.01"],
    "-wd": ["0.001"],
    "-bs": ["100"],
    "-ep": ["2000"],
    "-seed": ["0", "1", "2"],
    # "-alpha":["1.0"],
    "-prt": [str(i/10) for i in range(11)],
    "-nc": ["10"],
    "-ns": ["200", "500", "800", "1000", "1500", "2000", "5000"],
    "-nf": ["100"],
    "-csep": ["5"],
    "-dseed": ["1", "2"],
    }

run_all(args, outdir, gpus, jobs_per_gpu)

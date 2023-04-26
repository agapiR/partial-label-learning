from run import run_all

experiment=22
gpus=[0,1,2,3]
jobs_per_gpu=4
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
    "-seed": ["1", "2", "3", "4", "5"],
    "-dseed": ["11", "12", "13", "14", "15"],
    "-csep": ["0.5", "1.0", "1.5", "2.0"],
    "-cluster":["4"],
    "-prt": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"],
    "-num_groups":["1", "2", "4", "8", "16"],
    }

print("Experiment", experiment)

run_all(args, outdir, gpus, jobs_per_gpu)

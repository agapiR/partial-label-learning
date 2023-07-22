from run import run_all

experiment=33
gpus=[1,2,3,4]
jobs_per_gpu=3
outdir = "out/zs{}".format(experiment)

args = {
    "-seed": ["0", "1", "2"],
    "-dseed": ["0", "1", "2", "3", "4", "10", "11", "12", "13", "14", "20", "21", "22", "23", "24", "210", "211", "212", "213", "214"],
    "-ds": ['birdac', 'lost', 'MSRCv2', 'LYN', 'spd'],
    "-mo": ['mlp','linear'],
    "-lw": ['1'],
    "-lr": ["0.1"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["300"],
    "-lo": ['merit', 'bi_prp', "prp", 'nll', 'lws', 'rc', 'democracy'],
    "-beta": ['1.0', '0.0', '0.5'],
    "-noise_model": ["real"],
    "-logit_decay": ["0.03"],
}

run_all(args, outdir, gpus, jobs_per_gpu)
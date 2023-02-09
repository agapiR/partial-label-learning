import os
import subprocess
import itertools

args = {
    "-ds": ['synthetic'],
    "-pr": ['01'],
    "-mo": ['mlp'],
    "-lo": ['prp', 'lws', 'nll'],
    "-lw": ['1'],
    "-lr": ["0.005"],
    "-wd": ["0.001"],
    "-bs": ["256"],
    "-ep": ["100"],
    "-seed": ["5"],
    "-alpha":["1.0"],
    "-prt": ["0.1", "0.2"],
    "-nc": ["10"],
    "-ns": ["1000"],
    "-nf": ["5"],
    "-csep": ["0.1"],
    "-dseed": ["42"],
    }

def run_all(args, outdir="out", parallel=10):
    os.makedirs(outdir, exist_ok=True)
    tasks = []
    for values in itertools.product(*args.values()):
        command = ["python", "main-benchmarks.py"]
        outfile = "{}/experiment".format(outdir)
        for k, v in zip(args.keys(), values):
            command += [k, v]
            outfile += "_{}-{}".format(k, v)
        outfile += ".out"
        tasks.append((command, outfile))

    for i in range(0, len(tasks), parallel):
        parallel_tasks = tasks[i:i+parallel]

        handles = []
        for command, outfile in parallel_tasks:      
            with open(outfile, "w") as outstream:
                handle = subprocess.Popen(command, stdout=outstream)
                handles.append(handle)
        for handle in handles:
            handle.wait

run_all(args, "out2", 10)

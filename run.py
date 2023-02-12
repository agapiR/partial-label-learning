import os
import subprocess
import itertools
import math

def run_all(args, outdir, gpus, jobs_per_gpu=10):
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

    num_gpus = len(gpus)
    for i in range(0, len(tasks), jobs_per_gpu * num_gpus):
        parallel_tasks = tasks[i:i+ jobs_per_gpu * num_gpus]
        handles = []
        curr_gpu_index = 0
        for command, outfile in parallel_tasks:
            my_env = os.environ.copy()
            my_env["CUDA_VISIBLE_DEVICES"] = str(gpus[curr_gpu_index])
            print(command)
            handle = subprocess.Popen(command, env=my_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            handles.append((handle,outfile))
            curr_gpu_index = (curr_gpu_index + 1) % num_gpus            
        for (handle, outfile) in handles:
            handle.wait
            stdout, stderr = handle.communicate()
            with open(outfile, "w") as f:
                print(stdout, file=f)
            

                


#!/bin/bash

DATA_PATH="./data/realworld/"
OUT_PATH="./out/"
LOSS_LIST=(lws prp nll)
PARTIAL_RATE_LIST=(0.0 0.1 0.3 0.5 0.7 0.9)
ITERS=10

mkdir -p $OUT_PATH

for i in $(seq 1 $ITERS);
do
	for prt in "${PARTIAL_RATE_LIST[@]}"
	do
		dset=synthetic
		model=mlp
		for loss in "${LOSS_LIST[@]}"
		do
			python main-benchmarks.py -ds ${dset} -prt ${prt} -mo mlp -lo ${loss} -lw 1 -lr 0.05 -wd 0.001 -ldr 0.5 -lds 50 -bs 256 -ep 200 -gpu 1 -res ${OUT_PATH} >> "${OUT_PATH}hypercube_${model}_synthetic_${loss}_${prt}_rep_${i}.out"
		done
	done
done



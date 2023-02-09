#!/bin/bash

OUT_PATH="./results_synthetic/"
mkdir -p $OUT_PATH

DSET=synthetic
SAMPLE_NUM=1000
CLASS_NUM=10
FEATURE_NUM=5
RAND_SEED=5

MODEL_L=(mlp linear)
LOSS_L=(lws prp nll)
PARTIAL_RATE_L=(0.0 0.1 0.3 0.5 0.7 0.9)
CLASS_SEP_L=(0.5 1.0 1.5 2.0)
RAND_DATA_SEED_L=(41 42 43 44 45 46 47 48 49 50)

for model in "${MODEL_L[@]}"
do
	for loss in "${LOSS_L[@]}"
	do
		for prt in "${PARTIAL_RATE_L[@]}"
		do
			for csep in "${CLASS_SEP_L[@]}"
			do
				for dseed in "${RAND_DATA_SEED_L[@]}"
				do
					dset_name="synthetic_${SAMPLE_NUM}_${CLASS_NUM}_${FEATURE_NUM}_${prt}_${csep}_${dseed}"
					python main-benchmarks.py -mo ${model} -lo ${loss} -lw 1 -ds ${DSET} -ns ${SAMPLE_NUM} -nc ${CLASS_NUM} -nf ${FEATURE_NUM} -prt ${prt} -csep ${csep} -dseed ${dseed} -seed ${RAND_SEED} -lr 0.05 -wd 0.001 -ep 200 -gpu 1 -res ${OUT_PATH} >> "${OUT_PATH}${dset_name}_${model}_${loss}_${RAND_SEED}.out"
				done
			done
		done
	done
done



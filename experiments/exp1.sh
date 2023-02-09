#!/bin/bash

OUT_PATH="./results_synthetic/"
mkdir -p $OUT_PATH

DSET=synthetic
SAMPLE_NUM=1000
CLASS_NUM=10
FEATURE_NUM=5
RAND_DATA_SEED=42

MODEL_L=(mlp linear)
LOSS_L=(lws prp nll)
PARTIAL_RATE_L=(0.0 0.1 0.3 0.5 0.7 0.9)
CLASS_SEP_L=(0.5 1.0 1.5 2.0)
RAND_SEED_L=(1 2 3 4 5 6 7 8 9 10)

for model in "${MODEL_L[@]}"
do
	for loss in "${LOSS_L[@]}"
	do
		for prt in "${PARTIAL_RATE_L[@]}"
		do
			for csep in "${CLASS_SEP_L[@]}"
			do
				for seed in "${RAND_SEED_L[@]}"
				do
					dset_name="synthetic_${SAMPLE_NUM}_${CLASS_NUM}_${FEATURE_NUM}_${prt}_${csep}_${RAND_DATA_SEED}"
					python main-benchmarks.py -mo ${model} -lo ${loss} -lw 1 -ds ${DSET} -ns ${SAMPLE_NUM} -nc ${CLASS_NUM} -nf ${FEATURE_NUM} -prt ${prt} -csep ${csep} -dseed ${RAND_DATA_SEED} -seed ${seed} -lr 0.05 -wd 0.001 -ep 200 -gpu 1 -res ${OUT_PATH} >> "${OUT_PATH}${dset_name}_${model}_${loss}_${seed}.out"
				done
			done
		done
	done
done



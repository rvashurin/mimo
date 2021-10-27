#!/bin/bash

DEPTH=(22 28)
WIDTH=(1)
NUM=(1 2 3 5 7 9 12 15)

conda activate mimo

for depth in "${DEPTH[@]}"
do
  for width in "${WIDTH[@]}"
  do
    for ens_size in "${NUM[@]}"
    do
      dir="./runs/$ens_size/$depth/$width"
      mkdir -p $dir
      python custom/fmnist.py --output_dir=$dir --dataset=fashion_mnist --width=$width --per_core_batch_size=128 --train_epochs=250 --l2=3e-4 --checkpoint_interval=250 --use_gpu=True --ensemble_size=$ens_size --base_learning_rate=0.01 --depth=$depth
    done
  done
done

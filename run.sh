#!/bin/bash

python3_cmd=python3

stage=1
use_gpu=cuda:1

model=xlnet  # (bert|xlnet)
model_path=/home/M10815022/Models/xlnet-base-chinese
dataset="DRCD Lee Kaggle ASR"


if [ $stage -le 0 ]; then
  echo "==================================================="
  echo "     Convert traditional Chinese to simplified     "
  echo "==================================================="
  for dataset in DRCD Lee Kaggle ASR; do
    for split in training dev test; do
      
      file=dataset/$dataset/${dataset}_${split}.json
      echo "Converting '$file'..."
      opencc -i $file -o $file -c t2s.json
    done
  done
  echo "Done."
fi

if [ $stage -le 1 ]; then
  echo "======================"
  echo "     Prepare data     "
  echo "======================"
  rm -rf data
  for split in train dev test; do
    for dir in context context_no_unk question question_no_unk answer span; do
      mkdir -p data/$split/$dir
    done
  done
  $python3_cmd scripts/prepare_${model}_data.py $model_path $dataset FGC || exit 1
fi
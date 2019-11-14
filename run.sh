#!/bin/bash

python3_cmd=python3

stage=2
use_gpu=cuda:1

model=xlnet  # (bert|xlnet)
model_path=/home/M10815022/Models/xlnet-base-chinese
save_path=./models/xlnet
datasets="DRCD Lee Kaggle ASR"


if [ $stage -le 0 ]; then
  echo "==================================================="
  echo "     Convert traditional Chinese to simplified     "
  echo "==================================================="
  for dataset in $datasets FGC; do
    for split in training dev test; do
      file=dataset/$dataset/${dataset}_${split}.json
      if [ -f $file ]; then
        echo "Converting '$file'..."
        opencc -i $file -o $file -c t2s.json
      fi
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
  $python3_cmd scripts/prepare_${model}_data.py $model_path $datasets FGC || exit 1
fi


if [ $stage -le 2 ]; then
  echo "================================="
  echo "     Train and test QA model     "
  echo "================================="
  if [ -d $save_path ]; then
    echo "'$save_path' already exists! Please remove it and try again."; exit 1
  fi
  mkdir -p $save_path
  $python3_cmd scripts/train_${model}.py $use_gpu $model_path $save_path $datasets
fi

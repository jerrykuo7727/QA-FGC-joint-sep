#!/bin/bash

python3_cmd=python3

stage=1
use_gpu=cuda:1

pretrained_model=/home/M10815022/Models/bert-wwm-ext
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
  $python3_cmd scripts/prepare_bert_data.py $pretrained_model $dataset FGC || exit 1
fi
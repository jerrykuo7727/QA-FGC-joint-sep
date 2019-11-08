#!/bin/bash

python3_cmd=python3

stage=0
use_gpu=cuda:1


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
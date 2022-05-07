#!/usr/bin/env fish
# Usage example: fish avgmodel.sh 05042215 19x
set -gx timestamp $argv[1]
set -gx glob $argv[2]
set -gx globpat (string replace -a 'x' '*' $glob)
rm -f $timestamp-ep$glob.pth
python3 ~/pytorch-image-models/avg_checkpoints.py --input ../rift/checkpoints/$timestamp/ --filter ep$globpat --no-sort --no-use-ema --output $timestamp-ep$glob.pth
python3 ../craft/evaluate.py --dataset chairs --sofi --model $timestamp-ep$glob.pth

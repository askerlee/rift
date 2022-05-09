#!/usr/bin/env fish
# Usage example: fish avgmodel.sh 05042215 19x
set -gx timestamp $argv[1]
set -gx glob $argv[2]
set -gx globpat (string replace -a 'x' '*' $glob)
rm -f $timestamp-ep$glob.pth
python3 avg_checkpoints.py --input checkpoints/$timestamp/ --filter ep$globpat --no-sort --no-use-ema --output $timestamp-ep$glob.pth
set ckpt_dir (pwd)
cd ../craft/
python3 evaluate.py --dataset chairs --sofi --model $ckpt_dir/$timestamp-ep$glob.pth

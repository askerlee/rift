#!/usr/bin/env fish
set -gx eva_ip 10.2.1.199
echo 10.2.18.254:04191559
ssh li_shaohua@10.2.18.254 rsync -a --info=progress2 rift/checkpoints/04191559/ shaohua@$eva_ip:rift/checkpoints/04191559/
echo 10.2.18.238:04191744
ssh shaohua@10.2.18.238 rsync -a --info=progress2 rift/checkpoints/04191744/ shaohua@$eva_ip:rift/checkpoints/04191744/
#ssh shaohua@172.20.117.215 rsync -a --info=progress2 rift/checkpoints/04171710/ shaohua@$eva_ip:rift/checkpoints/04171710/

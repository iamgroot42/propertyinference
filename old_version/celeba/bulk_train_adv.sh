#!/bin/bash
for i in {1..1000}
do
    python train_models.py --split $1 --filter Male --ratio $2 --name $i --adv_train --eps 0.0157 --adv_name adv_train_4 --num_workers 10
done
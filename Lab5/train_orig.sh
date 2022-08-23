#!/usr/bin/env sh
python ./train.py --generator_lr 0.0002 --discriminator_lr 0.0001 --batch_size 64 --momentum 0.5 --epoch_size 300 --z_dim 100 --save_model_for_every_epoch --num_workers 8 --seed 1234 --comments "ground truth labels, momentum to 0.5"

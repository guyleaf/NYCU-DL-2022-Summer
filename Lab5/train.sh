#!/usr/bin/env sh
python ./train.py --lr 0.0002 --batch_size 64 --momentum 0.5 --epoch_size 100 --z_dim 100 --save_model_for_every_epoch --num_workers 8 --seed 1234 --comments "ground truth labels, interchange the order of leakyrelu and batchnorm, add projection, replace last layer of discriminator with FCN, add dropout2d to discriminator, momentum to 0.5"

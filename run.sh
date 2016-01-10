python -m DeepLearning.Seq2Seq.AIBot \
       --num_layers 2 \
       --size 256 \
       --steps_per_checkpoint 50 \
       --learning_rate 0.005 \
       --learning_rate_decay_factor 0.9 \
       --train_dir ./DeepLearning/cache/
#       --decode 1 # turn on for prediction

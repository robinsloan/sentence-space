# example training script
# you can snag the scifi-plus-gutenberg corpus, all derived from public domain sources,
# at https://www.dropbox.com/s/t38im4b5i1tvn87/scifi-plus-gutenberg.txt
python textproject_vae_charlevel.py \
-session sp15_trial \
-dataset data/scifi-plus-gutenberg.txt \
-z 128 \
-alpha 0.3 \
-lr 0.001 \
-lstm_size 1000 \
-batch_size 32 \
-max_len 128 \
-num_epochs 100 \
-anneal_start 200000 \
-anneal_end 400000

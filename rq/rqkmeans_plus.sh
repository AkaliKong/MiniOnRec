python rqkmeans_plus.py \
  --data_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
  --pretrained_codebook_path ../data/Amazon18/Industrial_and_Scientific/Industrial_and_Scientific.codebooks_constrained.npz \
  --num_emb_list 256 256 256 \
  --e_dim 2560 \
  --lr 1e-4 \
  --epochs 10000 \
  --batch_size 2048

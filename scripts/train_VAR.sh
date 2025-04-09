torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 train.py \
  --depth=16 --bs=384 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
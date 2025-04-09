# torchrun --nnodes=1 --nproc_per_node=1 class_cond_sample_ddp.py

torchrun --nnodes=1 --nproc_per_node=2 class_cond_sample_ddp.py --model-depth 30 --per-proc-batch-size 32

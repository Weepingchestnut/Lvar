# torchrun --nnodes=1 --nproc_per_node=1 class_cond_sample_ddp.py

torchrun --nnodes=1 --nproc_per_node=2 evaluation/imagenet/class_cond_sample_ddp.py --model-depth 24 --per-proc-batch-size 32

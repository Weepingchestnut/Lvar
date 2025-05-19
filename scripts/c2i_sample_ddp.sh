# ------ Random class order ------
# torchrun --nnodes=1 --nproc_per_node=2 evaluation/imagenet/class_cond_sample_ddp.py --model-depth 30 --per-proc-batch-size 32

# ------ sequence class order ------
torchrun --nnodes=1 --nproc_per_node=2 evaluation/imagenet/class_cond_sample_ddp_seq.py --model-depth 30 --per-proc-batch-size 32

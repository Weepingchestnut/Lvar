# torchrun --nproc_per_node=2 evaluation/coco/sample.py

# torchrun --nproc_per_node=2 evaluation/coco/sample_kv.py

python evaluation/coco/compute_metrics.py --input_root0 work_dir/evaluation/coco/samples/gt_infinity_2b/ --input_root1 /home/zekun/workspace/ScaleKV/samples/gt_2b



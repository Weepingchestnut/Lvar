# IMAGE_ROOT_PATH=$1
# RESOLUTION=$2
IMAGE_ROOT_PATH=work_dir/evaluation/dpg-bench/infinity_2b/dpg-bench_cfg4_tau1_cfg_insertion_layer0_rewrite_prompt0_round2_real_rewrite/images
RESOLUTION=1024
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-2}   # default GPU number
PORT=${PORT:-29500}

export MODELSCOPE_CACHE="./pretrained_models/.modelscope_cache" 
# mkdir -p $MODELSCOPE_CACHE

accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  evaluation/dpg_bench/compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --vqa-model mplug

# python evaluation/dpg_bench/compute_dpg_bench.py \
#   --image-root-path $IMAGE_ROOT_PATH \
#   --resolution $RESOLUTION \
#   --pic-num $PIC_NUM \
#   --vqa-model mplug

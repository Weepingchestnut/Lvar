# set arguments for inference
pn=1M
model_type=infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=pretrained_models/infinity/Infinity/infinity_2b_reg.pth
out_dir_root=work_dir/evaluation/dpg-bench/infinity_2b
vae_type=32
vae_path=pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth
cfg=4
tau=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=pretrained_models/infinity/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

# DPG-Bench
out_dir=${out_dir_root}/dpg-bench_${sub_fix}

# --- run inference ---
python evaluation/dpg_bench/infer4dpg.py \
    --cfg ${cfg} \
    --tau ${tau} \
    --pn ${pn} \
    --model_path ${infinity_model_path} \
    --vae_type ${vae_type} \
    --vae_path ${vae_path} \
    --add_lvl_embeding_only_first_block ${add_lvl_embeding_only_first_block} \
    --use_bit_label ${use_bit_label} \
    --model_type ${model_type} \
    --rope2d_each_sa_layer ${rope2d_each_sa_layer} \
    --rope2d_normalized_by_hw ${rope2d_normalized_by_hw} \
    --use_scale_schedule_embedding ${use_scale_schedule_embedding} \
    --cfg ${cfg} \
    --tau ${tau} \
    --checkpoint_type ${checkpoint_type} \
    --text_encoder_ckpt ${text_encoder_ckpt} \
    --text_channels ${text_channels} \
    --apply_spatial_patchify ${apply_spatial_patchify} \
    --cfg_insertion_layer ${cfg_insertion_layer} \
    --outdir ${out_dir}/images \
    --rewrite_prompt ${rewrite_prompt}


# --- calculate metrics ---
source ~/anaconda3/etc/profile.d/conda.sh
conda activate modelscope

IMAGE_ROOT_PATH=${out_dir}/images
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

# conda deactivate

#!/bin/bash

infer_eval_image_reward() {
    # --- step 1, infer images ---
    # single GPU
    # python evaluation/image_reward/infer4eval.py \
    # mutil GPUs
    torchrun --nproc_per_node=${gpu_num} --master-port ${master_port} evaluation/image_reward/infer4reward_ddp.py \
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
        --outdir ${out_dir} \
        --skip_last_scales ${skip_last_scales} \
        2>&1 | tee ${out_dir}/eval_image-reward.log

    # --- step 2, compute image reward ---
    source ~/anaconda3/etc/profile.d/conda.sh       # Make sure your anaconda3 is in your home path
    conda activate torch260                         # Requires Flash-Attention version >=2.7.1,<=2.8.0

    python evaluation/image_reward/cal_imagereward.py \
        --meta_file ${out_dir}/metadata.jsonl 2>&1 | tee ${out_dir}/cal_image_reward.log
    
    conda deactivate
}

infer_eval_hpsv21() {
    torchrun --nproc_per_node=${gpu_num} --master-port ${master_port} evaluation/hpsv2/infer4hpsv2_ddp.py \
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
        --skip_last_scales ${skip_last_scales} \
        2>&1 | tee ${out_dir}/eval_hpsv2.log
}

test_gen_eval() {
    # --- run inference ---
    torchrun --nproc_per_node=${gpu_num} --master-port ${master_port} evaluation/gen_eval/infer4eval_ddp.py \
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
        --rewrite_prompt ${rewrite_prompt} \
        --skip_last_scales ${skip_last_scales} \
        2>&1 | tee ${out_dir}/eval_gen-eval.log

    # --- detect objects ---
    source ~/anaconda3/etc/profile.d/conda.sh       # Make sure your anaconda3 is in your home path
    conda activate modelscope

    # export CUDA_LAUNCH_BLOCKING=1
    python evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
        --outfile ${out_dir}/results/det.jsonl \
        --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
        --model-path pretrained_models/mask2former

    # --- accumulate results ---
    python evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt

    unset CUDA_LAUNCH_BLOCKING
    conda deactivate
}

test_dpg_bench() {
    # --- run inference ---
    # single GPU
    # python evaluation/dpg_bench/infer4dpg.py \
    # mutil GPUs
    torchrun --nproc_per_node=${gpu_num} --master-port ${master_port} evaluation/dpg_bench/infer4dpg_ddp.py \
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
        --skip_last_scales ${skip_last_scales} \
        2>&1 | tee ${out_dir}/eval_dpg-bench.log
    
    # --- calculate metrics ---
    source ~/anaconda3/etc/profile.d/conda.sh       # Make sure your anaconda3 is in your home path
    conda activate modelscope

    IMAGE_ROOT_PATH=${out_dir}/images
    RESOLUTION=1024
    PIC_NUM=${PIC_NUM:-4}
    PROCESSES=${PROCESSES:-${gpu_num}}   # default GPU number
    PORT=${PORT:-${master_port}}

    export MODELSCOPE_CACHE="./pretrained_models/.modelscope_cache"
    # mkdir -p $MODELSCOPE_CACHE

    accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
        evaluation/dpg_bench/compute_dpg_bench.py \
        --image-root-path $IMAGE_ROOT_PATH \
        --resolution $RESOLUTION \
        --pic-num $PIC_NUM \
        --vqa-model mplug 2>&1 | tee ${out_dir}/matrics_dpg-bench.log

    conda deactivate
}

latency_profile() {
    python tools/latency_profile_infinity.py \
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
        --skip_last_scales ${skip_last_scales} \
        --batch_size ${batch_size} \
        2>&1 | tee ${out_dir}/infer_profile_batch-${batch_size}.log
}

# set arguments for inference
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2
master_port=29502
skip_last_scales=0

model_type=infinity_2b
# model_type=infinity_8b

model_exp=${model_type}

if [ "$model_type" == "infinity_2b" ]; then
    checkpoint_type='torch'
    infinity_model_path=pretrained_models/infinity/Infinity/infinity_2b_reg.pth
    vae_type=32
    vae_path=pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth
    apply_spatial_patchify=0
    cfg=4
    tau=1
elif [ "$model_type" == "infinity_8b" ]; then
    checkpoint_type='torch_shard'
    infinity_model_path=pretrained_models/infinity/Infinity/infinity_8b_weights
    vae_type=14
    vae_path=pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth
    apply_spatial_patchify=1
    cfg=4
    tau=1
else
    echo "Unknown model_type '$model_type'"
    echo "Support model_type: 'infinity_2b', 'infinity_8b'"
    exit 1
fi

pn=1M
use_scale_schedule_embedding=0
use_bit_label=1
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=pretrained_models/infinity/flan-t5-xl
text_channels=2048
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

# ------ Performance ------
out_dir_root=work_dir/infer_profile/${model_exp}
out_dir=${out_dir_root}/latency-profile_${sub_fix}
mkdir -p ${out_dir}

batch_size=1
latency_profile

batch_size=2
latency_profile

batch_size=4
latency_profile

batch_size=8
latency_profile
sleep 10


# ------ GenEval ------
out_dir_root=work_dir/evaluation/gen_eval/${model_exp}
rewrite_prompt=1    # default: 1, load prompt_rewrite_cache.json, from https://github.com/user-attachments/files/18260941/prompt_rewrite_cache.json
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
mkdir -p ${out_dir}

test_gen_eval
sleep 10


# ------ DPG-Bench ------
out_dir_root=work_dir/evaluation/dpg-bench/${model_exp}
out_dir=${out_dir_root}/dpg-bench_${sub_fix}
mkdir -p ${out_dir}

test_dpg_bench
sleep 10


# ------ HPS v2.1 ------
out_dir_root=work_dir/evaluation/hpsv2/${model_exp}
out_dir=${out_dir_root}/hpsv21_${sub_fix}
mkdir -p ${out_dir}

infer_eval_hpsv21
sleep 10


# ------ ImageReward ------
out_dir_root=work_dir/evaluation/image_reward/${model_exp}
out_dir=${out_dir_root}/image_reward_${sub_fix}
mkdir -p ${out_dir}

infer_eval_image_reward
sleep 10

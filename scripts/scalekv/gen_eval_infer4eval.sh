# set arguments for inference
model_type=scalekv_infinity_8b
# model_type=scalekv_infinity_2b

out_dir_root=work_dir/evaluation/gen_eval/${model_type}

if [ "$model_type" == "scalekv_infinity_2b" ]; then
    checkpoint_type='torch'
    infinity_model_path=pretrained_models/infinity/Infinity/infinity_2b_reg.pth
    vae_type=32
    vae_path=pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth
    apply_spatial_patchify=0
    cfg=4
    tau=1
elif [ "$model_type" == "scalekv_infinity_8b" ]; then
    checkpoint_type='torch_shard'
    infinity_model_path=pretrained_models/infinity/Infinity/infinity_8b_weights
    vae_type=14
    vae_path=pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth
    apply_spatial_patchify=1
    cfg=4
    tau=1
else
    echo "Unknown model_type '$model_type'"
    echo "Support model_type: 'scalekv_infinity_2b', 'scalekv_infinity_8b'"
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

# GenEval
rewrite_prompt=1    # default: 1, load prompt_rewrite_cache.json, from https://github.com/user-attachments/files/18260941/prompt_rewrite_cache.json
out_dir=${out_dir_root}/gen_eval_${sub_fix}_rewrite_prompt${rewrite_prompt}_round2_real_rewrite
mkdir -p ${out_dir}

# --- run inference ---
# single GPU
# python evaluation/gen_eval/infer4eval.py \
# mutil GPUs
unset CUDA_VISIBLE_DEVICES
torchrun --nproc_per_node=4 evaluation/gen_eval/infer4eval_ddp.py \
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
    --rewrite_prompt ${rewrite_prompt} 2>&1 | tee ${out_dir}/eval_geneval.log

# --- detect objects ---
source ~/anaconda3/etc/profile.d/conda.sh       # Make sure your anaconda3 is in your home path
conda activate modelscope

export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
python evaluation/gen_eval/evaluate_images.py ${out_dir}/images \
    --outfile ${out_dir}/results/det.jsonl \
    --model-config evaluation/gen_eval/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py \
    --model-path pretrained_models/mask2former

# --- accumulate results ---
python evaluation/gen_eval/summary_scores.py ${out_dir}/results/det.jsonl > ${out_dir}/results/res.txt
    cat ${out_dir}/results/res.txt

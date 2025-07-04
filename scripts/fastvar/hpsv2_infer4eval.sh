# set arguments for inference
pn=1M
model_type=fastvar_infinity_2b
use_scale_schedule_embedding=0
use_bit_label=1
checkpoint_type='torch'
infinity_model_path=pretrained_models/infinity/Infinity/infinity_2b_reg.pth
out_dir_root=work_dir/evaluation/hpsv2/fastvar_infinity_2b
vae_type=32
vae_path=pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth
cfg=3
tau=0.5
rope2d_normalized_by_hw=2
add_lvl_embeding_only_first_block=1
rope2d_each_sa_layer=1
text_encoder_ckpt=pretrained_models/infinity/flan-t5-xl
text_channels=2048
apply_spatial_patchify=0
cfg_insertion_layer=0
sub_fix=cfg${cfg}_tau${tau}_cfg_insertion_layer${cfg_insertion_layer}

# HPS v2.1
out_dir=${out_dir_root}/hpsv21_${sub_fix}
mkdir -p ${out_dir}

# --- single GPU ---
# python evaluation/hpsv2/eval_hpsv2.py \
# --- mutil GPUs ---
unset CUDA_VISIBLE_DEVICES
torchrun --nproc_per_node=2 evaluation/hpsv2/eval_hpsv2_ddp.py \
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
    --outdir ${out_dir}/images 2>&1 | tee ${out_dir}/eval_hpsv2.log

#!/bin/bash

test_vbench() {
    torchrun --nproc_per_node=${gpu_num} --master-port ${master_port} evaluation/vbench/infer4vbench_ddp.py \
        --pn ${pn} \
        --fps ${fps} \
        --generation_duration ${generation_duration} \
        --model_path ${model_path} \
        --image_scale_repetition "${image_scale_repetition}" \
        --video_scale_repetition "${video_scale_repetition}" \
        --append_enlarge2captain ${append_enlarge2captain} \
        --detail_scale_min_tokens ${detail_scale_min_tokens} \
        --semantic_scales ${semantic_scales} \
        --output_root ${out_dir} \
        --target_dimensions "${dimensions[@]}" \
        2>&1 | tee ${out_dir}/eval_vbench_${timestamp}.log
}

vbench_score() {
    source ~/anaconda3/etc/profile.d/conda.sh       # Make sure your anaconda3 is in your home path
    conda activate vbench

    # Loop over each dimension
    for i in "${!dimensions[@]}"; do
        # Get the dimension and corresponding folder
        dimension=${dimensions[i]}

        # Construct the video path
        videos_path=${dim_path}/${dimension}
        if [ ! -d "$videos_path" ]; then
            echo "Warning: Directory $videos_path does not exist. Skipping evaluation for $dimension."
            continue
        fi

        echo "Evaluating $dimension at $videos_path"

        vbench evaluate --ngpus=${gpu_num} \
            --output_path ${out_dir}/evaluation_results/ \
            --videos_path $videos_path \
            --dimension $dimension \
            2>&1 | tee ${dim_path}/vbench_${dimension}_${timestamp}.log
    done

    # calculate VBench final scores
    python evaluation/vbench/cal_vbench_final_score.py \
        --json_files_path ${out_dir}/evaluation_results/ \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_final_score_${timestamp}.log

    conda deactivate

    # ------ low-level score ------
    # # if you install `ffmpeg` in conda env or system `ffmpeg` version old, use 
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

    torchrun --nproc_per_node=${gpu_num} evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${vbench_gts} \
        --candidate-root ${out_dir} \
        --include-first-frame 1 \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_low-level_metrics_first-frame1_${timestamp}.log

    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    torchrun --nproc_per_node=${gpu_num} evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${vbench_gts} \
        --candidate-root ${out_dir} \
        --preferred-source video \
        --include-first-frame 1 \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_low-level_metrics_video_first-frame1_${timestamp}.log

    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    torchrun --nproc_per_node=${gpu_num} evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${vbench_gts} \
        --candidate-root ${out_dir} \
        --include-first-frame 0 \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_low-level_metrics_first-frame0_${timestamp}.log

    timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
    torchrun --nproc_per_node=${gpu_num} evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${vbench_gts} \
        --candidate-root ${out_dir} \
        --preferred-source video \
        --include-first-frame 0 \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_low-level_metrics_video_first-frame0_${timestamp}.log
    
    single GPU
    python3 evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${vbench_gts} \
        --candidate-root ${out_dir} \
        2>&1 | tee ${out_dir}/evaluation_results/vbench_low-level_metrics_${timestamp}.log
    
}

latency_profile() {
    # --- run inference ---
    # single GPU
    python -u tools/latency_profile_infistar.py \
        --pn ${pn} \
        --fps ${fps} \
        --generation_duration ${generation_duration} \
        --model_path ${model_path} \
        --image_scale_repetition "${image_scale_repetition}" \
        --video_scale_repetition "${video_scale_repetition}" \
        --append_enlarge2captain ${append_enlarge2captain} \
        --detail_scale_min_tokens ${detail_scale_min_tokens} \
        --semantic_scales ${semantic_scales} \
        --profile_output_root ${out_dir} \
        --infer_batch_size ${batch_size} \
        2>&1 | tee ${out_dir}/infer_profile_batch-${batch_size}_${timestamp}.log
    
    # ------ low-level metrics ------
    # if you install `ffmpeg` in conda env, use 
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    mkdir -p ${out_dir}/evaluation_results

    python -u evaluation/vbench/eval_low_level_metrics.py \
        --baseline-root ${profile_gts} \
        --candidate-root ${out_dir} \
        --collect-per-video \
        2>&1 | tee ${out_dir}/evaluation_results/profile_low-level_metrics_${timestamp}.log
}


# set arguments for inference
export CUDA_VISIBLE_DEVICES=0,1
gpu_num=2
master_port=29501

model_type=infinitystar
resolution=720p
fps=16
generation_duration=5
append_enlarge2captain=1    # use f'{prompt}, Close-up on big objects, emphasize scale and detail'
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")

# TODO: change to different exps
model_exp=${model_type}_${resolution}
sub_fix=fps${fps}_${generation_duration}s_enlarge2captain${append_enlarge2captain}

# Define the dimension list
# dimensions=(    # Quick eval
#     "multiple_objects" 
#     "scene"  
#     "human_action" 
#     "appearance_style"
# )
dimensions=(
    "subject_consistency" 
    "background_consistency" 
    "aesthetic_quality" 
    "imaging_quality" 
    "object_class" 
    "multiple_objects" 
    "color" 
    "spatial_relationship" 
    "scene" 
    "temporal_style" 
    "overall_consistency" 
    "human_action" 
    "temporal_flickering" 
    "motion_smoothness" 
    "dynamic_degree" 
    "appearance_style"
)

if [ "$resolution" == "720p" ]; then
    pn='0.90M'
    model_path=pretrained_models/infinitystar/infinitystar_8b_720p_weights
    image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    detail_scale_min_tokens=750
    semantic_scales=12
elif [ "$resolution" == "480p" ]; then
    pn='0.40M'
    model_path=pretrained_models/infinitystar/infinitystar_8b_480p_weights
    image_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    video_scale_repetition='[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1]'
    detail_scale_min_tokens=350
    semantic_scales=11
else
    echo "Unknown resolution '$resolution'"
    echo "Support resolution: '720p', '480p'"
    exit 1
fi

# ------ Performance ------
out_dir_root=work_dir/infer_profile/${model_exp}
out_dir=${out_dir_root}/latency-profile_${sub_fix}
mkdir -p ${out_dir}

batch_size=1
latency_profile

# ------ vbench ------
out_dir_root=work_dir/evaluation/vbench
out_dir=${out_dir_root}/${model_exp}/${sub_fix}
mkdir -p ${out_dir}
dim_path=${out_dir}/videos_by_dimension

start_sec=$(date +%s)

test_vbench
vbench_score

end_sec=$(date +%s)
total_sec=$((end_sec - start_sec))
printf "Time: %d day %d hour %d min %d s\n" \
    $((total_sec / 86400)) \
    $(( (total_sec % 86400) / 3600 )) \
    $(( (total_sec % 3600) / 60 )) \
    $((total_sec % 60))

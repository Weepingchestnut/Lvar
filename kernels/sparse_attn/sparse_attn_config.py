import copy
import yaml

BASE_CONFIG = {
    'num_model_invocations_per_inference_step': 1,
    'should_profile': False,
    'generation_index': 0,
    'steps': 13,
    # Multi-GPU currently only supported for Hunyuan
    'world_size': 1,

    'mlp': {
        'is_enabled': True,
        'is_fp8': False,

        'top_keys': 0.3,
        'random_keys': 0.05,
        'full_step_every': 10,
        'block_mask_cache': 2,
        'first_n_dense_layers': 2,

        'provider': 'triton', # either 'cuda' or 'triton'
    },
    "patchify": {
        'is_enabled': True,

        # To disable patching at any level, set that level's patch size to 1. To disable patching entirely, set all patch sizes to 1.
        "chunk_size_1": 8,
        "chunk_size_2": 4,
    },
    'attn': {
        'is_enabled': True,
        # which SparseDiffAttn path: 'image' (Infinity) or 'video' (InfinityStar)
        'model_family': 'video',
        # ------ CS4A ------
        'decision_scale': 10,                   # sparse decision scale `S`
        'decision_on_first_repeat': False,      # True: full decision on decision scale's repeat 0, later repeats sparse; False: legacy decision on last repeat
        'speedup': 0,                           # 1: use TK-CUDA dense_colsum_attn; 0: use Triton dense_colsum_attn
        'colsum_fa2': False,                    # true: decision colsum = FA2 dense (o + row-lse byproduct) + QK^T-only Triton colsum kernel (exact, drops the redundant o recompute); false: legacy double full-attention Triton path
        'top_keys': 0.05,
        'use_current_scale_topklen': False,
        'random_keys': 0.01,
        'local_voxels': 0,                      # Number of local voxels to use for static local attention
        'local_1d_window': 0,
        'cs4a_qgroup_map_mode': 'nearest',
        'cs4a_qgroup_interp_align': 'center',
        'cs4a_index_expand_mode': 'footprint',
        'cs4a_band_neighbor_frames': 1,         # frame_band: own frame +/- N neighbour frames
        # ------ CS4A (image / Infinity only) ------
        'cs4a_image_kv_map': 'relative',        # 'relative': Decompose-Align-Project LUT; 'identity': keep decision coords (conference behavior)
        'cs4a_sink_scales': 5,                  # union tokens of the first N scales as attention sink (>= k - decision_scale when 'relative')
        'cs4a_image_expand_mode': 'center_union',  # 'center_union': mapped centers ∪ local ∪ sink; 'footprint': inverse parent-LUT mask expansion
        'cs4a_footprint_union_local': False,    # footprint: also union the local window (cs4a_ws)
        'use_o_cache': False,
        'update_o_cache_on_sparse_scale': 'none',
        'first_n_dense_layers': 2,
        # ------ CSLA ------
        'attn_sink': 5,
        'win_size': [-1,-1,-1,-1,-1,-1,-1,-1,1,1,3,5,7],
        # ------ D-CSLA (block-sparse FlexAttention path; alternative to csp_attn) ------
        'dcsla_enabled': False,
        'dcsla_block_size': 128,
        'dcsla_select_mode': 'coverage',        # ['coverage', 'topk_ratio']
        'dcsla_tau': 0.9,                       # coverage: cover tau of the row's TOTAL mass (sink-covered mass credited)
        'dcsla_topk_ratio': 0.2,                # topk_ratio: fixed fraction of V (iso-budget vs cs4a top_keys)
        'dcsla_min_blocks': 16,                 # min selected video blocks per q-block row
        'dcsla_max_blocks_ratio': 0.5,          # cap on selected video blocks, fraction of V
        'dcsla_local_diag_blocks': 1,           # force +/-N blocks around the diagonal-aligned video position; 0=off
        'dcsla_keep_sink': True,                # True: force-keep all ctx(sink) blocks; False: sink selected per dcsla_sink_select_tau
        'dcsla_sink_select_tau': 0.0,           # keep_sink=False: >0 = sink region runs its OWN coverage selection (sink mass is exact, identity transfer); 0 = sink competes jointly with video
        'dcsla_mass_normalize': True,           # mass-conserving transfer (divide by footprint multiplicity); False = raw counts (v1 behavior)
        'dcsla_colsum_group_rows': 192,         # decision colsum query-group rows, multiple of 64 ({192,128,64}); Triton colsum only (CUDA speedup colsum is fixed at 192); ignored unless dcsla_enabled

        'dcsla_flex_mode': 'max-autotune-no-cudagraphs',   # torch.compile mode; ~ / 'none' for default
        'dcsla_flex_dynamic': True,             # torch.compile(dynamic=...) for varying text/kv lengths
        # ------ CS4A + CSLA ------
        'bound_layer': 10,
        # 
        'full_step_every': 10,
        'full_step_schedule': None,

        'recompute_mask': True,
        'should_compress_indices': True,
        'should_keep_tail_dense': False,
        
        'provider': 'triton', # either 'cuda' or 'triton'
    },
    "offloading": {
        'global_disable_offloading': False,

        'mlp.out_cache': False,
        'mlp.indices': False,
        'mlp.counts': False,
        'mlp.sparse_act_T': False,
        'mlp.blockmean_mid_cache': False,

        'attn.out_cache': True,
        'attn.indices': True,
        'attn.counts': False,
        'attn.lse_constants': False,

        'text_encoders': True,
    },
    "step_caching": {
        'is_enabled': False,
        'skip_step_schedule': set([7, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43])
    }
}

def get_kernel_config_mlp():
    # use the same block sizes for MLP across triton and CUDA implementations
    return {
        'bm': 128,
        'mbm': 128,
        'counts_multiple_of': 256,
    }

def get_kernel_config_attn():
    if GLOBAL_CONFIG['attn']['provider'] == 'triton': 
        # Triton-based FA2 uses a blocksize of 64x64 but we still use 192 for the BM for memory efficiency
        return {
            'bm': 192,
            'counts_multiple_of': 64,
            'indices_pad_to': 1,
        }
    elif GLOBAL_CONFIG['attn']['provider'] == 'cuda':
        # CUDA-based FA3 uses a blocksize of 192x~128
        return {
            'bm': 192,
            'counts_multiple_of': 112,
            'indices_pad_to': 4,
        }
    else:
        raise ValueError(f"Invalid provider: {GLOBAL_CONFIG['attn']['provider']}")

GLOBAL_CONFIG = copy.deepcopy(BASE_CONFIG)

def update_global_config(config):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG.update({
        **GLOBAL_CONFIG,
        **config,
    })

import sys
import yaml
from typing import Dict, Any

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> None:
    """Recursively update dictionary d with values from u"""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v

def load_from_file(config_file: str) -> None:
    with open(config_file, 'r') as f:
        yaml_config = yaml.safe_load(f)
        
    # Update global config
    if yaml_config:
        _deep_update(GLOBAL_CONFIG, yaml_config)
        # update_global_config(yaml_config)
        print(f"SparVAR: using config file {config_file}")
        # print(yaml_config)
        print(yaml.dump(yaml_config, sort_keys=False, allow_unicode=True, default_flow_style=False))

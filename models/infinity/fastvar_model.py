import numpy as np
import torch
import torch.nn.functional as F

from models.infinity.basic_infinity import CrossAttnBlock
from models.infinity.infinity_model import Infinity, sample_with_top_k_top_p_also_inplace_modifying_logits_

from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


class FastVAR_Infinity(Infinity):
    def __init__(
        self,
        # *====== FastVAR specific parameter ======*
        fastvar_enabled=True,
        fastvar_cache_step_N=4,
        fastvar_pruning_ratios=None,
        **kwargs_for_infinity
    ):
        """FastVAR add new args

        Args:
            fastvar_enabled (bool): Defaults to True.
            fastvar_cache_step_N (int): Number of last N steps to apply FastVAR (Texture Filling Stage), K-N is the caching step. 
                                        So if K=13, N=4, caching at step 9. Defaults to 4.
            N (int): List of pruning ratios (1.0 - keep_ratio) for the N steps, 
                        e.g., [0.6, 0.5, 0.0, 0.0] for keep ratios 40%, 50%, 100%, 100%.
                        A pruning ratio of 1.0 means all tokens are pruned (step is skipped). Defaults to 4.
        """
        super().__init__(**kwargs_for_infinity)

        # *====== FastVAR ======*
        self.fastvar_enabled = fastvar_enabled
        self.fastvar_cache_step_N = fastvar_cache_step_N
        if fastvar_enabled:
            if fastvar_pruning_ratios is None:
                self.fastvar_pruning_ratios = [0.4, 0.5, 1.0, 1.0][:fastvar_cache_step_N]
                print(f"\n[FastVAR_per-layer enabled with default pruning ratios: {self.fastvar_pruning_ratios} for last {fastvar_cache_step_N} steps.]")
            else:
                self.fastvar_pruning_ratios = fastvar_pruning_ratios
            assert len(self.fastvar_pruning_ratios) == self.fastvar_cache_step_N, \
                "Length of fastvar_pruning_ratios must match fastvar_cache_step_N."
        self.fastvar_cached_output = None           # To store the output of the K-N step
        self.fastvar_cache_resolution = None        # To store (H,W) of cached step output for interpolation check
        # ======================
    
    def add_lvl_embeding_pivotal(self, feature, scale_ind, scale_schedule, pivotal_indices_for_current_scale,
                                 need_to_pad=0):
        bs, p_seq_len, _ = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w

        # lvl_embed of original full tokens
        full_lvl_embed = self.lvl_embed(scale_ind*torch.ones((bs, t_mul_h_mul_w),dtype=torch.int).to(feature.device))   # [bs, full_seq_len, 2048]
        # get pivotal tokens lvl_embed
        assert p_seq_len == pivotal_indices_for_current_scale.shape[1]
        p_lvl_embed = full_lvl_embed[:, pivotal_indices_for_current_scale.squeeze(0), :]    # [bs, p_seq_len, 2048]
        
        feature += p_lvl_embed
        return feature
    
    def _pivotal_token_selection(self, x_k_input, current_scale_spatial_shape, keep_k_tokens):
        """Performs Pivotal Token Selection (PTS).

        Args:
            x_k_input (_type_): Input token map for the current scale (B, NumTokens, C)
            current_scale_spatial_shape (_type_): (H, W) of the token map for x_k_input
            keep_k_tokens (_type_): Number of pivotal tokens to keep.

        Returns:
            pivotal_indices: (B, keep_k_tokens) or a flat tensor of indices for gather.
            original_indices_for_scattering: Indices to scatter pivotal outputs back.
        """
        B, NumTokens, C = x_k_input.shape
        H, W = current_scale_spatial_shape
        assert NumTokens == H * W, "NumTokens does not match H*W"

        x_k_spatial = x_k_input.reshape(B, H, W, C)

        # Estimate low-frequency component (global average pooling)
        # Pool over H, W dimensions, keep C and B.
        x_k_low_freq = torch.mean(x_k_spatial, dim=(1, 2), keepdim=True) # (B, 1, 1, C)

        # High-frequency component
        x_k_high_freq = x_k_spatial - x_k_low_freq # (B, H, W, C)

        # Pivotal score (L2 norm of high-frequency maps per token)
        # Norm over the channel dimension C
        s_k_scores = torch.linalg.norm(x_k_high_freq, ord=2, dim=3) # (B, H, W)
        s_k_scores_flat = s_k_scores.reshape(B, -1) # (B, NumTokens)

        if keep_k_tokens >= NumTokens: # Keep all tokens if keep_k is too large
            pivotal_indices = torch.arange(NumTokens, device=x_k_input.device).unsqueeze(0).expand(B, -1)
            return pivotal_indices, pivotal_indices # original_indices = pivotal_indices

        # TopK selection
        _, pivotal_indices = torch.topk(s_k_scores_flat, k=keep_k_tokens, dim=1, sorted=False) # (B, keep_k_tokens)

        return pivotal_indices # Shape (B, keep_k_tokens), to be used with gather for each batch item
    
    def _cached_token_restoration(self, pivotal_output, pivotal_indices, cached_output_at_KN_step, current_scale_spatial_shape, target_B_NumTokens_C_shape):
        """Performs Cached Token Restoration (CTR).

        Args:
            pivotal_output (tensor): Output from VAR module for pivotal tokens (bs, NumPivotalTokens, C) where bs could be 2*B for CFG
            pivotal_indices (tensor): Indices of pivotal tokens (B, NumPivotalTokens) - batch-local
            cached_output_at_KN_step (tensor): Cached output from K-N step (bs, H_cache, W_cache, C)
            current_scale_spatial_shape (tuple): (H, W) of the current full token map
            target_B_NumTokens_C_shape (tuple): (bs, H*W, C) - desired shape of the full output

        Returns:
            restored_output: Full token map for the current scale (bs, H*W, C)
        """
        bs, NumPivotalTokens, C_out = pivotal_output.shape
        B_orig, _ = pivotal_indices.shape # Original batch size before CFG doubling
        H_curr, W_curr = current_scale_spatial_shape
        NumTokens_curr = H_curr * W_curr
        assert target_B_NumTokens_C_shape == (bs, NumTokens_curr, C_out)

        # *--- 1. Upsample cached_output_at_KN_step to current scale's spatial resolution ---*
        cached_output_spatial_for_interp = cached_output_at_KN_step.permute(0, 3, 1, 2)     # permute: (bs, H_cache, W_cache, C) --> (bs, C, H_cache, W_cache) for interpolate
        y_k_cache_spatial = F.interpolate(
            cached_output_spatial_for_interp,
            size=(H_curr, W_curr),
            mode='bilinear',
            align_corners=False)
        # permute back and flatten: (bs, C, H_curr, W_curr) -> (bs, H_curr, W_curr, C) -> (bs, NumTokens_curr, C)
        y_k_cache_flat = y_k_cache_spatial.permute(0, 2, 3, 1).reshape(bs, NumTokens_curr, C_out)

        # *--- 2. Create the restored output tensor ---*
        restored_output = torch.zeros_like(y_k_cache_flat) # Initialize with zeros or from y_k_cache_flat

        # *--- 3. Scatter pivotal_output to their original locations ---*
        for b_idx in range(B_orig):
            # Conditional part
            restored_output[b_idx].scatter_(dim=0, index=pivotal_indices[b_idx].unsqueeze(-1).expand(-1, C_out), src=pivotal_output[b_idx])
            if bs > B_orig: # Unconditional part for CFG
                restored_output[b_idx + B_orig].scatter_(dim=0, index=pivotal_indices[b_idx].unsqueeze(-1).expand(-1, C_out), src=pivotal_output[b_idx + B_orig])

        # *--- 4. Fill pruned slots using y_k_cache_flat ---*
        # create a mask for pivotal token locations
        pivotal_mask_flat = torch.zeros(B_orig, NumTokens_curr, dtype=torch.bool, device=pivotal_output.device)
        for b_idx in range(B_orig):
            pivotal_mask_flat[b_idx].scatter_(dim=0, index=pivotal_indices[b_idx], src=torch.ones_like(pivotal_indices[b_idx], dtype=torch.bool))

        if bs > B_orig: # Expand mask for CFG
            pivotal_mask_flat_eff = torch.cat([pivotal_mask_flat, pivotal_mask_flat], dim=0)
        else:
            pivotal_mask_flat_eff = pivotal_mask_flat

        pruned_mask_flat_eff = ~pivotal_mask_flat_eff.unsqueeze(-1) # (bs, NumTokens_curr, 1)

        # Fill pruned locations
        restored_output = torch.where(pruned_mask_flat_eff, y_k_cache_flat, restored_output)

        return restored_output

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,    # [(1, 1, 1), (1, 2, 2), (1, 4, 4), ..., (1, 48, 48), (1, 64, 64)]
        label_B_or_BLT=None,    # tuple(torch.Size([9, 2048]), list[9], tensor[0, 9] shape=[2])
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,    # top_k=900, top_p=0.97
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,               # returns_vemb=1
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],     # cfg_insertion_layer=[0], CFG to Transformer layer: 0 for logits; 1 for probability, 负值表示从末尾开始的第n层
        vae_type=0, softmax_merge_topk=-1, ret_img=False,   # vae_type=32, ret_img=True
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,   # True for infra, to start KV-Cache
        save_img_path=None,
        sampling_per_bits=1,
    ):   # returns List[idx_Bl]
        
        # *====== FastVAR: Determine K_total and K_N_step_idx for caching ======*
        K_total_scales = len(scale_schedule)
        if self.fastvar_enabled and K_total_scales <= self.fastvar_cache_step_N :
            print(f"Warning: FastVAR enabled, but K_total_scales ({K_total_scales}) <= fastvar_cache_step_N ({self.fastvar_cache_step_N}). Disabling FastVAR for this run.")
            _fastvar_active = False
        else:
            _fastvar_active = self.fastvar_enabled

        if _fastvar_active:
            self.fastvar_K_N_caching_step_idx = K_total_scales - self.fastvar_cache_step_N - 1
            if self.fastvar_K_N_caching_step_idx < 0: # Not enough steps for full structure + texture stage
                print(f"Warning: Not enough scales for FastVAR structure/texture split. \
                      Caching step index {self.fastvar_K_N_caching_step_idx} < 0. Disabling FastVAR.")
                _fastvar_active = False
            # else:
            #     print(f"FastVAR: Total scales K={K_total_scales}, N={self.fastvar_cache_step_N}. \
            #           Caching at scale index {self.fastvar_K_N_caching_step_idx}.")
            self.fastvar_cached_output = None # Reset cache
            self.fastvar_cache_resolution = None
        # ======================================================================

        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)     # for each scale has cfg and tau
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:     # default: 0
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        # ------ text prompt process: for CFG ------
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)                              # [2*t_len, text_dim(2048)]
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)      # tensor[0, 9, 18], adjust the cumulative seq_len of 2*bs
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B
        # ------------------------------------------

        kv_compact = self.text_norm(kv_compact)     # FastRMSNorm
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))        # sos: [2, v_dim(2048)], cond_BD: [2, v_dim]
        kv_compact = self.text_proj_for_ca(kv_compact)                                          # kv_compact: [2*t_len, v_dim], linear -> tanh -> linear
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)      # [bs, 1, v_dim], Transformer first scale input

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()      # shared_ada_lin: silu -> SharedAdaLin; gss: gamma, scale, shift for AdaLN
        accu_BChw, cur_L, ret = None, 0, []     # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []      # Bl: 存储每个尺度预测的 token indices的列表; Bld: 存储每个尺度预测的逐位 token 标签的列表

        if inference_mode:      # start Transformer block's attn KV-Cache
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
        
        # ------ where and how to use CFG ------
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        # ---------------------------------------
        
        num_stages_minus_1 = len(scale_schedule) - 1
        summed_codes = 0
        # *------------------------------------
        # *------ scale auto-regressive ------*
        # *------------------------------------
        for si, pn in enumerate(scale_schedule):        # si: i-th segment, pn: current scale patch number (patch_t, patch_h, patch_w)
            print(f'{si=}, {pn=}')                      # for debug
            cfg = cfg_list[si]                          # get current scale si's CFG

            if si >= trunk_scale: break                 # trunk_scale=1000
            cur_L += np.array(pn).prod()                # current cumulative token length

            # ====== if the current scale is texture fill stage ======
            is_texture_filling_stage = _fastvar_active and si > self.fastvar_K_N_caching_step_idx
            pivotal_indices_for_current_scale = None        # (B_orig, NumPivotal)
            # input_to_var_module = last_stage                # This is (bs, NumTokensPrevScaleOutput_OR_SOS, C)

            if is_texture_filling_stage:
                pruning_idx_in_list = si - (self.fastvar_K_N_caching_step_idx + 1)
                pruning_ratio = self.fastvar_pruning_ratios[pruning_idx_in_list]
                num_tokens_current_scale_total = pn[0] * pn[1] * pn[2]
                keep_k_tokens = round(num_tokens_current_scale_total * (1.0 - pruning_ratio))
            # ========================================================

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:      # ? test default = 0, no use
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)
            
            if is_texture_filling_stage and (keep_k_tokens == 0 or pruning_ratio == 1.0):      # skip current scale
                if self.fastvar_cached_output is None: # Should not happen if logic is correct
                    raise ValueError("FastVAR cache is None during a skipped texture filling stage.")
                
                _H_cache, _W_cache = self.fastvar_cache_resolution
                _cached_for_interp = (self.fastvar_cached_output.reshape(bs, _H_cache, _W_cache, -1)
                                      .permute(0,3,1,2))
                interp_output_spatial_current = F.interpolate(_cached_for_interp,
                                                              size=(pn[1], pn[2]),
                                                              mode='bilinear',
                                                              align_corners=False)
                last_stage = (interp_output_spatial_current
                              .permute(0,2,3,1)
                              .reshape(bs, num_tokens_current_scale_total, -1))
            
            else:
                # *====== VAR module forward pass (on full or pivotal tokens) ======*
                # temp_last_stage = input_to_var_module # This will be modified by transformer blocks

                # *------ Transformer blocks ------*
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                layer_idx = 0
                for block_idx, b in enumerate(self.block_chunks):
                    # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                    if self.add_lvl_embeding_only_first_block and block_idx == 0:
                        # last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                        # -->
                        if pivotal_indices_for_current_scale is not None:
                            last_stage = self.add_lvl_embeding_pivotal(last_stage, si, scale_schedule, pivotal_indices_for_current_scale, need_to_pad=need_to_pad)
                        else:
                            last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    if not self.add_lvl_embeding_only_first_block:
                        # last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                        # -->
                        if pivotal_indices_for_current_scale is not None:
                            last_stage = self.add_lvl_embeding_pivotal(last_stage, si, scale_schedule, pivotal_indices_for_current_scale, need_to_pad=need_to_pad)
                        else:
                            last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                    
                    for m in b.module:
                        # ------ add for vis_attn_map ------
                        # self.vis_attn_map.set_cur_scale(pn)
                        # self.vis_attn_map.set_cur_block(layer_idx)
                        # ----------------------------------
                        # last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si,
                        #                vis_attn_map=self.vis_attn_map)
                        # --> 
                        # *====== FastVAR: Pivotal Token Selection (PTS) for Texture Filling Stage ======*
                        if is_texture_filling_stage:
                            pivotal_indices_for_current_scale = self._pivotal_token_selection(
                                last_stage[:B],        # PTS on the conditional part only (B, NumTokens, C)
                                (pn[1], pn[2]),                 # Assuming t=1 for spatial shape
                                keep_k_tokens)                  # torch.Size([B, num_pivotal_tokens])
                            # Gather pivotal tokens
                            gathered_input_list = []
                            for b_idx_gather in range(bs):      # bs could be 2*B
                                orig_b_idx_for_indices = b_idx_gather % B # pivotal_indices is (B, N_piv)
                                gathered_input_list.append(last_stage[b_idx_gather].gather(dim=0, 
                                                                                           index=pivotal_indices_for_current_scale[orig_b_idx_for_indices].unsqueeze(-1).expand(-1, self.C)))
                            last_stage = torch.stack(gathered_input_list, dim=0) # (bs, NumPivotal, C)
                            # print(f"FastVAR: Scale {si}, Pruning Ratio {pruning_ratio:.2f}, \
                            #       Kept {keep_k_tokens}/{num_tokens_current_scale_total} tokens.")

                        last_stage = m(
                            x=last_stage,
                            cond_BD=cond_BD_or_gss,
                            ca_kv=ca_kv,
                            attn_bias_or_two_vector=None,
                            attn_fn=attn_fn,
                            scale_schedule=scale_schedule,
                            rope2d_freqs_grid=self.rope2d_freqs_grid,
                            scale_ind=si,
                            vis_attn_map=self.vis_attn_map
                        )
                        if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                            # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                            last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                            last_stage = torch.cat((last_stage, last_stage), 0)
                        layer_idx += 1

                        # *====== FastVAR: Cached Token Restoration (CTR) ======
                        output_from_var_module = last_stage    # This is (bs, NumPivotal_or_Full, C)
                        if is_texture_filling_stage and pivotal_indices_for_current_scale is not None and pruning_ratio < 1.0:
                            if self.fastvar_cached_output is None:
                                raise ValueError("FastVAR cache is None during texture filling stage.")
                            
                            # Ensure cached output is in spatial shape (B,H,W,C_vae) for interpolation by CTR
                            _H_cache, _W_cache = self.fastvar_cache_resolution
                            _cached_transformer_output_for_ctr = self.fastvar_cached_output.reshape(bs, _H_cache, _W_cache, self.C)     # torch.Size([bs, 24, 24, C])

                            output_from_var_module_full = self._cached_token_restoration(
                                pivotal_output=output_from_var_module, # (bs, NumPivotal, C)
                                pivotal_indices=pivotal_indices_for_current_scale, # (B, NumPivotal)
                                cached_output_at_KN_step=_cached_transformer_output_for_ctr, # (bs, H_cache, W_cache, C)
                                current_scale_spatial_shape=(pn[1], pn[2]),
                                target_B_NumTokens_C_shape=(bs, pn[0]*pn[1]*pn[2], self.C)
                            )
                        else:   # Structure stage or FastVAR disabled or no pruning for this step
                            output_from_var_module_full = output_from_var_module    # No restoration needed, it's already full
                        
                        last_stage = output_from_var_module_full
                        # =======================================================

            # *====== FastVAR: Cache the output for the K-N step ======*
            # todo: per-layer cache
            if _fastvar_active and si == self.fastvar_K_N_caching_step_idx:
                self.fastvar_cached_output = output_from_var_module_full.detach().clone() # (bs, NumTokens_at_KN, C)
                self.fastvar_cache_resolution = (pn[1], pn[2]) # (H,W) at K-N
                # print(f"FastVAR: Cached output at scale index {si}, \
                #       shape {self.fastvar_cached_output.shape}, resolution {self.fastvar_cache_resolution}")
            # =========================================================
            
            if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])       # torch.Size([1, 1, 64]), get 2*bs(cond and uncond)'s logits
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            # ------ sample from logits_BlV, get current scale predicted tokens ------
            if self.use_bit_label:      # use_bit_label = 1
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)      # torch.Size([1, 1, 32])
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            
            # ------ process sampled tokens, prepare next AR step's input ------
            if vae_type != 0:           # vae_type = 32
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)   # mode is 'trilinear'
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            # -------------------------------------------------------------------

            if si != num_stages_minus_1:    # not last scale
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:  # disable attn's KV-Cache --> free memory
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        # ------ decode latent feature to img ------
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img


if __name__ == '__main__':

    import argparse
    import time
    import random
    from torch import autocast
    from tools.run_infinity import load_visual_tokenizer, load_transformer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g_seed = random.randint(0, 10000)

    args=argparse.Namespace(
        pn='1M',
        model_path=None,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=None,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=None,
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=0,
        bf16=1,
        save_file='tmp.jpg',
        # 
        enable_model_cache=0)

    # load vae
    vae = load_visual_tokenizer(args)
    # infinity = load_transformer(vae, args)
    # --> do not load transformer weight, for fast debug
    if args.model_type == 'infinity_2b':
        model_kwargs = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
        infinity_test: FastVAR_Infinity = FastVAR_Infinity(
            vae_local=vae, text_channels=2048, text_maxlen=512,
            shared_aln=True, raw_scale_schedule=None,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=0,
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            pn='1M',
            apply_spatial_patchify=0,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={args.bf16}')

        if args.bf16:
            for block in infinity_test.unregistered_blocks:
                block.bfloat16()

        infinity_test.eval()
        infinity_test.requires_grad_(False)

        infinity_test.cuda()
        torch.cuda.empty_cache()

        # *------ t2i forward ------*
        h_div_w = 1/1 # aspect ratio, height:width
        h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
        scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

        # dummy_input
        text_cond_tuple = (
            torch.randn(size=[9, 2048], device=device, dtype=torch.float32),
            [9],
            torch.tensor([0, 9], device=device, dtype=torch.int32),
            9
        )

        # warmup GPU
        warmup_iterations = 10
        print(f"Starting GPU warm-up for {warmup_iterations} iterations...")
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
            for _ in range(warmup_iterations):
                _, _, _ = infinity_test.autoregressive_infer_cfg(
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
                    B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                    returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                    cfg_exp_k=0.0, cfg_insertion_layer=[0],
                    vae_type=32, softmax_merge_topk=-1,
                    ret_img=True, trunk_scale=1000,
                    gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                    sampling_per_bits=1,
                )
                torch.cuda.synchronize(device=device)
        print("GPU warm-up finished.")

        # Influence speed test
        num_test_iterations = 100
        timings = []

        print(f"Starting inference speed test for {num_test_iterations} iterations...")
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
            stt = time.time()
            for i in range(num_test_iterations):
                start_time = time.perf_counter()    # for accurate timing

                _, _, img_list = infinity_test.autoregressive_infer_cfg(
                    vae=vae,
                    scale_schedule=scale_schedule,
                    label_B_or_BLT=text_cond_tuple, g_seed=g_seed,
                    B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                    cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                    returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                    cfg_exp_k=0.0, cfg_insertion_layer=[0],
                    vae_type=32, softmax_merge_topk=-1,
                    ret_img=True, trunk_scale=1000,
                    gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                    sampling_per_bits=1,
                )
                torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time

                end_time = time.perf_counter()
                timings.append(end_time - start_time)
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{num_test_iterations} done.")

        print("Inference speed test finished.")
        
        batch_size = 1
        avg_latency = sum(timings) / len(timings)
        std_latency = torch.tensor(timings).std().item()
        throughput = batch_size / avg_latency if avg_latency > 0 else float('inf')

        print(f"\n--- Inference Performance ---")
        print(f"Batch Size: {batch_size}")
        print(f"Average Latency: {avg_latency * 1000:.2f} ms")
        print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
        print(f"Throughput: {throughput:.2f} samples/sec")

        # --- print first and last times to observe warmup effect ---
        # print("\nFirst 5 latencies (ms):")
        # for t in timings[:5]:
        #     print(f"{t*1000:.2f}")

        # print("\nLast 5 latencies (ms):")
        # for t in timings[-5:]:
        #     print(f"{t*1000:.2f}")
        
        img = img_list[0]
        print(f"{img.shape=}")


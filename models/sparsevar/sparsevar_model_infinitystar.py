"""SparseVAR (ICCV'2025) applied to InfinityStar video generation.

Reproduction of the FastSTAR comparison baseline: the image-Infinity SparseVAR
low-frequency token exclusion (models/sparsevar/sparsevar_model.py) is applied
as-is across the spatiotemporal pyramid of InfinityStar, targeting the final 4
scales. Code organization mirrors models/faststar/faststar_model.py: all
sparsity lives in the overridden ar_infer_infinity_elegant.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from timm.models import register_model

from models.infinitystar.apg import normalized_guidance
from models.infinitystar.infinitystar_model import TIMM_KEYS, InfinityStar
from models.schedules.dynamic_resolution import \
    get_first_full_spatial_size_scale_index

from .basic_sparsevar_infinitystar import (
    average_branch_maps,
    dynamic_keep_ratio,
    lf_anchor_video,
    restore_pruned_tokens,
    token_mse,
    zero_nonrecoverable_codes,
)


@register_model
def sparsevar_infinitystar(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096 // 128,
                           num_key_value_heads=4096 // 128 // 4, mlp_ratio=4, drop_path_rate=0, **kwargs):
    return SparseVarStar(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


class SparseVarStar(InfinityStar):
    """SparseVAR-accelerated InfinityStar inference (FastSTAR baseline)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SparseVAR-specific init (defaults follow the image-Infinity setup,
        # utils/arg_util.py sparsevar_*)
        sparsevar_args = self.other_args
        self.sparsevar_compress_ratio = getattr(sparsevar_args, 'sparsevar_compress_ratio', 0.6)
        self.sparsevar_local_window_size = getattr(sparsevar_args, 'sparsevar_local_window_size', 4)
        self.sparsevar_specific_mse_layer = getattr(sparsevar_args, 'sparsevar_specific_mse_layer', 3)
        self.sparsevar_beta = getattr(sparsevar_args, 'sparsevar_beta', 0.8)
        self.sparsevar_target_scales = set(sparsevar_args.sparsevar_target_scale_list(self.scale_schedule))
        self.sparsevar_force_keep_anchors = bool(int(getattr(sparsevar_args, 'sparsevar_force_keep_anchors', 1)))
        # {target scale: nominal keep ratio} for the FastSTAR fixed-ratio
        # protocol; empty dict = SparseVAR-native dynamic-threshold mode.
        self.sparsevar_nominal_keep_ratios = sparsevar_args.sparsevar_nominal_keep_ratios(self.sparsevar_target_scales)

        if not 0 <= self.sparsevar_specific_mse_layer < self.num_block_chunks:
            raise ValueError(
                f"sparsevar_specific_mse_layer must be in [0, {self.num_block_chunks - 1}], "
                f"got {self.sparsevar_specific_mse_layer}."
            )
        if self.apply_spatial_patchify:
            raise NotImplementedError(
                "SparseVAR-InfinityStar assumes apply_spatial_patchify=0 "
                "(latent code grid must match the token grid)."
            )

        if self.sparsevar_nominal_keep_ratios:
            ratio_mode = 'nominal pruning ' + str({
                si: round(1.0 - keep, 4) for si, keep in sorted(self.sparsevar_nominal_keep_ratios.items())})
        else:
            ratio_mode = f'dynamic (compress_ratio={self.sparsevar_compress_ratio})'
        print(
            f"[SparseVAR->InfinityStar] (ICCV'2025)\n"
            f'    target_scales={sorted(self.sparsevar_target_scales)}\n'
            f'    ratio_mode={ratio_mode}\n'
            f'    force_keep_anchors={self.sparsevar_force_keep_anchors}\n'
            f'    local_window_size={self.sparsevar_local_window_size}\n'
            f'    specific_mse_layer={self.sparsevar_specific_mse_layer}\n'
            f'    beta={self.sparsevar_beta}\n'
            f'    final_iteration_full={bool(int(getattr(sparsevar_args, "sparsevar_final_iteration_full", 0)))}\n'
        )

    @torch.no_grad()
    def ar_infer_infinity_elegant(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None,
        g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        low_vram_mode=False,
        args=None,
        get_visual_rope_embeds=None,
        context_info=None,
        return_summed_code_only=False,
        # ------ for attn map vis -----
        attn_map_recorder=None,
        cfg_similarity_recorder=None,
        **kwargs,
    ):
        assert self.scale_schedule == scale_schedule
        from models.schedules.infinity_elegant import interpolate

        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1  # CFG / APG mutual exclusion
        device = label_B_or_BLT[0].device

        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[
            -1]  # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT,
                                                           vae_scale_schedule, text_token_only=False,
                                                           text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]

        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)

        # TODO: ------ for attn map vis ------
        self.set_attention_map_recorder(attn_map_recorder)

        if attn_map_recorder is not None:
            attn_map_recorder.start_inference(scale_schedule=scale_schedule)
        if cfg_similarity_recorder is not None:
            cfg_similarity_recorder.start_inference(scale_schedule=scale_schedule)

        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(
            video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(
            image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'

        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (
                    len(scale_schedule) // len(video_scale_repetition) - 1) + 1  # +1 is prefix text token forward step
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        #* Count the suffix over actual visual forward steps, e.g. scale_27_repeat1, scale_28_repeat0.
        visual_repeat_steps = []
        for step_si in range(len(scale_schedule)):
            rel_step_si = step_si % scales_in_one_clip
            if step_si < scales_in_one_clip:
                step_repeat_times = image_scale_repetition[rel_step_si]
            else:
                step_repeat_times = video_scale_repetition[rel_step_si]
            step_infer_repeat_times = min(int(step_repeat_times), args.max_repeat_times)
            for step_repeat_idx in range(step_infer_repeat_times):
                visual_repeat_steps.append((step_si, step_repeat_idx))
        drop_uncond_steps = set(visual_repeat_steps[-self.drop_uncond_last_scales:]) if self.drop_uncond_last_scales > 0 else set()
        active_branch_repeat = bs // B

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype,
                                device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype,
                                device=prefix_tokens.device)  # torch.Size([1, 64, 1, 1, 1])

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, active_branch_repeat)  # torch.Size([2, 1, 4096])

        # ------ text tokens forward ------
        rope_cache = self.rope2d_freqs_grid['freqs_text'][
            :, :, :, :, :text_maxlen_this_iter]  # torch.Size([2, 1, 1, 1, text_len, 64])
        last_stage = prefix_tokens  # torch.Size([2, text_len, 4096])
        pbar.update(1)
        if attn_map_recorder is not None:
            attn_map_recorder.set_step(step_key='t0', file_label='scale_t0')
        # get text KV cache
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(
                x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask,
                attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache,
                scale_ind='t0', context_info=context_info, last_repetition_step=True)

        # ---------------------------------------
        # -------- visual tokens forward --------
        # ---------------------------------------
        ref_text_scale_inds = ['t0']
        last_stage = sos_token  # torch.Size([2, 1, 4096])
        cum_scales = 0  # real repetition-aware scale
        # Per-target-scale inputs prepared at the previous scale:
        # {si: {'mse': [B, L_prev], 'keep_ratio': float, 'logits': [B, L_prev, V], 'prev_pn': (t, h, w)}}
        sparsevar_pending = {}
        for si, pn in enumerate(scale_schedule):  # si: i-th segment

            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip:  # image
                repeat_times = image_scale_repetition[si % scales_in_one_clip]  # ! repeat time setting
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[si % scales_in_one_clip]
                target_pn = vae_scale_schedule[-1]

            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)

            # -------- SparseVAR: build the token plan once per target scale --------
            token_plan = None
            if si in self.sparsevar_target_scales:
                pending = sparsevar_pending.pop(si, None)
                if pending is None or pending.get('logits') is None:
                    raise RuntimeError(
                        f"SparseVAR token plan inputs for target scale {si} were not prepared by scale {si - 1}."
                    )
                high_freq_idx, low_freq_idx, recoverable_mask, low_to_anchor_idx = lf_anchor_video(
                    pending['logits'], pending['mse'], pending['prev_pn'], pn,
                    keep_ratio=pending['keep_ratio'],
                    window=self.sparsevar_local_window_size,
                    beta=self.sparsevar_beta,
                    force_keep_anchors=self.sparsevar_force_keep_anchors,
                )
                if high_freq_idx.shape[0] > 1 and not torch.equal(
                        high_freq_idx, high_freq_idx[:1].expand_as(high_freq_idx)):
                    raise ValueError("SparseVAR requires a shared token plan across the inference batch.")
                token_plan = dict(
                    high_freq_idx=high_freq_idx,
                    low_freq_idx=low_freq_idx,
                    recoverable_mask=recoverable_mask,
                    low_to_anchor_idx=low_to_anchor_idx,
                    keep_indices=high_freq_idx[0],  # 1-D, shared across CFG branches
                )

            for repeat_idx in range(infer_repeat_times):
                drop_uncond_this_step = (si, repeat_idx) in drop_uncond_steps and active_branch_repeat > 1
                if drop_uncond_this_step:
                    last_stage = last_stage[:B]
                    active_branch_repeat = 1
                    self.keep_cond_branch_in_kv_cache(B)

                # --- visual RoPE ---
                #       Recalculate visual RoPE for each repetition
                rope_cache = get_visual_rope_embeds(
                    self.rope2d_freqs_grid, scale_schedule, si, cum_scales + repeat_idx,
                    device, args, context_info, first_full_spatial_size_scale_index)
                pbar.update(1)
                # TODO: ------ for attn map vis ------
                if attn_map_recorder is not None:
                    attn_map_recorder.set_step(
                        step_key=f'scale_{si:02d}_repeat{repeat_idx}',
                        file_label=f'scale_{si:02d}_repeat{repeat_idx}',
                    )

                last_repetition_step = (repeat_idx == (infer_repeat_times - 1))
                pruning_this_step = (
                    token_plan is not None
                    and args.sparsevar_should_prune(
                        si,
                        repeat_idx,
                        infer_repeat_times,
                        self.sparsevar_target_scales,
                    )
                )
                full_seq_len = last_stage.shape[1]
                keep_indices = None
                x_origin = None
                if pruning_this_step:
                    expected_seq_len = int(np.prod(pn))
                    if full_seq_len != expected_seq_len:
                        raise ValueError(
                            f"SparseVAR expected {expected_seq_len} tokens at scale {si}, got {full_seq_len}."
                        )
                    if rope_cache.shape[4] != full_seq_len:
                        raise ValueError(
                            f"SparseVAR token/RoPE length mismatch: {full_seq_len} != {rope_cache.shape[4]}."
                        )
                    keep_indices = token_plan['keep_indices']
                    x_origin = last_stage
                    last_stage = last_stage.index_select(1, keep_indices)
                    rope_cache = rope_cache.index_select(4, keep_indices)

                    if bool(int(args.sparsevar_log_tokens)):
                        print(
                            f'[SparseVAR] scale={si} repeat={repeat_idx} '
                            f'tokens={keep_indices.numel()}/{full_seq_len} '
                            f'keep_ratio={keep_indices.numel() / full_seq_len:.4f}'
                        )
                elif token_plan is not None and bool(int(args.sparsevar_log_tokens)):
                    print(
                        f'[SparseVAR] scale={si} repeat={repeat_idx} '
                        f'full-token refinement tokens={full_seq_len}'
                    )

                # MSE change map of a specific block chunk feeds the next
                # target scale's token plan (image impl: sparsevar_model.py:432, 451-466).
                need_mse = (si + 1) in self.sparsevar_target_scales and last_repetition_step
                chunk_in = chunk_out = None
                for block_idx, b in enumerate(block_chunks):
                    if need_mse and block_idx == self.sparsevar_specific_mse_layer:
                        chunk_in = last_stage
                    last_stage = b(
                        x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask,
                        attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache,
                        scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step,
                        ref_text_scale_inds=ref_text_scale_inds
                    )
                    if need_mse and block_idx == self.sparsevar_specific_mse_layer:
                        chunk_out = last_stage

                if need_mse:
                    if pruning_this_step:
                        # Scatter pruned-length chunk features into the full-length input
                        # buffer: excluded positions get identical before/after values
                        # (MSE = 0), so they stay low-frequency at the next scale.
                        hr_expand = token_plan['high_freq_idx'].repeat(
                            x_origin.shape[0] // token_plan['high_freq_idx'].shape[0], 1
                        ).unsqueeze(-1).expand(-1, -1, x_origin.shape[-1])
                        x_after = x_origin.to(chunk_out).scatter(1, hr_expand, chunk_out)
                        x_before = x_origin.to(chunk_in).scatter(1, hr_expand, chunk_in)
                    else:
                        x_after = chunk_out
                        x_before = chunk_in
                    mse_BL = average_branch_maps(token_mse(x_before, x_after), B)
                    nominal_keep = self.sparsevar_nominal_keep_ratios.get(si + 1)
                    sparsevar_pending[si + 1] = dict(
                        mse=mse_BL,
                        keep_ratio=(nominal_keep if nominal_keep is not None
                                    else dynamic_keep_ratio(mse_BL, self.sparsevar_compress_ratio)),
                        prev_pn=tuple(pn),
                        logits=None,
                    )
                    if bool(int(args.sparsevar_log_tokens)):
                        print(
                            f'[SparseVAR] prepared plan scale={si}->{si + 1} '
                            f'keep_ratio={sparsevar_pending[si + 1]["keep_ratio"]:.4f} '
                            f'({"nominal" if nominal_keep is not None else "dynamic"})'
                        )

                if pruning_this_step:
                    # Anchor-copy restore: recoverable low-frequency tokens take the
                    # matched anchor's transformer output; non-recoverable ones keep
                    # their input embedding (their codes are zeroed below).
                    last_stage = restore_pruned_tokens(
                        x_origin,
                        last_stage,
                        token_plan['high_freq_idx'],
                        token_plan['low_freq_idx'],
                        token_plan['recoverable_mask'],
                        token_plan['low_to_anchor_idx'],
                    )

                if cfg_similarity_recorder is not None and last_repetition_step:
                    cfg_similarity_recorder.capture_last_stage(
                        scale_index=si,
                        repeat_index=repeat_idx,
                        last_stage=last_stage,
                        batch_size=B,
                        is_semantic_scale=rel_si_in_one_clip < args.semantic_scales,
                    )

                logits_BlV = self.get_logits_during_infer(last_stage,
                                                          is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(
                    1 / tau_list[si])
                if cfg != 1:
                    # print(f'add cfg on add_cfg_on_logits')
                    if active_branch_repeat == 1:
                        logits_BlV = logits_BlV[:B]
                    elif args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg,
                                                          momentum_buffer=None, eta=0,
                                                          norm_threshold=args.apg_norm_threshold)
                        # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                if need_mse:
                    # Post-CFG logits drive the anchor matching at scale si+1
                    # (image impl: logits_BlV_copy, sparsevar_model.py:486-487).
                    sparsevar_pending[si + 1]['logits'] = logits_BlV

                # -------- bit-wise sample --------
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1)  # [B, thwd or thw4d, 2]
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1,
                                            replacement=True, generator=rng).view(tmp_bs, -1)  # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    """Reshape the bit labels back to the spatiotemporal grid of the current scale
                    """
                    item = item.reshape(tmp_bs, tmp_seq_len, -1)  # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1)  # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify:  # unpatchify operation
                        item = item.permute(0, 1, 4, 2, 3)  # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2)  # [B, t, d, 2h, 2w]
                        item = item.permute(0, 1, 3, 4, 2)  # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)

                # for I2V / reference-conditioned inference
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[cum_scales + repeat_idx]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]

                # -------- bit-label --> latent code -------
                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    if pruning_this_step:
                        # Zero non-recoverable low-frequency codes BEFORE upsampling;
                        # zeros survive the interpolate because the 480p path runs
                        # pure interpolation (use_learnable_dim_proj=0).
                        codes = zero_nonrecoverable_codes(
                            codes, token_plan['low_freq_idx'], token_plan['recoverable_mask'])
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn),
                                        mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer,
                                        is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    if pruning_this_step:
                        codes = zero_nonrecoverable_codes(
                            codes, token_plan['low_freq_idx'], token_plan['recoverable_mask'])
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                # -------- residual accumulation (base InfinityStar semantics) --------
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes

                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si],
                                               mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, active_branch_repeat)

            # After a scale is completed, update the real scale counter
            cum_scales += repeat_times

            # -------- prepare next scale input --------
            if si < len(scale_schedule) - 1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si + 1]),
                                                        device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si + 1]),
                                                        device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si + 1],
                                               mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, active_branch_repeat)

        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)

        # TODO: ------ for attn map vis ------
        self.set_attention_map_recorder(None)

        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img

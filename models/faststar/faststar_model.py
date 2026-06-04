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

from .basic_faststar import (compute_st_score, partial_update,
                             save_pruning_mask, topk_pruning_mask)


@register_model
def faststar_qwen8b(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096 // 128,
                    num_key_value_heads=4096 // 128 // 4, mlp_ratio=4, drop_path_rate=0, **kwargs):
    return FastStar(
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


class FastStar(InfinityStar):
    """Compatibility wrapper for FastSTAR-enabled InfinityStar inference."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FastSTAR-specific init
        faststar_args = self.other_args

        self.faststar_prune_ratio_by_scale = faststar_args.faststar_prune_ratio_by_scale(self.scale_schedule)
        self.faststar_p_norm = faststar_args.faststar_p_norm_value()
        print(
            f'[FastSTAR]\n'
            f'    target_scales={list(self.faststar_prune_ratio_by_scale.keys())}\n'
            f'    prune_ratios={list(self.faststar_prune_ratio_by_scale.values())}\n'
            f'    p_norm={self.faststar_p_norm}\n'
            f'    final_iteration_full={bool(int(faststar_args.faststar_final_iteration_full))}\n'
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

        # faststar_prune_ratio_by_scale = args.faststar_prune_ratio_by_scale(scale_schedule)
        # faststar_p_norm = args.faststar_p_norm_value()
        # print(
        #     f'\n[FastSTAR]\n'
        #     f'    target_scales={list(faststar_prune_ratio_by_scale.keys())}\n'
        #     f'    prune_ratios={list(faststar_prune_ratio_by_scale.values())}\n'
        #     f'    p_norm={faststar_p_norm}\n'
        #     f'    final_iteration_full={bool(int(args.faststar_final_iteration_full))}\n'
        # )

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
        # drop_uncond_last_scales = max(int(getattr(args, "drop_uncond_last_scales", 0)), 0)
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
            for repeat_idx in range(infer_repeat_times):
                drop_uncond_this_step = (si, repeat_idx) in drop_uncond_steps and active_branch_repeat > 1
                # print(f'{(si, repeat_idx)=}')
                if drop_uncond_this_step:
                    # print(f'    {drop_uncond_this_step=}')
                    last_stage = last_stage[:B]
                    active_branch_repeat = 1
                    self.keep_cond_branch_in_kv_cache(B)
                # print(f'real scale ind is : {cum_scales+repeat_idx}')

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
                for block_idx, b in enumerate(block_chunks):
                    last_stage = b(
                        x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask,
                        attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache,
                        scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step,
                        ref_text_scale_inds=ref_text_scale_inds
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
                # print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape}')

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
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn),
                                        mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer,
                                        is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                # -------- residual accumulation --------
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                if args.faststar_should_prune(si, repeat_idx, infer_repeat_times, self.faststar_prune_ratio_by_scale):
                    previous_feature = summed_codes[-1]
                    current_feature = previous_feature + codes
                    previous_temporal_feature = None
                    if len(summed_codes) > 1:
                        previous_temporal_feature = F.interpolate(
                            summed_codes[-2],
                            size=target_pn,
                            mode=vae.quantizer.z_interplote_up,
                        )
                    st_score = compute_st_score(
                        previous_feature=previous_feature,
                        current_feature=current_feature,
                        previous_temporal_feature=previous_temporal_feature,
                        p_norm=self.faststar_p_norm,
                        temporal_fallback=args.faststar_first_clip_temporal_fallback,
                    )
                    pruning_mask = topk_pruning_mask(
                        st_score,
                        prune_ratio=self.faststar_prune_ratio_by_scale[si],
                        per_frame_topk=bool(int(args.faststar_per_frame_topk)),
                    )
                    summed_codes[-1] = partial_update(previous_feature, codes, pruning_mask)
                    if bool(int(args.faststar_log_masks)):
                        kept_tokens = int(pruning_mask.sum().item())
                        total_tokens = pruning_mask.numel()
                        print(
                            f'[FastSTAR] scale={si} repeat={repeat_idx} '
                            f'prune_ratio={self.faststar_prune_ratio_by_scale[si]:.2f} '
                            f'kept={kept_tokens}/{total_tokens} keep_ratio={kept_tokens / total_tokens:.4f}'
                        )
                    if bool(int(args.faststar_save_masks)):
                        save_path = save_pruning_mask(
                            pruning_mask,
                            args.faststar_mask_save_dir,
                            scale_index=si,
                            repeat_index=repeat_idx,
                        )
                        if bool(int(args.faststar_log_masks)):
                            print(f'[FastSTAR] saved pruning mask: {save_path} (+ .png visualization)')
                else:
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

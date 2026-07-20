import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from timm.models import register_model

from models.fastvar.basic_fastvar_infinitystar import FastVARSelfAttnBlock
from models.infinitystar.apg import normalized_guidance
from models.infinitystar.infinitystar_model import TIMM_KEYS, InfinityStar, MultipleLayers
from models.schedules.dynamic_resolution import \
    get_first_full_spatial_size_scale_index


@register_model
def fastvar_infinitystar(depth=36, block_chunks=6, embed_dim=4096, num_heads=4096 // 128,
                   num_key_value_heads=4096 // 128 // 4, mlp_ratio=4, drop_path_rate=0, **kwargs):
    return FastVAR_InfinityStar(
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


class FastVAR_InfinityStar(InfinityStar):
    """InfinityStar accelerated by FastVAR (ICCV'2025), the comparison baseline of the FastSTAR paper."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # FastVAR-specific init
        fastvar_args = self.other_args
        self.fastvar_prune_ratio_by_scale = fastvar_args.fastvar_prune_ratio_by_scale(self.scale_schedule)
        target_scales = sorted(self.fastvar_prune_ratio_by_scale.keys())
        self.fastvar_cache_scale = fastvar_args.fastvar_cache_scale_index(target_scales)
        self.fastvar_prune_layers = set(fastvar_args.fastvar_prune_layer_list(self.depth))
        # ratio == 1 scales are skipped entirely (original FastVAR's 100% pruning / skip last scales)
        self.fastvar_skip_scales = {
            si for si, ratio in self.fastvar_prune_ratio_by_scale.items() if ratio >= 1.0}
        restore_interp_mode = fastvar_args.fastvar_restore_interp_mode
        per_frame_pts = bool(int(fastvar_args.fastvar_per_frame_pts))

        # Rebuild the transformer blocks with FastVAR-enabled blocks. Parameter names and
        # structure match SelfAttnBlock, so the pretrained checkpoint loads unchanged.
        self.unregistered_blocks = []
        for layer_idx in range(self.depth):
            block = FastVARSelfAttnBlock(
                embed_dim=self.C,
                cond_dim=self.D,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                mlp_ratio=self.mlp_ratio,
                use_flex_attn=self.use_flex_attn,
                pad_to_multiplier=self.pad_to_multiplier,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                mask_type=self.other_args.mask_type,
                context_frames=self.other_args.context_frames,
                steps_per_frame=self.other_args.steps_per_frame,
                arch=self.arch,
                qwen_qkvo_bias=self.qwen_qkvo_bias,
                inject_sync=self.other_args.inject_sync,
                prune_layer=layer_idx in self.fastvar_prune_layers,
                restore_interp_mode=restore_interp_mode,
                per_frame_pts=per_frame_pts,
            )
            block.layer_idx = layer_idx
            block.attn.layer_idx = layer_idx
            self.unregistered_blocks.append(block)

        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
            assert self.blocks[0] is self.unregistered_blocks[0]
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(
                    self.unregistered_blocks,
                    self.num_blocks_in_a_chunk,
                    i * self.num_blocks_in_a_chunk,
                ))
            assert self.block_chunks[0].module[0] is self.unregistered_blocks[0]

        print(
            f"\n[FastVAR (ICCV'2025) x InfinityStar]\n"
            f'    target_scales={target_scales}\n'
            f'    prune_ratios={[self.fastvar_prune_ratio_by_scale[s] for s in target_scales]}\n'
            f'    cache_scale={self.fastvar_cache_scale}\n'
            f'    skip_scales={sorted(self.fastvar_skip_scales)}\n'
            f'    prune_layers={sorted(self.fastvar_prune_layers)}\n'
            f'    final_iteration_full={bool(int(fastvar_args.fastvar_final_iteration_full))}\n'
            f'    per_frame_pts={per_frame_pts}\n'
            f'    restore_interp_mode={restore_interp_mode}\n'
        )

    def set_fastvar_step_state(self, scale_ind, x_shape, pruning_this_step):
        """Inject the per-(scale, repeat) FastVAR state into every block.

        MultipleLayers.forward has no kwargs pass-through, so the pruning decision is
        injected as block state instead of extending the block forward signature.
        """
        prune_ratio = self.fastvar_prune_ratio_by_scale.get(scale_ind, 0.0) if pruning_this_step else 0.0
        cache_this_step = scale_ind == self.fastvar_cache_scale
        for block in self.unregistered_blocks:
            block.fastvar_prune_this_step = pruning_this_step
            block.fastvar_prune_ratio = prune_ratio
            block.fastvar_x_shape = tuple(x_shape) if x_shape is not None else None
            block.fastvar_cache_this_step = cache_this_step

    def reset_fastvar_caches(self):
        for block in self.unregistered_blocks:
            block.reset_fastvar_cache()

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
        self.reset_fastvar_caches()

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

        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        #* Count the suffix over actual visual forward steps, e.g. scale_27_repeat1, scale_28_repeat0.
        #* 100%-pruned (skipped) scales contribute no steps, so drop-uncond lands on executed steps.
        visual_repeat_steps = []
        for step_si in range(len(scale_schedule)):
            if step_si in self.fastvar_skip_scales:
                continue
            rel_step_si = step_si % scales_in_one_clip
            if step_si < scales_in_one_clip:
                step_repeat_times = image_scale_repetition[rel_step_si]
            else:
                step_repeat_times = video_scale_repetition[rel_step_si]
            step_infer_repeat_times = min(int(step_repeat_times), args.max_repeat_times)
            for step_repeat_idx in range(step_infer_repeat_times):
                visual_repeat_steps.append((step_si, step_repeat_idx))
        drop_uncond_steps = set(visual_repeat_steps[-self.drop_uncond_last_scales:]) if self.drop_uncond_last_scales > 0 else set()

        total_steps = len(visual_repeat_steps) + 1  # +1 is prefix text token forward step
        pbar = tqdm.tqdm(total=total_steps)
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
        self.set_fastvar_step_state(scale_ind='t0', x_shape=None, pruning_this_step=False)
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
            if si in self.fastvar_skip_scales:
                # FastVAR 100% pruning: skip this scale's forward passes entirely; the
                # accumulated feature map (already at target_pn) is interpolated as the
                # final output for these tokens, cf. the FastVAR paper's skip-last-scales.
                infer_repeat_times = 0
                if bool(int(args.fastvar_log_pruning)):
                    print(f'[FastVAR] scale={si} skipped (prune_ratio=1.00, 100% pruning)')
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

                # -------- FastVAR: inject the per-step pruning state into all blocks --------
                pruning_this_step = args.fastvar_should_prune(
                    si, repeat_idx, infer_repeat_times, self.fastvar_prune_ratio_by_scale)
                self.set_fastvar_step_state(scale_ind=si, x_shape=pn, pruning_this_step=pruning_this_step)
                if bool(int(args.fastvar_log_pruning)) and si in self.fastvar_prune_ratio_by_scale:
                    full_seq_len = last_stage.shape[1]
                    if pruning_this_step:
                        prune_ratio = self.fastvar_prune_ratio_by_scale[si]
                        token_kept = full_seq_len - int(full_seq_len * prune_ratio)
                        print(
                            f'[FastVAR] scale={si} repeat={repeat_idx} '
                            f'prune_ratio={prune_ratio:.2f} '
                            f'tokens={token_kept}/{full_seq_len} per prune layer '
                            f'token_keep_ratio={token_kept / full_seq_len:.4f}'
                        )
                    else:
                        print(
                            f'[FastVAR] scale={si} repeat={repeat_idx} '
                            f'full-token refinement tokens={full_seq_len}'
                        )

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

                # -------- residual accumulation (baseline behavior, no partial update) --------
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
        self.reset_fastvar_caches()  # free the CTR caches between inferences

        # TODO: ------ for attn map vis ------
        self.set_attention_map_recorder(None)

        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img

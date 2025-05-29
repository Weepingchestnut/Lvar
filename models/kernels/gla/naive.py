from typing import Optional

import torch
import torch.nn.functional as F


def ceildiv(a, b):
    return -(a // -b)


def naive_recurrent_gla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
):
    dtype = q.dtype
    q, k, v, gk = map(lambda x: x.transpose(1, 2).float(), (q, k, v, gk))
    B, H, T, K, V = *q.shape, v.shape[-1]
    o = torch.zeros_like(v)
    scale = K ** -0.5

    h = q.new_zeros(B, H, K, V, dtype=torch.float32)
    if initial_state is not None:
        h += initial_state.float()

    for i in range(T):
        q_i = q[:, :, i] * scale
        k_i = k[:, :, i]
        v_i = v[:, :, i]
        gk_i = gk[:, :, i].exp()
        kv_i = k_i[..., None] * v_i[..., None, :]
        h = h * gk_i[..., None] + kv_i
        o[:, :, i] = (q_i[..., None] * h).sum(-2)

    if not output_final_state:
        h = None
    return o.transpose(1, 2).to(dtype), h


def naive_chunk_gla(
    q_input: torch.Tensor,      # Query: (batch_size, num_heads, seq_len, head_dim_k)
    k_input: torch.Tensor,      # Key: (batch_size, num_heads, seq_len, head_dim_k)
    v_input: torch.Tensor,      # Value: (batch_size, num_heads, seq_len, head_dim_v)
    log_a_input: torch.Tensor,  # Log forget gate: (batch_size, num_heads, seq_len, head_dim_k)
    C: int,                     # Chunk size
    c: int                      # Sub-chunk size
) -> torch.Tensor:
    """
    PyTorch implementation of Gated Linear Attention forward pass with two-level chunking,
    based on Listing 1 in Appendix A.3 of "Gated Linear Attention Transformers
    with Hardware-Efficient Training".

    Args:
        q_input: Query tensor of shape (batch_size, num_heads, seq_len, head_dim_k)
        k_input: Key tensor of shape (batch_size, num_heads, seq_len, head_dim_k)
        v_input: Value tensor of shape (batch_size, num_heads, seq_len, head_dim_v)
        log_a_input: Logarithm of the forget gate alpha,
                     shape (batch_size, num_heads, seq_len, head_dim_k)
        C: Primary chunk size.
        c: Secondary (sub-chunk) chunk size.

    Returns:
        Output tensor of shape (batch_size, num_heads, seq_len, head_dim_v)
    """

    # Reshape inputs to handle batch and heads seamlessly
    # (batch_size * num_heads, seq_len, dim)
    batch_times_heads, L, d_k = q_input.shape[0] * q_input.shape[1], q_input.shape[2], q_input.shape[3]
    d_v = v_input.shape[3]

    # Flatten batch and head dimensions for easier processing
    Q = q_input.reshape(batch_times_heads, L, d_k)
    K = k_input.reshape(batch_times_heads, L, d_k)
    V = v_input.reshape(batch_times_heads, L, d_v)
    log_a = log_a_input.reshape(batch_times_heads, L, d_k) # log_a is used for B in the paper (log of cumulative product)

    # Initialize the recurrent state S (d_k, d_v) for each item in the batch*heads
    # S_bh corresponds to S in the paper, initialized to zeros
    S_bh = torch.zeros(batch_times_heads, d_k, d_v, device=Q.device, dtype=Q.dtype)

    # Initialize output tensor O
    O_bh = torch.empty_like(V)

    # Precompute B: cumulative sum of log_a (log of cumulative product of decay gates alpha)
    # B_paper[i] = log_a[0] + ... + log_a[i]
    # This corresponds to b_t = product(alpha_j for j=1 to t) in log space in the paper.
    # The paper's B[i] in the pseudo-code is actually log(product_{l=0 to j} alpha_{i*C+l})
    # which is a local cumulative product within a chunk.
    # The pseudo-code's `B` stores `b` for each chunk element, where `b` is cumulative log decay *within* the chunk.
    B_paper_log = torch.empty_like(log_a) # This B_paper_log corresponds to B in Listing 1
    for i in range(0, L // C): # Loop over primary chunks
        chunk_start_idx = i * C
        chunk_end_idx = (i + 1) * C
        # b_log is the cumulative sum of log_a within the current chunk, corresponds to `b` in Listing 1
        b_log_current_chunk = torch.zeros(batch_times_heads, d_k, device=Q.device, dtype=Q.dtype)
        for j in range(0, C): # Loop within a primary chunk
            current_idx_in_seq = chunk_start_idx + j
            # Accumulate log_a for the current element in the chunk
            b_log_current_chunk = b_log_current_chunk + log_a[:, current_idx_in_seq, :]
            # Store this cumulative sum for the current sequence position
            B_paper_log[:, current_idx_in_seq, :] = b_log_current_chunk


    # Main loop over primary chunks
    for i in range(0, L // C):
        # Define range for the current primary chunk
        r = slice(i * C, (i + 1) * C)

        # Extract chunk data for Q, K, V, and the precomputed B_paper_log
        # bq, bk, bv, bb correspond to Q[r], K[r], V[r], B[r] in Listing 1
        q_chunk = Q[:, r, :]    # (bh, C, d_k)
        k_chunk = K[:, r, :]    # (bh, C, d_k)
        v_chunk = V[:, r, :]    # (bh, C, d_v)
        # bb is B_paper_log for the current chunk
        bb_chunk_log = B_paper_log[:, r, :] # (bh, C, d_k)

        # Get the log of the total decay for the current chunk (gamma_i+1 in paper, Section 4.2)
        # b_log_total_chunk corresponds to `b = bb[-1, None]` in Listing 1
        # This is log(product of all alphas in the current chunk)
        b_log_total_chunk = bb_chunk_log[:, -1, :].unsqueeze(1) # (bh, 1, d_k)

        # Inter-chunk computation (Equation for O_inter in Section 4.2)
        # o_inter = (Q_chunk * exp(bb_chunk_log)) @ S_bh
        # q_for_inter corresponds to `q = bq * (bb.exp())` in Listing 1 (error in pseudo-code, should be related to Lambda)
        # Lambda_iC+j = b_iC+j / b_iC. If S_bh carries decay from b_iC, then Q_chunk needs to be scaled by exp(bb_chunk_log)
        # The pseudo code uses `q = bq * (bb.exp())` for `o = q @ S`
        # Let's follow the paper's text more closely for O_inter: O_inter = (Q_chunk * Lambda) @ S_prev
        # Lambda for element j in chunk i+1 is (b_iC+j / b_iC).
        # If S_bh = S_i * product(alphas in chunk i), then Q_chunk should be scaled by exp(bb_chunk_log - b_log_total_chunk_prev) effectively
        # The pseudo-code `o = (bq * bb.exp()) @ S` seems to imply S is S_t-1 from previous chunk,
        # and bb.exp() is the decay from start of current chunk to current token.
        # Q_chunk * exp(bb_chunk_log) corresponds to (Q_i+1 elementwise_prod Lambda_i+1) in Sec 4.2 if S_bh is S_i
        # S_bh here refers to S_[i] from the paper, which is the state *before* processing current chunk i.
        # So, for the j-th element in current chunk Q_j, its interaction with S_bh needs decay from start of chunk to j.
        # This is effectively Q_j * exp(B_paper_log_j)
        # This corresponds to Q_tilde in Alg 3/5: Q_tilde = Q * Lambda
        # Lambda_iC+j = b_iC+j / b_iC. In our local chunk context, if S_bh is S_i (state from end of chunk i-1),
        # then Lambda for element `j` in current chunk is `exp(B_paper_log[:,j,:])` if B_paper_log starts from 0 at chunk start.
        q_scaled_for_inter = q_chunk * torch.exp(bb_chunk_log) # (bh, C, d_k)
        # o_inter corresponds to `o = q @ S` in Listing 1
        o_inter_chunk = torch.bmm(q_scaled_for_inter, S_bh) # (bh, C, d_v)

        # Update hidden state S_bh (Equation for S_i+1 in Section 4.2) [cite: 136]
        # S_new = exp(log_total_decay_this_chunk) * S_old + sum(K_eff_j.T @ V_j for j in chunk)
        # K_eff_j = K_j * Gamma_j where Gamma_j = product(alphas from j+1 to end of chunk)
        # Gamma_j = exp(b_log_total_chunk - bb_chunk_log_j)
        # k_for_update corresponds to `k = bk * ((b-bb).exp())` in Listing 1 [cite: 433]
        k_scaled_for_update = k_chunk * torch.exp(b_log_total_chunk - bb_chunk_log) # (bh, C, d_k)
        # g_decay corresponds to `g = b.exp()` in Listing 1 (total decay factor for the chunk) [cite: 433]
        g_decay_chunk = torch.exp(b_log_total_chunk) # (bh, 1, d_k)

        # S_bh update: S_bh = Diag(g_decay_chunk) @ S_bh + K_scaled_for_update.T @ V_chunk
        # Unsqueeze g_decay_chunk for element-wise multiplication with S_bh columns
        # S_bh has shape (bh, d_k, d_v). g_decay_chunk has shape (bh, 1, d_k)
        # We need to scale rows of S_bh. So (g_decay_chunk.transpose(1,2) * S_bh) if S_bh was (d_k, d_v)
        # Here S_bh is (bh, d_k, d_v). g_decay_chunk is (bh, 1, d_k).
        # So, S_bh_updated_decay_part = g_decay_chunk.permute(0,2,1) * S_bh (element-wise broadcasting) is not quite right.
        # It's Diag(alpha_vec) * S_matrix. So each row i of S is scaled by alpha_vec[i].
        # S_bh[batch, k_dim, v_dim]. g_decay_chunk[batch, 1, k_dim].
        # So S_bh_updated_decay_part should be S_bh * g_decay_chunk.transpose(1,2)
        S_bh = S_bh * g_decay_chunk.permute(0, 2, 1) # (bh, d_k, d_v) element-wise with (bh, d_k, 1) [cite: 433]
        # Sum (K_scaled^T @ V) over the chunk dimension C
        # k_scaled_for_update is (bh, C, d_k), v_chunk is (bh, C, d_v)
        # We need sum_{j in C} (k_scaled_for_update_j^T @ v_chunk_j)
        # This is equivalent to k_scaled_for_update.transpose(1,2) @ v_chunk if we sum the result
        # k_scaled_for_update.transpose(1,2) gives (bh, d_k, C)
        # torch.bmm(k_scaled_for_update.transpose(1,2), v_chunk) gives (bh, d_k, d_v)
        S_bh = S_bh + torch.bmm(k_scaled_for_update.transpose(1, 2), v_chunk) # [cite: 433]

        # Intra-chunk computation (secondary chunking) [cite: 433]
        o_intra_chunk = torch.zeros_like(v_chunk) # Initialize intra-chunk output accumulator

        for j in range(0, C // c): # Loop over sub-chunks (secondary level chunking) [cite: 434]
            # Define range for the current sub-chunk
            t = slice(j * c, (j + 1) * c) # [cite: 434]

            # Extract sub-chunk data
            # q_sub, k_sub, v_sub, b_sub_log correspond to q, k, v, b in Listing 1 inner loop [cite: 434]
            q_sub = q_chunk[:, t, :]        # (bh, c, d_k)
            k_sub = k_chunk[:, t, :]        # (bh, c, d_k)
            v_sub = v_chunk[:, t, :]        # (bh, c, d_v)
            # b_sub_log is B_paper_log for the current sub-chunk
            b_sub_log = bb_chunk_log[:, t, :] # (bh, c, d_k)

            # P_sub_chunk corresponds to `p = (c,c)` in Listing 1 [cite: 434]
            # This is the attention-like matrix P_ij = sum_k Q_ik K_jk exp(logB_ik - logB_jk) (Eq. 4) [cite: 127]
            # This part is computed in full precision as per paper [cite: 142]
            # P_sub_chunk has shape (bh, c, c)
            P_sub_chunk = torch.zeros(batch_times_heads, c, c, device=Q.device, dtype=Q.dtype)

            # Intra-subchunk computation (elements m >= n) (Pink tiles in Fig. 3) [cite: 135]
            for m_idx in range(c): # Corresponds to `m` in `for m in range(c):` [cite: 435]
                for n_idx in range(m_idx + 1): # Corresponds to `n` in `for n in range(m+1):` [cite: 436]
                    # q_m is (bh, 1, d_k), k_n is (bh, 1, d_k)
                    # b_m_log is (bh, 1, d_k), b_n_log is (bh, 1, d_k)
                    q_m = q_sub[:, m_idx, :].unsqueeze(1)
                    k_n = k_sub[:, n_idx, :].unsqueeze(1)
                    b_m_log = b_sub_log[:, m_idx, :].unsqueeze(1)
                    b_n_log = b_sub_log[:, n_idx, :].unsqueeze(1)

                    # exp_term = exp(log_B_m - log_B_n) = exp(b_m_log - b_n_log)
                    # (bh, 1, d_k)
                    exp_term = torch.exp(b_m_log - b_n_log) # [cite: 436]

                    # value = Q_m * K_n * exp_term (element-wise product)
                    # then sum over d_k dimension
                    # P_sub_chunk[:, m_idx, n_idx] = torch.sum(q_m * k_n * exp_term, dim=2) # This is wrong based on pseudo
                    # Pseudo: p[m,n] = torch.sum(q[m]*k[n]*((b[m]-b[n]).exp()))
                    # (q[m]*k[n]) is element-wise, then * exp_term element-wise, then sum over d_k
                    # q_sub[:, m_idx, :] is (bh, d_k)
                    # k_sub[:, n_idx, :] is (bh, d_k)
                    # bb_chunk_log[:, t, :][:, m_idx, :] which is b_sub_log[:, m_idx, :] is (bh, d_k)
                    # So, (q_sub[:, m_idx, :] * k_sub[:, n_idx, :] * torch.exp(b_sub_log[:, m_idx, :] - b_sub_log[:, n_idx, :]))
                    # then sum over d_k (last dimension)
                    # This is for P_mn = (Q_m Hadamard_prod B_m) (K_n / B_n)^T
                    # The paper Eq.4 is P_ij = sum_k Q_ik K_jk exp(logB_ik - logB_jk)
                    # Q_ik is q_sub[:, m_idx, k_idx]
                    # K_jk is k_sub[:, n_idx, k_idx]
                    # exp(logB_ik - logB_jk) is exp(b_sub_log[:, m_idx, k_idx] - b_sub_log[:, n_idx, k_idx])
                    term_to_sum = q_sub[:, m_idx, :] * \
                                  k_sub[:, n_idx, :] * \
                                  torch.exp(b_sub_log[:, m_idx, :] - b_sub_log[:, n_idx, :]) # [cite: 436]
                    P_sub_chunk[:, m_idx, n_idx] = torch.sum(term_to_sum, dim=-1) # Sum over d_k [cite: 436]

            # o_intra_sub = P_sub_chunk @ v_sub
            # o_intra_chunk[:, t, :] gets updated with o_intra_sub
            # `o[t] += p @ v` in Listing 1 [cite: 436]
            o_intra_chunk[:, t, :] = o_intra_chunk[:, t, :] + torch.bmm(P_sub_chunk, v_sub)

            # Inter-subchunk computation (Orange tiles in Fig. 3) [cite: 135]
            # This computes interactions between current sub-chunk q_sub (queries)
            # and previous sub-chunks k_prev_sub, v_prev_sub.
            # `z = b[0, None]` in Listing 1 [cite: 436]
            # This `z` is the log decay at the start of the current sub-chunk `t`.
            z_log_start_of_subchunk = b_sub_log[:, 0, :].unsqueeze(1) # (bh, 1, d_k)

            # `q_inter_sub = q * (b-z).exp()` in Listing 1 [cite: 436]
            # q_sub is (bh, c, d_k), b_sub_log is (bh, c, d_k)
            # This scales queries in current sub-chunk by decay relative to start of sub-chunk.
            q_scaled_for_inter_sub = q_sub * torch.exp(b_sub_log - z_log_start_of_subchunk) # (bh, c, d_k)

            # Loop over previous sub-chunks `u` within the current primary chunk `i`
            # `for u in range(0, j):` [cite: 436]
            for u_idx in range(0, j):
                # Define range for the previous sub-chunk `u`
                y = slice(u_idx * c, (u_idx + 1) * c) # [cite: 436]

                # Extract K, V, B_log from the previous sub-chunk `y`
                k_prev_sub = k_chunk[:, y, :]             # (bh, c, d_k)
                v_prev_sub = v_chunk[:, y, :]             # (bh, c, d_v)
                b_log_prev_sub = bb_chunk_log[:, y, :]    # (bh, c, d_k)

                # Effective K for inter-subchunk: `bk[y]*(z-bb[y]).exp()` in Listing 1 [cite: 436]
                # This term means K_prev_sub scaled by decay from end of k_prev_sub to start of current q_sub.
                # z_log_start_of_subchunk is logB at start of current query sub-chunk
                # b_log_prev_sub is logB within the key's sub-chunk
                # (z_log_start_of_subchunk - b_log_prev_sub) is log(B_q_start / B_k_element)
                k_eff_inter_sub = k_prev_sub * torch.exp(z_log_start_of_subchunk - b_log_prev_sub) # (bh, c, d_k)

                # `p = q @ (bk[y]*(z-bb[y]).exp()).t()` in Listing 1 [cite: 436]
                # This is (q_scaled_for_inter_sub @ k_eff_inter_sub.transpose(1,2))
                # (bh, c, d_k) @ (bh, d_k, c) -> (bh, c, c)
                P_inter_sub_chunk = torch.bmm(q_scaled_for_inter_sub, k_eff_inter_sub.transpose(1, 2))

                # `o[t] += p @ bv[y]` in Listing 1 [cite: 436]
                # Add to the output of the current sub-chunk `t`
                o_intra_chunk[:, t, :] = o_intra_chunk[:, t, :] + torch.bmm(P_inter_sub_chunk, v_prev_sub)

        # Accumulate the total output for the current primary chunk
        # O_bh[:, r, :] = o_inter_chunk + o_intra_chunk
        # The pseudo-code has `O[r] = o` where `o` seems to be `o_inter_chunk` initialized at chunk start.
        # Then `o[t] += p@v` for intra-subchunk and `o[t] += p@bv[y]` for inter-subchunk.
        # This suggests o_inter_chunk should be the base, and o_intra_chunk adds to it.
        # My o_intra_chunk already accumulates both intra-sub and inter-sub parts.
        O_bh[:, r, :] = o_inter_chunk + o_intra_chunk # [cite: 436] (implicit sum from o_inter and intra-chunk updates to `o`)

    # Reshape output back to (batch_size, num_heads, seq_len, d_v)
    return O_bh.reshape(q_input.shape[0], q_input.shape[1], L, d_v)


if __name__ == '__main__':
    B = 2           # batch [4]
    H = 16          # num_heads [4]
    T = 680         # seq_len [300, 512]
    D = 100          # head_dim [32, 64, 100]

    head_first = True
    gate_logit_normalizer = 1   # [1, 0.05, 20]
    expand_ration = 1           # [1, 2]
    dtype = torch.bfloat16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if head_first:
        q = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()                  # batch, num_heads, seq_len, head_dim
        k = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((B, H, T, D), dtype=dtype, device=device).requires_grad_()
        g = F.logsigmoid(torch.randn((B, H, T, D), dtype=dtype, device=device)).requires_grad_()    # batch, num_heads, seq_len, head_dim
    else:
        q = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
        k = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
        v = torch.randn((B, T, H, D), dtype=dtype, device=device).requires_grad_()
        # g = F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device=device)).requires_grad_()
        g = (F.logsigmoid(torch.randn((B, T, H, D), dtype=dtype, device=device)) / gate_logit_normalizer).requires_grad_()
    h0 = torch.randn(B, H, D, D, device=device).requires_grad_()                                # batch, num_heads, head_dim, head_dim

    out = naive_chunk_gla(
        q_input=q, k_input=k, v_input=v,
        log_a_input=g,
        C=64, c=4
    )

    print(f'{out.shape=}')





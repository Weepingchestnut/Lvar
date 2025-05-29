import os
# os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

from typing import Optional, Tuple
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from models.kernels.gla.chunk import chunk_gla
from models.kernels.gla.fused_recurrent import fused_recurrent_gla
from models.kernels.gla.naive import naive_recurrent_gla
from models.kernels.utils import assert_close, device


if __name__ == '__main__':
    B = 4           # batch [4]
    H = 16          # num_heads [4]
    T = 680         # seq_len [300, 512]
    D = 64          # head_dim [32, 64, 100]

    head_first = False
    gate_logit_normalizer = 1   # [1, 0.05, 20]
    expand_ration = 1           # [1, 2]
    dtype = torch.bfloat16

    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

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
    
    do = torch.randn_like(v)
    dht = torch.zeros((B, H, D, D), dtype=dtype, device=device)

    # ----------------------
    # *test_fused_recurrent
    # ----------------------
    ref, ref_ht = naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True)
    print(f'{ref.shape=}')
    print(f'{ref_ht.shape=}')
    ((ref * do).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None
    print("ref backward finish!")

    tri, tri_ht = fused_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=g,
        initial_state=h0,
        output_final_state=True
    )
    ((tri * do).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None

    # ------------
    # *test chunk
    # ------------
    ref, ref_ht = fused_recurrent_gla(
        q,
        k,
        v,
        g,
        initial_state=h0,
        output_final_state=True,
    )
    ref, _ = fused_recurrent_gla(
        q,
        k,
        v,
        g,
        initial_state=h0,
        output_final_state=False,
    )
    (ref * do).sum().backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None
    ref_dh0, h0.grad = h0.grad.clone(), None

    tri, tri_ht = chunk_gla(
        q,
        k,
        v,
        g,
        initial_state=h0,
        output_final_state=True,
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    tri_dh0, h0.grad = h0.grad.clone(), None
    print("tri backward finish!")

    # ------ Compare ------
    assert_close('  o', ref, tri, 0.005)
    assert_close(' ht', ref_ht, tri_ht, 0.005)
    assert_close(' dq', ref_dq, tri_dq, 0.005)
    assert_close(' dk', ref_dk, tri_dk, 0.005)
    assert_close(' dv', ref_dv, tri_dv, 0.005)
    assert_close(' dg', ref_dg, tri_dg, 0.005)
    assert_close('dh0', ref_dh0, tri_dh0, 0.005)

    print("complete!")

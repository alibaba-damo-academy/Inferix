import math
import time
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

import torch.distributed
from yunchang import LongContextAttention
from yunchang.ring.utils import RingComm, update_out_and_lse
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")

from yunchang.comm.all_to_all import SeqAllToAll4D
from yunchang.globals import PROCESS_GROUP

from xfuser.logger import init_logger

from xfuser.core.distributed import (
    get_ring_parallel_world_size,
    )

from inferix.models.attention.backends import collect_supported_attn

torch.inference_mode(True)


@torch.jit.script
def update_out_and_lse_pass_q(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    block_out = block_out.to(torch.float32)
    # block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    # For additional context and discussion, please refer to:
    # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)

    return out, lse




class CoreAttention(torch.nn.Module):

    def __init__(self, 
                 scatter_idx: int = 2, 
                 gather_idx: int = 1, 
                 ring_impl_type: str = "basic", 
                 use_pack_qkv: bool = False, 
                 attn_type: AttnType = AttnType.FA,
                 q_descale=None,
                 k_descale=None,
                 v_descale=None,
                 strategy: str = "auto",
                 ) -> None:
        """
        Arguments:
            scatter_idx: int = 1, the scatter dimension index for Ulysses All2All
            gather_idx: int = 0, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
            attn_type: AttnType = AttnType.FA, the attention type supported inside long context attention, including "TORCH", "FA", "FA3", "SAGE_FP16", "SAGE_FP8"
            attn_processor: nn.Module = None, the attention processor can be passed in to replace the attention processor if attn_type is do not support it.
            strategy: str = "auto", the attention strategy to use. Options: "auto", "pass_q", "pass_kv", "ulysses"
        """
        super(CoreAttention, self).__init__()
        self.ulysses_pg = PROCESS_GROUP.ULYSSES_PG
        self.ring_pg = PROCESS_GROUP.RING_PG

        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        self.strategy = strategy
        self.attn_type = attn_type

        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_pack_qkv = use_pack_qkv

        self.supported_attn = collect_supported_attn()


        # here we try to use four different attention backends: 1. flash_attention v3, 2. magi_attn, 3. flash_attn, 4. flash_infer.

    def _select_strategy(self, query: Tensor, key: Tensor, value: Tensor, 
                        k_cache: Tensor = None, v_cache: Tensor = None) -> str:
        """
        Automatically select the best attention strategy based on input characteristics.
        
        Args:
            query: Query tensor
            key: Key tensor  
            value: Value tensor
            k_cache: Key cache tensor (for Ulysses)
            v_cache: Value cache tensor (for Ulysses)
            
        Returns:
            Selected strategy: "pass_q", "pass_kv", or "ulysses"
        """
        if self.strategy != "auto":
            return self.strategy
            
        # If KV cache is provided, use Ulysses strategy
        if k_cache is not None and v_cache is not None:
            return "ulysses"
            
        # Get tensor dimensions
        seq_len = query.shape[1]
        num_heads = query.shape[2]
        head_dim = query.shape[3]
        
        # Get ring world size
        ring_world_size = torch.distributed.get_world_size(self.ring_pg)
        
        # Simple heuristic: for longer sequences, pass_q tends to be more efficient
        # For shorter sequences or when we have many heads, pass_kv might be better
        if seq_len > 2048 or (seq_len > 1024 and num_heads >= 16):
            return "pass_q"
        else:
            return "pass_kv"

    @torch.compiler.disable
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor|List[Tensor] = None,
        v_cache: Tensor|List[Tensor] = None,
        k_cache_offset: int|List[int] = 0,
        v_cache_offset: int|List[int] = 0,
        *,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        custom_mask: Tensor|List[Tensor] = None,
        q_ranges=None,
        k_ranges=None,
        attn_type_map=None,
        attn_backend=None,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer. NOTE: the batch size must be 1.
            key (Tensor): key input to the layer. NOTE: the batch size must be 1
            value (Tensor): value input to the layer. NOTE: the batch size must be 1
            k_cache (Tensor): key cache for Ulysses strategy, it is the caller's responsibility to provide the k_cache with sufficient length to write the new key.
            v_cache (Tensor): value cache for Ulysses strategy, it is the caller's responsibility to provide the v_cache with sufficient length to write the new value.
            k_cache_offset (int): the offset of the k_cache, which is used to write the new key.
            v_cache_offset (int): the offset of the v_cache, which is used to write the new value.
            *args: the args same as flash_attn_interface
            custom_mask: Tensor = None, a custom mask for the attention, which is applied in PipeFusion.

        Returns:
            * output (Tensor): context output
        """
        # Select the best strategy
        selected_strategy = self._select_strategy(query, key, value, k_cache, v_cache)


        # 3 X (seq_len/N, head_cnt, head_size) -> 3 X (seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:
            # (3, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).contiguous()
            # (3, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(
                self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx
            )
            query_layer, key_layer, value_layer = qkv[0], qkv[1], qkv[2]
        else:
            query_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, query, self.scatter_idx, self.gather_idx
            )
            key_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, key, self.scatter_idx, self.gather_idx
            )
            value_layer = SeqAllToAll4D.apply(
                self.ulysses_pg, value, self.scatter_idx, self.gather_idx
            )

        # Handle KV cache for Ulysses strategy
        if k_cache is not None and v_cache is not None:
            if isinstance(k_cache, Tensor):
                k_cache[0, k_cache_offset:k_cache_offset+key_layer.shape[1], ...] = key_layer
                v_cache[0, v_cache_offset:v_cache_offset+value_layer.shape[1], ...] = value_layer
                key_layer = k_cache[:, :k_cache_offset+key_layer.shape[1], ...]
                value_layer = v_cache[:, :v_cache_offset+value_layer.shape[1], ...]
            elif isinstance(k_cache, List):
                for i in range(len(k_cache)):
                    k_cache[i][0, k_cache_offset[i]:k_cache_offset[i]+key_layer.shape[1], ...] = key_layer
                    v_cache[i][0, v_cache_offset[i]:v_cache_offset[i]+value_layer.shape[1], ...] = value_layer


        # the selected strategy: ["ulysses", "pass-q", "pass-kv", "ulysses-pass-q", "ulysses-pass-kv"]
        # Call the appropriate ring attention function based on strategy
        pass_q = selected_strategy in ["pass-q", "ulysses-pass-q"]
        if isinstance(k_cache, List):
            for i in range(len(k_cache)):
                key_layer = k_cache[i][:, :k_cache_offset[i]+key_layer.shape[1], ...]
                value_layer = v_cache[i][:, :v_cache_offset[i]+value_layer.shape[1], ...]
                query_layer = query_layer[i].unsqueeze(0)
                mask = custom_mask[i] if custom_mask is not None else None
                out = self._attention_forward(
                    query_layer,
                    key_layer,
                    value_layer,
                    k_cache_offset=k_cache_offset[i],
                    v_cache_offset=v_cache_offset[i],
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    q_descale=self.q_descale,
                    k_descale=self.k_descale,
                    v_descale=self.v_descale,
                    pass_q=pass_q,
                    custom_mask=mask,
                    q_ranges=q_ranges,
                    k_ranges=k_ranges,
                    attn_type_map=attn_type_map,
                    attn_backend=attn_backend,
                )
        else:
            out = self._attention_forward(
                query_layer,
                key_layer,
                value_layer,
                k_cache_offset=k_cache_offset,
                v_cache_offset=v_cache_offset,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                q_descale=self.q_descale,
                k_descale=self.k_descale,
                v_descale=self.v_descale,
                pass_q=pass_q,
                custom_mask=custom_mask,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                attn_backend=attn_backend,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(
            self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx
        )

        # out e.g., [s/p::h]
        return output


    def _attention_forward(
        self, 
        query: Tensor, 
        key: Tensor, 
        value: Tensor, 
        k_cache_offset=0,
        v_cache_offset=0,
        dropout_p=0.0, 
        softmax_scale=None, 
        causal=False, 
        window_size=(-1, -1), 
        alibi_slopes=None, 
        deterministic=False, 
        return_attn_probs=False, 
        custom_mask: Tensor|List[Tensor] = None, 
        pass_q=False,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        q_ranges=None,
        k_ranges=None,
        attn_type_map=None,
        attn_backend=None,
        ) -> Tensor:
        """
        Arguments:
            query: Tensor, the query tensor
            key: Tensor, the key tensor
            value: Tensor, the value tensor
            dropout_p: float, the dropout probability
            softmax_scale: float, the softmax scale
            causal: bool, whether to use causal attention
        """

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(query.size(-1))

        assert q_descale is None # or flash attention 3 is available.
        assert alibi_slopes is None
        if pass_q:
            out, softmax_lse = self.ring_attention_forward_pass_q(
                self.ring_pg,
                query,
                key,
                value,
                softmax_scale=softmax_scale,
                k_cache_offset=k_cache_offset,
                v_cache_offset=v_cache_offset,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                custom_mask=custom_mask,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                attn_backend=attn_backend,
            )
        else:
            out, softmax_lse = self.ring_attention_forward_pass_kv(
                self.ring_pg,
                query,
                key,
                value,
                softmax_scale=softmax_scale,
                k_cache_offset=k_cache_offset,
                v_cache_offset=v_cache_offset,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                custom_mask=custom_mask,
                q_ranges=q_ranges,
                k_ranges=k_ranges,
                attn_type_map=attn_type_map,
                attn_backend=attn_backend,
            )

        return out if not return_attn_probs else (out, softmax_lse)


    # @nvtx_range("ring_attention_forward_pass_q")
    def ring_attention_forward_pass_q(
        self,
        process_group,
        q: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        softmax_scale,
        k_cache_offset=0,
        v_cache_offset=0,
        dropout_p=0,
        deterministic=False,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        custom_mask=None,
        q_ranges=None,
        k_ranges=None,
        attn_type_map=None,
        attn_backend=None,
    ):
        """
        Arguments:
            query: Tensor, the query tensor
            key: Tensor, the key tensor
            value: Tensor, the value tensor
            dropout_p: float, the dropout probability
            softmax_scale: float, the softmax scale
            causal: bool, whether to use causal attention
            window_size: tuple, the window size
            alibi_slopes: Tensor, the alibi slopes
            q_descale: float, the query descale
            k_descale: float, the key descale
            v_descale: float, the value descale
            custom_mask: Tensor, the custom mask
        """
        comm = RingComm(process_group)

        # For pass_q, each node processes a different slice of the query sequence in each iteration
        # We need to stack (concatenate) the block_outs and block_lses along the sequence dimension
        block_outs = [None for _ in range(comm.world_size)]
        block_lses = [None for _ in range(comm.world_size)]

        # TODO(lhf): need to determine the correct attention type to use.
        if attn_backend is not None:
            if attn_backend not in self.supported_attn:
                available_backends = list(self.supported_attn.keys())
                raise ValueError(f"Specified attention backend '{attn_backend}' is not available. Available backends: {available_backends}")
            attn = self.supported_attn[attn_backend]
        elif custom_mask is not None:
            # select between FlashInfer and FlexAttention
            attn = self.supported_attn["FlashInfer"] if "FlashInfer" in self.supported_attn else self.supported_attn["FlexAttention"]
            attn_backend = "FlashInfer" if "FlashInfer" in self.supported_attn else "FlexAttention"
        elif q_ranges is not None and k_ranges is not None:
            attn = self.supported_attn["MagiAttention"]
            attn_backend = "MagiAttention"
        else:
            raise RuntimeError("No attention backend specified and no fallback conditions met. Please configure attn_backend in ParallelConfig.")
        next_q = None

        with torch.cuda.nvtx.range("ring_attn_forward_pass_q_compute"):

            # If the attention is causal, we need to process two parts: full attention on the past kv, and causal attention on the current kv.
            for step in range(comm.world_size):
                if step + 1 != comm.world_size:
                    next_q: torch.Tensor = comm.send_recv(q)
                    comm.commit()
                if not causal or comm.rank <= (comm.rank - step + comm.world_size) % comm.world_size:
                    input_key, input_value = key, value
                elif comm.rank > (comm.rank - step + comm.world_size) % comm.world_size:
                    # for these steps, we need to process all the previous keys and values.
                    if k_cache_offset > 0 and v_cache_offset > 0:
                        input_key = key[:, :k_cache_offset]
                        input_value = value[:, :v_cache_offset]
                if not causal or (comm.rank <= (comm.rank - step + comm.world_size) % comm.world_size) or (k_cache_offset > 0 and v_cache_offset > 0):
                    if attn_backend == "FlashAttnV3": 
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale
                        )
                    elif attn_backend == "FlashAttn":
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                            return_softmax=True and dropout_p > 0,
                        )
                    elif attn_backend == "FlashInfer":
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                            custom_mask=custom_mask,
                        )
                    elif attn_backend == "MagiAttention":
                        if causal and step == 0:
                            # use the provided mask
                            input_attn_type_map = attn_type_map
                        else:
                            input_attn_type_map = torch.tensor([0], device=torch.device("cuda"), dtype=torch.int32)
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            max_seqlen_q=q.shape[1],
                            max_seqlen_k=input_key.shape[1],
                            attn_type_map=input_attn_type_map,
                            softmax_scale=softmax_scale,
                            softcap=0.0,
                            deterministic=deterministic,
                        )
                    else:
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                            return_softmax=True and dropout_p > 0,
                        )

                    # print(f"Q: {(comm.rank - step + comm.world_size) % comm.world_size}, KV: {comm.rank}, block_out: {block_out}, block_lse: {block_lse}", flush=True)
                    # Stack results for this node
                    block_outs[(comm.rank - step + comm.world_size) % comm.world_size] = block_out.to(torch.float32)
                    block_lses[(comm.rank - step + comm.world_size) % comm.world_size] = block_lse.transpose(-2, -1).unsqueeze(dim=-1).contiguous()

                    # print(f"rank: {comm.rank}, step: {step}, q: {query}, k: {k}, v: {v}, block_out: {block_out}, block_lse: {block_lse}", flush=True)

                if step + 1 != comm.world_size:
                    comm.wait()
                    q = next_q

        # Now, out: [bs, n_heads, seq_len_total, d], lse: [bs, n_heads, seq_len_total, ...]
        # Each node wants only its original local query slice (seq_len_local = q.shape[2])
        # here, we also do the reordering. i in out and lse is the rank of the node that we want to send to.
        with torch.cuda.nvtx.range("ring_attn_forward_pass_q_gather"):
            out = [torch.zeros_like(block) if block is not None else torch.zeros_like(block_outs[comm.rank]) for block in block_outs]
            lse = [torch.zeros_like(block) if block is not None else torch.zeros_like(block_lses[comm.rank]) for block in block_lses]


            for i in range(comm.world_size-1):
                # gather the block_outs and block_lses
                block_to_send = (comm.world_size+comm.rank-i-1)%comm.world_size
                block_will_receive = (comm.world_size+comm.rank-i-2)%comm.world_size
                # print(f"rank: {comm.rank}, i: {i}, block_to_send: {block_to_send}, block_will_receive: {block_will_receive}", flush=True)
                # check if they are contiguous
                out[block_will_receive] = comm.send_recv(block_outs[block_to_send], out[block_will_receive])
                lse[block_will_receive] = comm.send_recv(block_lses[block_to_send], lse[block_will_receive])
                # TODO(lhf): For now, we assume all tensors are the same shape.
                comm.commit()
                comm.wait()
                # TODO(lhf): do the log-sum-exp trick here.
                # print(f"rank: {comm.rank}, step: {i}, merging block: {out[block_will_receive]}", flush=True)
                # print(f"rank: {comm.rank}, step: {i}, sending block: {block_outs[block_to_send]}, receiving block: {out[block_will_receive]}", flush=True)
                block_outs[block_will_receive], block_lses[block_will_receive] = update_out_and_lse_pass_q(block_outs[block_will_receive], block_lses[block_will_receive], out[block_will_receive], lse[block_will_receive])

            # select the correct slice of out and lse and return.
        # print(f"final output: rank: {comm.rank}, out: {block_outs[comm.rank]}, lse: {block_lses[comm.rank]}", flush=True)

        return block_outs[comm.rank].to(q.dtype), block_lses[comm.rank].to(q.dtype)


    def ring_attention_forward_pass_kv(
        self,
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale,
        k_cache_offset=0,
        v_cache_offset=0,
        dropout_p=0,
        deterministic=False,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        custom_mask=None,
        q_ranges=None,
        k_ranges=None,
        attn_type_map=None,
        attn_backend=None,
    ):
        comm = RingComm(process_group)

        out = None
        lse = None

        next_k, next_v = None, None

        if attn_backend is not None:
            if attn_backend not in self.supported_attn:
                available_backends = list(self.supported_attn.keys())
                raise ValueError(f"Specified attention backend '{attn_backend}' is not available. Available backends: {available_backends}")
            attn = self.supported_attn[attn_backend]
        elif custom_mask is not None:
            # select between FlashInfer and FlexAttention
            attn = self.supported_attn["FlashInfer"] if "FlashInfer" in self.supported_attn else self.supported_attn["FlexAttention"]
            attn_backend = "FlashInfer" if "FlashInfer" in self.supported_attn else "FlexAttention"
        elif q_ranges is not None and k_ranges is not None:
            attn = self.supported_attn["MagiAttention"]
            attn_backend = "MagiAttention"
        else:
            raise RuntimeError("No attention backend specified and no fallback conditions met. Please configure attn_backend in ParallelConfig.")

        with torch.cuda.nvtx.range("ring_attn_forward_pass_kv_compute"):
            for step in range(comm.world_size):
                if step + 1 != comm.world_size:
                    next_k: torch.Tensor = comm.send_recv(k)
                    next_v: torch.Tensor = comm.send_recv(v)
                    comm.commit()


                key, value = k, v

                if not causal or step <= comm.rank:
                    input_key, input_value = key, value
                elif not causal or step > comm.rank:
                    # for these steps, we need to process all the previous keys and values.
                    if k_cache_offset > 0 and v_cache_offset > 0:
                        input_key = key[:, :k_cache_offset]
                        input_value = value[:, :v_cache_offset]

                if not causal or step <= comm.rank or (k_cache_offset > 0 and v_cache_offset > 0):
                    # if comm.rank == 0:
                        # print(f"rank: {comm.rank}, step: {step}, input_key: {input_key}, input_value: {input_value}", flush=True)
                    if attn_backend == "FlashAttnV3": 
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            q_descale=q_descale,
                            k_descale=k_descale,
                            v_descale=v_descale
                        )
                    elif attn_backend == "FlashAttn":
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                        )
                    elif attn_backend == "FlashInfer":
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                            custom_mask=custom_mask,
                        )
                    elif attn_backend == "MagiAttention":
                        if causal and step == 0:
                            # use the provided mask
                            input_attn_type_map = attn_type_map
                        else:
                            input_attn_type_map = torch.tensor([0], device=torch.device("cuda"), dtype=torch.int32)
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            q_ranges=q_ranges,
                            k_ranges=k_ranges,
                            max_seqlen_q=q.shape[1],
                            max_seqlen_k=key.shape[1],
                            attn_type_map=input_attn_type_map,
                            softmax_scale=softmax_scale,
                            softcap=0.0,
                            deterministic=deterministic,
                        )
                    else:
                        block_out, block_lse = attn(
                            q,
                            input_key,
                            input_value,
                            dropout_p=dropout_p,
                            softmax_scale=softmax_scale,
                            causal=causal and step == 0,
                            window_size=window_size,
                            softcap=0.0,
                            alibi_slopes=alibi_slopes,
                        )
                    # print(f"Q: {comm.rank}, KV: {(comm.rank - step + comm.world_size) % comm.world_size}, block_out: {block_out}, block_lse: {block_lse}", flush=True)
                    out, lse = update_out_and_lse(out, lse, block_out, block_lse)

                if step + 1 != comm.world_size:
                    comm.wait()
                    k = next_k
                    v = next_v

        with torch.cuda.nvtx.range("ring_attn_forward_pass_kv_gather"):
            out = out.to(q.dtype)
            lse = lse.squeeze(dim=-1).transpose(1, 2)

        return out, lse

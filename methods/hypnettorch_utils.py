from math import ceil
from typing import List, Union, Tuple, Callable

import numpy as np
import torch
from hypnettorch.hnets import StructuredHMLP, HyperNetInterface, HMLP, ChunkedHMLP

hn_types = ["hmlp", "chmlp", "shmlp"]


def build_hypnettorch(
        target_shapes: List[torch.Size],
        uncond_in_size: int,
        params,
) -> Union[HyperNetInterface, torch.nn.Module]:
    layers = [params.hn_hidden_size] * params.hn_neck_len

    hypernet_type = params.hn_lib_type

    if hypernet_type == "hmlp":
        return HMLP(
            target_shapes=target_shapes,
            uncond_in_size=uncond_in_size,
            cond_in_size=0,
            layers=layers,
            no_cond_weights=True
        )

    elif hypernet_type == "chmlp":
        return ChunkedHMLP(
            target_shapes=target_shapes,
            chunk_size=params.hn_lib_chunk_size,
            chunk_emb_size=params.hn_lib_chunk_emb_size,
            layers=layers,
            uncond_in_size=uncond_in_size,
            no_cond_weights=True,
            cond_in_size=0,

        )

    elif hypernet_type == "shmlp":
        if params.hn_lib_chunk_size == -1:
            print("Chunk size = -1, there will be no chunking")
            chunk_shapes = [[ts] for ts in target_shapes]
            num_per_chunk = [1] * len(target_shapes)
            assembly_fn = lambda tlists: [tl[0] for tl in tlists]
        else:
            chunk_shapes, num_per_chunk, assembly_fn = shmlp_args(target_shapes,
                                                                  max_chunk_size=params.hn_lib_chunk_size)
        # chunk_shapes= [[[3]], [[10, 5], [5]]]
        # num_per_chunk = [2, 1]
        print("Chunk shapes", chunk_shapes)
        print("num per chunk", num_per_chunk)
        return StructuredHMLP(
            target_shapes=target_shapes,
            chunk_shapes=chunk_shapes,
            num_per_chunk=num_per_chunk,
            uncond_in_size=uncond_in_size,
            cond_in_size=0,
            hmlp_kwargs=[
                            {
                                "layers": layers
                            }
                        ] * len(chunk_shapes),  #len(target_shapes),
            chunk_emb_sizes=params.hn_lib_chunk_emb_size,
            assembly_fct=assembly_fn,
            no_cond_weights=False
        )
    else:
        raise TypeError(f"Allowed types: {hn_types}")


def shmlp_args(target_shapes: List[Union[torch.Size, List[int]]], max_chunk_size: int) -> Tuple[
    List[List[List[int]]],
    List[int],
    Callable[
        [
            List[List[torch.Tensor]]
        ],
        List[torch.Tensor]
    ]
]:
    chunk_shapes = []
    num_per_chunk = []

    for ts in target_shapes:
        ts_size = np.prod(ts)
        num_chunks = ceil(ts_size / max_chunk_size)
        chunk_size = ts[:]

        num_chunks_acc = 1
        for i, s in enumerate(ts):
            if num_chunks_acc >= num_chunks:
                break

            needed_splits = ceil(num_chunks / num_chunks_acc)
            if needed_splits > s:
                chunk_size[i] = 1
                num_chunks_acc *= s
            else:
                cs = s // needed_splits
                num_chunks_acc *= ceil(s / cs)
                chunk_size[i] = cs

        num_per_chunk.append(num_chunks_acc)
        chunk_shapes.append([chunk_size])

    for i, (ts, cs, ns) in enumerate(zip(target_shapes, chunk_shapes, num_per_chunk)):
        tp = np.prod(ts)
        cp = np.prod(cs)

        print("Chunking param", i)
        print(f"target: {ts} -> {tp} params")
        print(f"chunk shape: {cs} -> {cp} params, repeated {ns} times,  {ns * cp} after assembling")
        print(f"{(ns * cp) - tp} redundant parameters")
        print("-----")

    def assembly_fn(chunks: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        gathered_chunks = []
        i = 0
        for nc in num_per_chunk:
            gathered_chunks.append([c[0] for c in chunks[i:(i+nc)]])
            i+=nc
        assert len(gathered_chunks) == len(target_shapes), ([[c.shape for c in cs] for cs in chunks], target_shapes)

        targets = []
        for ts, t_chunks in zip(target_shapes, gathered_chunks):
            ts_size = np.prod(ts)
            flat = torch.stack(t_chunks).reshape(-1)
            assert len(flat) >= ts_size

            targets.append(flat[:ts_size].reshape(ts))

        return targets

    return chunk_shapes, num_per_chunk, assembly_fn


# shmlp_args(
#     [
#         [128, 128],
#         [1024],
#         [128, 30],
#     ],
#
#     1024
# )

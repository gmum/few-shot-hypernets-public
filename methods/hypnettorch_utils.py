from pprint import pprint
from typing import List, Dict, Type, Union

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
        assert params.hn_lib_chunk_size==-1, "only -1 supported at the moment!"
        return StructuredHMLP(
            target_shapes=target_shapes,
            chunk_shapes=[[ts] for ts in target_shapes],
            num_per_chunk=[1] * len(target_shapes),
            uncond_in_size=uncond_in_size,
            cond_in_size=0,
            hmlp_kwargs=[
                {
                    "layers": layers
                }
            ] * len(target_shapes),
            chunk_emb_sizes=params.hn_lib_chunk_emb_size,
            assembly_fct=lambda tlists: [tl[0] for tl in tlists],
            no_cond_weights=False
        )
    else:
        raise TypeError(f"Allowed types: {hn_types}")




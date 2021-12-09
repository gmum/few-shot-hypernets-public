from pprint import pprint
from typing import List, Dict, Type, Union

import numpy as np
import torch
from hypnettorch.hnets import StructuredHMLP, HyperNetInterface, HMLP, ChunkedHMLP

hn_types = ["hmlp", "chmlp"]

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

    else:
        raise TypeError(f"Allowed types: {hn_types}")



# def build_shmlp(
#         target_shapes: List[torch.Size],
#         uncond_in_size: int,
#         cond_in_size: int,
#         layers: List[int],
#         no_cond_weights: bool
# ) -> StructuredHMLP:
#
#     max_chunk_size = 30
#     print(target_shapes)
#     print([np.prod(ts) for ts in target_shapes])
#
#     chunk_shapes = []
#     num_per_chunk = []
#
#
#
#
#     assert False
#
#
#     shmlp = StructuredHMLP(
#         target_shapes=target_shapes,
#         chunk_shapes=[[ts] for ts in target_shapes],
#         num_per_chunk=[2] * len(target_shapes),
#         uncond_in_size=uncond_in_size,
#         cond_in_size=cond_in_size,
#         hmlp_kwargs=[
#             {
#                 "layers": layers
#             }
#         ] * len(target_shapes),
#         chunk_emb_sizes=15,
#         assembly_fct=lambda tlists: [tl[0] for tl in tlists],
#         no_cond_weights=no_cond_weights
#     )
#
#     for i, hn in enumerate(shmlp._hnets):
#         print("HN", i)
#         pprint(
#             {
#                 pn: pt.shape
#                 for (pn, pt) in hn.named_parameters()
#             }
#         )
#
#     return shmlp
#
#
# hn_lib_types: Dict[str, Type[HyperNetInterface]] = {
#     "hmlp": ,
#     "shmlp": build_shmlp,
# }
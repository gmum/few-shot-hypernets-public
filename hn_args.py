from argparse import ArgumentParser


def add_hn_args_to_parser(parser: ArgumentParser) -> ArgumentParser:

    hn_args = parser.add_argument_group("Hypernet-related arguments")
    hn_args.add_argument("--hn_hidden_size", type=int, default=256, help="HN hidden size")
    hn_args.add_argument("--hn_tn_hidden_size", type=int, default=128, help="TN hidden size")
    hn_args.add_argument("--hn_taskset_size", type=int, default=1, help="Taskset size")
    hn_args.add_argument("--hn_neck_len", type=int, default=0, help="Number of layers in the neck of the hypernet")
    hn_args.add_argument("--hn_head_len", type=int, default=2, help="Number of layers in the heads of the hypernet, must be >= 1")
    hn_args.add_argument("--hn_taskset_repeats", type=str, default="10:10-20:5-30:2", help="A sequence of <epoch:taskset_repeats_until_the_epoch>")
    hn_args.add_argument("--hn_taskset_print_every", type=int, default=20, help="It's a utility")
    hn_args.add_argument("--hn_detach_ft_in_hn", type=int, default=10000, help="Detach FE output before hypernetwork in training *after* this epoch")
    hn_args.add_argument("--hn_detach_ft_in_tn", type=int, default=10000, help="Detach FE output before target network in training *after* this epoch")
    hn_args.add_argument("--hn_tn_depth", type=int, default=2, help="Depth of target network")
    hn_args.add_argument("--hn_transformer_layers_no", type=int, default=1, help="Number of layers in transformer")
    hn_args.add_argument("--hn_transformer_heads_no", type=int, default=1, help="Number of attention heads in transformer")
    hn_args.add_argument("--hn_transformer_feedforward_dim", type=int, default=512, help="Transformer's feedforward dimensionality")
    hn_args.add_argument("--hn_attention_embedding", action='store_true', help="Utilize attention-based embedding")
    hn_args.add_argument("--hn_kernel_layers_no", type=int, default=2, help="Depth of a kernel network")
    hn_args.add_argument("--hn_kernel_hidden_dim", type=int, default=128, help="Hidden dimension of a kernel network")
    hn_args.add_argument("--kernel_transformer_layers_no", type=int, default=1, help="Number of layers in kernel's transformer")
    hn_args.add_argument("--kernel_transformer_heads_no", type=int, default=1, help="Number of attention heads in kernel's transformer")
    hn_args.add_argument("--kernel_transformer_feedforward_dim", type=int, default=512, help="Kernel transformer's feedforward dimensionality")
    hn_args.add_argument("--hn_kernel_invariance", action='store_true', help="Should the HyperNet's kernel be sequence invariant")
    hn_args.add_argument("--hn_kernel_invariance_type", default='attention',  choices=['attention', 'convolution'], help="The type of invariance operation for the kernel's output")
    hn_args.add_argument("--hn_kernel_convolution_output_dim", type=int, default=625, help="Kernel convolution's output dim")
    hn_args.add_argument("--use_scalar_product", action='store_true', help="Use scalar product instead of a more specific kernel")
    hn_args.add_argument("--hn_kernel_invariance_pooling", default='mean',  choices=['average', 'mean', 'min', 'max'], help="The type of invariance operation for the kernel's output")
    hn_args.add_argument("--hn_kernel_invariance_without_pooling", action='store_true',  help="Remove pooling after invariant operation")
    hn_args.add_argument("--use_support_embeddings", action='store_true', help="Concatenate support embeddings with kernel features")
    hn_args.add_argument("--no_self_relations", action='store_true', help="Multiply matrix K to remove self relations (i.e., kernel(x_i, x_i))")
    hn_args.add_argument("--hn_queries_relations", action='store_true', help="Queries relations kernel")
    hn_args.add_argument("--hn_queries_convolution_output_dim", type=int, default=625, help="Kernel convolution's output dim")

    return parser
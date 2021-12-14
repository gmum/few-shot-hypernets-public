from argparse import ArgumentParser
from methods.hypnettorch_utils import hn_types


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
    hn_args.add_argument("--hn_ln", action="store_true", default=False, help="Add BatchNorm to hypernet")
    hn_args.add_argument("--hn_dropout", type=float, default=0, help="Dropout probability in hypernet")
    hn_args.add_argument("--hn_tn_activation", type=str, default="relu", choices=["relu", "sin", "tanh"], help="Activation in the target network")
    hn_args.add_argument("--hn_lib_type", type=str, default="hmlp", choices =hn_types, help="Hypernet type from hypnettorch package")
    hn_args.add_argument("--hn_lib_chunk_size", type=int, default=1024, help="Hypnettorch chunk size")
    hn_args.add_argument("--hn_lib_chunk_emb_size", type=int, default=8, help="Hypnettorch chunk embedding size")
    hn_args.add_argument("--hn_val_epochs", type=int, default=0, help="Epochs for finetuning on support set during validation")
    hn_args.add_argument("--hn_val_lr", type=float, default=1e-4, help="LR for finetuning on support set during validation")
    hn_args.add_argument("--hn_val_optim", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer for finetuning on support set during validation")
    return parser
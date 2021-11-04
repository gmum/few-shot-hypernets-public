from argparse import ArgumentParser


def add_hn_args_to_parser(parser: ArgumentParser) -> ArgumentParser:

    hn_args = parser.add_argument_group("Hypernet-related arguments")
    hn_args.add_argument("--hn_hidden_size", type=int, default=256, help="HN hidden size")
    hn_args.add_argument("--hn_tn_hidden_size", type=int, default=128, help="TN hidden size")
    hn_args.add_argument("--hn_taskset_size", type=int, default=1, help="Taskset size")
    hn_args.add_argument("--hn_detach_support", action="store_true", default=False, help="Detach support after FE in classification")
    hn_args.add_argument("--hn_detach_query", action="store_true", default=False, help="Detach query after FE in classification")
    hn_args.add_argument("--hn_neck_len", type=int, default=0, help="Number of layers in the neck of the hypernet")
    hn_args.add_argument("--hn_head_len", type=int, default=2, help="Number of layers in the heads of the hypernet, must be >= 1")
    hn_args.add_argument("--hn_taskset_repeats", type=str, default="10:10-20:5-30:2", help="A sequence of <epoch:taskset_repeats_until_the_epoch>")
    hn_args.add_argument("--hn_taskset_print_every", type=int, default=20, help="It's a utility")

    return parser
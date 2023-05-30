import argparse
from ast import parse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    # _______________
    # General Setting
    timestr = time.strftime("%m%d-%H%M")

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)  # 50
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--dataset", type=str,
                        default="ogbn-arxiv", choices=['Amazon_clothing', 'Amazon_electronics', 'corafull', 'dblp', 'coauthorCS', 'ogbn-arxiv'])
    parser.add_argument('--data_dir', type=str,
                        default='datasets', help='dir of datasets')
    parser.add_argument("--num_seed", type=int, default=5)
    parser.add_argument("--summary", type=str, default=timestr)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--way", type=int, default=5)  # N
    parser.add_argument("--shot", type=int, default=3)  # K
    parser.add_argument("--qry", type=int, default=5)  # M
    parser.add_argument(
        "--episodes", type=int, default=50, help="# of episodes to train"
    )
    parser.add_argument(
        "--meta_val_num", type=int, default=50, help="# of episodes for val"
    )
    parser.add_argument(
        "--meta_test_num", type=int, default=50, help="# of episodes for test"
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="loss weight coefficient")
    parser.add_argument("--n_layers", type=int,
                        default=2, help="# of EGNN layers")
    parser.add_argument("--anchor_size", type=int,
                        default=16, help="# of virtual anchor nodes")

    return parser.parse_known_args()


def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals


def printConfig(args):
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ""
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in [
            "lr",
            "epochs",
            "device",
            "seed",
            "data_dir"
            "num_seed",
            "summary",
            "dropout",
            "episodes",
            "meta_val_num",
            "meta_test_num",
            "optim",
            "eps",
            "l1",
            "l2",
            "final_result",
            "n_layers",
            "anchor_size"
        ]:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]

import os
from argument import config2string, parse_args
from utils import seed_everything
import torch
import numpy as np
import yaml
import time
import warnings
from model import teg_trainer

warnings.filterwarnings(action='ignore')

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# limit CPU usage
torch.set_num_threads(3)


def main(args):

    config_str = config2string(args)

    with open('configuration.yaml') as f:
        conf = yaml.safe_load(f)

    embedder = teg_trainer(args, conf, set_seed)

    (
        best_acc_train,
        best_f1_train,
        best_epoch_train,
        best_acc_valid,
        best_f1_valid,
        best_epoch_valid,
        best_acc_test,
        best_f1_test,
        best_epoch_test,
        test_acc_at_best_valid,
        test_f1_at_best_valid,
    ) = embedder.train()

    timestr = time.strftime("%m%d-%H%M")

    print("")
    print(f"# Final Result : {config_str} at {timestr}")
    print(
        f"# Best_Acc_Train : {best_acc_train}, F1 : {best_f1_train} at {best_epoch_train} epoch"
    )
    print(
        f"# Best_Acc_Valid : {best_acc_valid}, F1 : {best_f1_valid} at {best_epoch_valid} epoch"
    )
    print(
        f"# Best_Acc_Test : {best_acc_test}, F1 : {best_f1_test} at {best_epoch_test} epoch"
    )
    print(
        f"==> Acc_Test_At_Best_Valid : {test_acc_at_best_valid}, F1 : {test_f1_at_best_valid} at {best_epoch_valid} epoch"
    )
    print("")

    return test_acc_at_best_valid, test_f1_at_best_valid


if __name__ == "__main__":
    args, unknown = parse_args()

    accs = []
    f1s = []

    for set_seed in range(args.seed, args.seed + args.num_seed):
        seed_everything(set_seed)
        acc, f1 = main(args)
        try:
            accs.append(acc.item())
        except:
            accs.append(acc)

    accs = np.array(accs)

    acc_mean = np.mean(accs)
    acc_std = np.std(accs)

    timestr = time.strftime("%m%d-%H%M")
    config_str = config2string(args)

    np.set_printoptions(
        formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})

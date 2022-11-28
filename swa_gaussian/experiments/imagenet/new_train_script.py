from transformers import AutoModel, BertForMaskedLM

import argparse
import time
import tabulate
import os
import sys
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

from swag import utils
from swag.posteriors import SWAG
from swag.losses import bert_loss
from wiki_dataset import get_loaders

parser = argparse.ArgumentParser(description="SGD/SWA training")
parser.add_argument(
    "--dir",
    type=str,
    default=None,
    required=True,
    help="training directory (default: None)",
)

parser.add_argument(
    "--pretrained",
    action="store_true",
    help="pretrained model usage flag (default: off)",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5,
    metavar="N",
    help="number of epochs to train (default: 5)",
)

parser.add_argument(
    "--save_freq", type=int, default=1, metavar="N", help="save frequency (default: 1)"
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=1,
    metavar="N",
    help="evaluation frequency (default: 1)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    metavar="M",
    help="SGD momentum (default: 0.9)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)

parser.add_argument("--swa", action="store_true", help="swa usage flag (default: off)")
parser.add_argument(
    "--swa_cpu", action="store_true", help="store swag on cpu (default: off)"
)
parser.add_argument(
    "--swa_start",
    type=float,
    default=161,
    metavar="N",
    help="SWA start epoch number (default: 161)",
)
parser.add_argument(
    "--swa_lr", type=float, default=0.02, metavar="LR", help="SWA LR (default: 0.02)"
)
parser.add_argument(
    "--swa_freq",
    type=int,
    default=4,
    metavar="N",
    help="SWA model collection frequency/ num samples per epoch (default: 4)",
)
parser.add_argument("--cov_mat", action="store_true", help="save sample covariance")
parser.add_argument("--no_schedule", action="store_true", help="store schedule")

args = parser.parse_args()

if args.cov_mat:
    args.no_cov_mat = False
else:
    args.no_cov_mat = True

##############################################################################################

print("Preparing directory %s" % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, "command.sh"), "w") as f:
    f.write(" ".join(sys.argv))
    f.write("\n")

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
model = BertForMaskedLM.from_pretrained("prajjwal1/bert-tiny")
vocab_size = model.config.vocab_size
print("model vocab size: {}".format(vocab_size))

if args.swa:
    print("SWAG training")
    args.swa_device = device
    swag_model = SWAG(
        model,
        no_cov_mat=args.no_cov_mat,
        max_num_models=20
    )
    swag_model.to(args.swa_device)
    if args.pretrained:
        model.to(args.swa_device)
        swag_model.collect_model(model)
        model.to(args.device)
else:
    print("SGD training")

def schedule(epoch):
    if args.swa and epoch >= args.swa_start:
        return args.swa_lr
    else:
        return args.lr_init * (0.1 ** (epoch // 30))



# print("model parameters: ", len(list(model.parameters())))

optimizer = torch.optim.SGD(
    list(model.parameters()), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd
)
model.to(device)

mask_token, loaders = get_loaders()
print(loaders.keys())
# criterion = partial(bert_loss, mask_token=mask_token)
criterion = bert_loss

start_epoch = 0

columns = ["ep", "lr", "tr_loss", "tr_acc", "te_loss", "te_acc", "time", "mem_usage"]
if args.swa:
    columns = columns[:-2] + ["swa_te_loss", "swa_te_acc"] + columns[-2:]
    swag_res = {"loss": None, "accuracy": None}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    optimizer=optimizer.state_dict(),
)
num_iterates = 0

train_losses = np.empty(args.epochs)
validation_losses = np.empty(args.epochs)
swag_validation_losses = np.empty(args.epochs)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    if not args.no_schedule:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    print("EPOCH %d. TRAIN" % (epoch + 1))
    if args.swa and (epoch + 1) > args.swa_start:
        subset = 1.0 / args.swa_freq
        for i in range(args.swa_freq):
            print("PART %d/%d" % (i + 1, args.swa_freq))
            train_res = utils.train_epoch(
                loaders["train"],
                model,
                criterion,
                optimizer,
                subset=subset,
                verbose=True,
            )

            num_iterates += 1
            utils.save_checkpoint(
                args.dir, num_iterates, name="iter", state_dict=model.state_dict()
            )

            model.to(args.swa_device)
            swag_model.collect_model(model)
            model.to(device)
            swag_model.to(device)
    else:
        train_res = utils.train_epoch(
            loaders["train"], model, criterion, optimizer, verbose=True
        )

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        print("EPOCH %d. EVAL" % (epoch + 1))
        test_res = utils.eval(loaders["test"], model, criterion, verbose=True)
    else:
        test_res = {"loss": None, "accuracy": None}

    if args.swa and (epoch + 1) > args.swa_start:
        if (
            epoch == args.swa_start
            or epoch % args.eval_freq == args.eval_freq - 1
            or epoch == args.epochs - 1
        ):
            swag_res = {"loss": None, "accuracy": None}
            swag_model.to(device)
            swag_model.sample(0.0)
            print("EPOCH %d. SWAG BN" % (epoch + 1))
            utils.bn_update(loaders["train"], swag_model, verbose=True, subset=0.1)
            print("EPOCH %d. SWAG EVAL" % (epoch + 1))
            swag_res = utils.eval(loaders["test"], swag_model, criterion, verbose=True)
            swag_model.to(args.swa_device)
        else:
            swag_res = {"loss": None, "accuracy": None}

    if (epoch + 1) % args.save_freq == 0:
        if args.swa:
            utils.save_checkpoint(
                args.dir, epoch + 1, name="swag", state_dict=swag_model.state_dict()
            )
        else:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                state_dict=model.state_dict(),
                optimizer=optimizer.state_dict(),
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated() / (1024.0 ** 3)
    values = [
        epoch + 1,
        lr,
        train_res["loss"],
        train_res["accuracy"],
        test_res["loss"],
        test_res["accuracy"],
        time_ep,
        memory_usage,
    ]
    train_losses[epoch] = train_res["loss"]
    validation_losses[epoch] = test_res["loss"]
    if args.swa:
        values = values[:-2] + [swag_res["loss"], swag_res["accuracy"]] + values[-2:]
        swag_validation_losses = swag_res["loss"]
    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    table = table.split("\n")
    table = "\n".join([table[1]] + table)
    print(table)

plt.figure(0)
plt.plot(train_losses)
plt.savefig("train_plots/train_losses")
plt.figure(1)
plt.plot(validation_losses)
plt.savefig("train_plots/validation_losses")
if args.swa:
    plt.figure(2)
    plt.plot(swag_validation_losses)
    plt.savefig("train_plots/swag_validation_losses")




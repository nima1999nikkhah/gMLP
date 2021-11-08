import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import math
from efficientnet_pytorch import EfficientNet
from icecream import ic
import json
from PIL import Image
from einops.layers.torch import Rearrange
from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='ClassifierHybrid')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='Number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='Learning rate (default: 1e-3)')
parser.add_argument('--img_size', type=int, default=256,
                    help='Image size as input (default: 128)')
parser.add_argument('--num_classes', type=int, default=49,
                    help='Number of classes (default: 196)')
parser.add_argument('--multiplier', type=int, default=6,
                    help='Multiplier in SGU (default: 6)')
parser.add_argument('--patch_dim', type=int, default=192,
                    help='Dimention of patches (default: 192)')
parser.add_argument('--patch_size', type=int, default=16,
                    help='Patche sizes (default: 8)')
parser.add_argument('--num_layers', type=int, default=30,
                    help='Patche sizes (default: 30)')
parser.add_argument('--data_json', type=str, default='tmpe.json',
                    help='The root of the json file')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='Input batch size for training (default: 10)')
parser.add_argument('--survival_prob', type=int, default=1, metavar='N',
                    help='Survival probability (default: 1)')
parser.add_argument('--weight_decay', type=float, default=0.05, metavar='WD',
                    help='Weight decay (default: 0.05)')

args = parser.parse_args()


class gmlp_data(Dataset):

    def __init__(self, pth, transform_image, img_size):

        with open(pth, 'r') as f:
            self.dlist = json.load(f)

        self.labels = [
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            2,
            2,
            2,
            2,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            3,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            4,
            5,
            5,
            5,
            5,
            5,
            5,
            6,
            6,
            7,
            7,
            7,
            7,
            8,
            8,
            8,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            9,
            10,
            10,
            10,
            10,
            10,
            10,
            11,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            12,
            13,
            14,
            14,
            15,
            15,
            15,
            15,
            16,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            17,
            18,
            18,
            18,
            18,
            18,
            19,
            20,
            20,
            21,
            21,
            21,
            21,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            22,
            23,
            23,
            24,
            25,
            26,
            26,
            26,
            26,
            26,
            27,
            27,
            27,
            27,
            28,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            34,
            34,
            34,
            34,
            34,
            35,
            36,
            36,
            36,
            36,
            37,
            38,
            39,
            40,
            40,
            40,
            41,
            42,
            42,
            43,
            43,
            43,
            43,
            44,
            45,
            45,
            45,
            45,
            46,
            46,
            46,
            47,
            47,
            47,
            48]

        self.transform = transform_image
        self.img_size = img_size

    def __len__(self):

        return len(self.dlist)

    def __getitem__(self, indx):

        image_ = Image.open(self.dlist[indx][0])

        img_ = image_.resize((self.img_size, self.img_size))

        img = self.transform(img_)

        label_index = self.dlist[indx][1]
        clss = torch.tensor(self.labels[label_index])

        return (img, clss)


class Residual(nn.Module):
    def __init__(self, survival_prob, fn):
        super().__init__()
        self.prob = torch.rand(1)

        self.survival_prob = survival_prob
        self.fn = fn

    def forward(self, x):
        if self.prob <= self.survival_prob:
            return self.fn(x) + x
        else:
            return self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, **kwargs):
        super().__init__(**kwargs)
        self.norm = nn.LayerNorm(normalized_shape=dim, eps=1e-6)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x))


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim_seq, dim_ff):
        super().__init__()

        self.proj = nn.Linear(dim_seq, dim_seq)
        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
        self.norm = nn.LayerNorm(normalized_shape=dim_ff // 2, eps=1e-6)

        self.dim_ff = dim_ff

        self.activation = nn.GELU()

    def forward(self, x):

        # x: shape = (batch,  dim_seq,  channel)
        res, gate = torch.split(
            tensor=x, split_size_or_sections=self.dim_ff // 2, dim=2)  # ch
        # res, gate: shape = (batch,  dim_seq,  channel//2)
        gate = self.norm(gate)
        # gate: shape = (batch,  dim_seq,  channel//2)
        gate = torch.transpose(gate, 1, 2)
        # gate: shape = (batch,  channel//2,  dim_seq)
        gate = self.proj(gate)
        # gate: shape = (batch,  channel//2,  dim_seq)
        gate = torch.transpose(gate, 1, 2)
        # gate: shape = (batch,  dim_seq,  channel//2)
        return gate * res


class gMLPBlock(nn.Module):
    def __init__(self, dim, dim_ff, seq_len):
        super().__init__()

        self.proj_in = nn.Linear(dim, dim_ff)
        self.activation = nn.GELU()
        self.sgu = SpatialGatingUnit(seq_len, dim_ff)
        self.proj_out = nn.Linear(dim_ff // 2, dim)

    def forward(self, x):
        # shape=(B,  seq,  dim) --> (B,  seq,  dim_ff) --> (B,  seq,  dim_ff/2)
        # --> (B,  seq,  dim)
        x = self.proj_in(x)
        x = self.activation(x)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x


class gMLPNet(nn.Module):
    def __init__(
            self,
            survival_prob=0.5,
            image_size=256,
            patch_size=16,
            dim=128,
            depth=30,
            ff_mult=2,
            num_classes=196):
        super().__init__()

        self.image_size = image_size

        self.patch_size = patch_size

        self.patch_rearrange = Rearrange(
            'b c (h p) (w q) -> b (h w) (c p q)',
            p=self.patch_size,
            q=self.patch_size)  # (b,  3 ,  256,  256) -> (b,  16*16,  3*16*16)

        dim_ff = dim * ff_mult

        initial_dim = 3 * (patch_size ** 2)

        num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(initial_dim, dim)  # shape=(B,  seq,  dim)
        self.query = torch.nn.Parameter(torch.randn(1, 1, dim))
        self.query.requires_grad = True
        self.dim = dim

        module_list = [Residual(survival_prob,
                                PreNorm(dim,
                                        gMLPBlock(dim=dim,
                                                  dim_ff=dim_ff,
                                                  seq_len=num_patches + 1,
                                                  ))) for i in range(depth)]

        self.glayers = nn.Sequential(*module_list)

        self.norm = nn.LayerNorm(normalized_shape=dim, eps=1e-5)

        self.classification_rearrange = Rearrange('b s d -> b (s d)')

        self.clssification_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes)
        )

    def extract_patches(self, images):

        batch_size = images.size(0)
        patches = self.patch_rearrange(images)

        return patches

    def forward(self, x):

        # shape=(B,  num_patches,  patch_size**2 * C)
        x = self.extract_patches(x)
        x = self.patch_embed(x)  # shape=(B,  num_patches,  dim)
        B, seq, dim = x.shape
        query_1 = self.query.repeat((B, 1, 1))
        x = torch.cat([query_1, x], dim=1)
        x = self.glayers(x)  # shape=(B,  num_patches,  dim)
        x = self.norm(x)  # shape=(B,  num_patches,  dim)
        x = x[:, 0:1, :]

        x = self.classification_rearrange(x)

        x = self.clssification_head(x)

        return x


def main():

    transform_image = transforms.Compose([transforms.ToTensor()])

    dtst = gmlp_data(
        pth=args.data_json,
        transform_image=transform_image,
        img_size=args.img_size)
    dtld = DataLoader(dtst, shuffle=True, batch_size=args.batch_size)

    model = gMLPNet(
        survival_prob=args.survival_prob,
        image_size=args.img_size,
        patch_size=args.patch_size,
        dim=args.patch_dim,
        depth=args.num_layers,
        ff_mult=args.multiplier,
        num_classes=args.num_classes)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    ic(count_parameters(model))
    model = model.to(device).train()

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(
            0.9,
            0.999),
        eps=1e-08,
        weight_decay=args.weight_decay,
        amsgrad=False)

    log_dir = './logs'
    writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    iteration = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        num_true = 0
        num_total = 0
        count = 0
        with tqdm(dtld, unit="batch") as t_iter:
            for data in t_iter:
                iteration += 1
                inpt, label = data
                inpt = inpt.to(device)
                label = label.to(device)
                optimizer.zero_grad()

                output = model(inpt)

                flag = (torch.argmax(output, dim=1) == label)

                sm = torch.sum(flag)
                num_true_1_batch = sm.cpu().numpy()
                num_total += args.batch_size
                num_true += num_true_1_batch

                error = criterion(output, label)
                error.backward()
                y = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += error.detach().cpu().numpy()
                count += 1
                train_accuracy = num_true_1_batch / args.batch_size

                writer.add_scalar(
                    'Loss/train',
                    error.detach().cpu().numpy(),
                    iteration)
                writer.add_scalar('Accuracy/train', train_accuracy, iteration)
                writer.flush()
                t_iter.set_postfix(loss=error.item(),
                                   running_loss=running_loss / count,
                                   total_train_accuracy=num_true / num_total,
                                   train_accuracy=train_accuracy
                                   )


if __name__ == "__main__":
    main()

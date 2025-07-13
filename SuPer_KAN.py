import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from shape_kan import SHAP_KAN_layer

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

class classification(nn.Module):
    def __init__(self, input_size, output_size):
        super(classification, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, seg_dim=8, num_class=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim = seg_dim

        # self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        # self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.mlp_c = SHAP_KAN_layer(dim, dim, SimpleNN(dim, dim), classification(dim, num_class))
        self.mlp_w = SHAP_KAN_layer(dim, dim, SimpleNN(dim, dim), classification(dim, num_class))

        self.reweighting = MLP(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, W, C = x.shape  # 64,21,5, n
        c_embed = self.mlp_c(x)
        S = C // self.seg_dim
        w_embed = x.reshape(B, W, self.seg_dim, S).permute(0, 2, 1, 3).reshape(B, self.seg_dim, W * S)
        w_embed = self.mlp_w(w_embed).reshape(B, self.seg_dim, W, S).permute(0, 2, 1, 3).reshape(B, W, C)

        weight = (c_embed + w_embed).permute(0, 2, 1).flatten(2).mean(2)
        weight = self.reweighting(weight).reshape(B, C, 3).permute(2, 0, 1).softmax(0).unsqueeze(2)

        x = c_embed * weight[0] + w_embed * weight[1]

        x = self.proj_drop(self.proj(x))

        return x


class block(nn.Module):
    def __init__(self, dim, depth, mlp_dim, seg_dim, num_class, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, WeightedPermuteMLP(dim, seg_dim, num_class)),  # dim = head_dim
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):  # 64, 8, 5120

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SuPerKAN(nn.Module):
    def __init__(self, *, get_feature, patch_size, forecast_len, num_class, dim=64, depth=6, mlp_dim=64, pool='cls', in_channel=3, dropout=0.1,
                 emb_dropout=0., get_get_middle=64):  # dim = head_dim
        super().__init__()
        image_height = get_feature
        patch_height = patch_size

        # assert image_height % patch_height == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height)
        patch_dim = in_channel * patch_height
        dim = dim * num_patches
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.get_get_middle = get_get_middle
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) -> b (h) (p1 c)', p1=patch_height),  # b c (h p1) (w p2) -> b (h w) (p1 p2 c)
            nn.Linear(patch_dim, dim),  # b, h*w, dim
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.block = block(dim, depth, mlp_dim, num_patches, dropout, num_class)  # dim = head_dim

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.get_get_middle)
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(self.get_get_middle),
            nn.Linear(self.get_get_middle, forecast_len)
        )
        self.patcher = nn.Sequential(
            nn.Conv1d(in_channel, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c l -> b l c"),
        )



    def forward(self, img):
        # x = self.to_patch_embedding(img)  # b, c, dim
        x = self.patcher(img)  # b, c, dim
        # print('to-patch-embedding', x.shape)
        b, n, _ = x.shape  #
        # print(x.shape)
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # print('cat-cls', x.shape)
        # x += self.pos_embedding[:, :(n + 1)]
        # print('add-pos', x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.block(x)
        # print('after transformer', x.shape)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head1(x)
        xs = self.mlp_head2(x)
        out_x = xs.view(xs.size(0), -1)
        return x, out_x

if __name__ == '__main__':
    model = SuPerKAN(get_feature=21, patch_size=3, forecast_len=1, in_channel=14, get_get_middle = 64, num_class = 8).cuda()
    out1, out2 = model(torch.rand(64, 14, 21).cuda())
    print('out_1:',out1.shape)
    print('out_2:',out2.shape)
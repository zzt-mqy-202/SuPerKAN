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
import shap

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

class KAN_layer(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=2,
            spline_order=2,
            scale_noise=0.05,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=False,
            # base_activation=torch.nn.SiLU,
            base_activation=torch.nn.GELU,
            grid_eps=0.1,
            grid_range=[-3, 3],
    ):
        super(KAN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.net = SimpleNN(int(in_features*(grid_size + spline_order)), out_features)
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        # print(x.shape)
        # print("--KAN_layer--")
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        # print(self.scaled_spline_weight.view(self.out_features, -1).shape)
        # print(self.b_splines(x).view(x.size(0), -1).shape)
        b_splines = self.b_splines(x).view(x.size(0), -1)
        explainer = shap.DeepExplainer(self.net.cuda(), b_splines[:10].cuda())
        shap_values = np.mean(explainer.shap_values(b_splines[:10].cuda()), axis=0).swapaxes(1, 0)
        # print(shap_values.shape)
        # print(self.b_splines(x).view(x.size(0), -1).shape)

        # spline_output = F.linear(
        #     self.b_splines(x).view(x.size(0), -1),
        #     self.scaled_spline_weight.view(self.out_features, -1),
        # )
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            torch.from_numpy(shap_values).type(torch.FloatTensor).cuda(),
        )
        output = base_output + spline_output

        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


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
        self.fc1 = KAN_layer(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = KAN_layer(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim = seg_dim

        # self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        # self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        self.mlp_c = KAN_layer(dim, dim)
        self.mlp_w = KAN_layer(dim, dim)

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
    def __init__(self, dim, depth, mlp_dim, seg_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, WeightedPermuteMLP(dim, seg_dim)),  # dim = head_dim
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):  # 64, 8, 5120

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SuPerKAN(nn.Module):
    def __init__(self, *, get_feature, patch_size, num_class, dim=64, depth=3, mlp_dim=64, pool='cls', in_channel=3, dropout=0.1,
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

        self.block = block(dim, depth, mlp_dim, num_patches, dropout)  # dim = head_dim

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.get_get_middle)
        )

        self.mlp_head2 = nn.Sequential(
            nn.LayerNorm(self.get_get_middle),
            nn.Linear(self.get_get_middle, num_class)
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
    model = SuPerKAN(get_feature=21, patch_size=3, num_class=1, in_channel=14, get_get_middle = 64).cuda()
    out1, out2 = model(torch.rand(64, 14, 21).cuda())
    print('out_1:',out1.shape)
    print('out_2:',out2.shape)

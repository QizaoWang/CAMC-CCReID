"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ACCV 2022 Oral paper: Co-Attention Aligned Mutual Cross-Attention for Cloth-Changing Person Re-Identification
URL: https://openaccess.thecvf.com/content/ACCV2022/html/Wang_Co-Attention_Aligned_Mutual_Cross-Attention_for_Cloth-Changing_Person_Re-Identification_ACCV_2022_paper.html
GitHub: https://github.com/QizaoWang/CAMC-CCReID
"""

from __future__ import absolute_import
import math
import cv2
import numpy as np

import torch
from torch import nn
from torch.nn import init
import torchvision
import torch.nn.functional as F

from models.pose_net import SimpleHRNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0.0)


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class ResnetBackbone(nn.Module):
    def __init__(self, pretrain=True, last_stride=1):
        super(ResnetBackbone, self).__init__()
        self.feature_dim = 2048

        resnet = torchvision.models.resnet50(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        x = self.backbone(x)
        return x


class GetHeatmap(nn.Module):
    def __init__(self, args, resolution=(256, 128)):
        super().__init__()
        self.k = 13  # 17, max face
        self.pose_net = SimpleHRNet(c=48, nof_joints=17, checkpoint_path=args.pose_net_path,
                                    model_name='HRNet', resolution=resolution, interpolation=cv2.INTER_CUBIC,
                                    multiperson=False, return_heatmaps=True, return_bounding_boxes=False,
                                    max_batch_size=32, device=torch.device("cuda"))

    def forward(self, x):
        batch_size = x.shape[0]
        heatmap, _ = self.pose_net.predict(x)  # [B, 17, H//4, W//4]

        # max face
        heatmap_temp = np.zeros((batch_size, self.k, heatmap.shape[-2], heatmap.shape[-1]), dtype=np.float32)
        heatmap_temp[:, 0, :, :] = np.max(heatmap[:, 0: 5, :, :], axis=1)
        heatmap_temp[:, 1:, :, :] = heatmap[:, 5:, :, :]
        heatmap = heatmap_temp

        return torch.from_numpy(heatmap).cuda()  # [B, k, H//4, W//4]


class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=[256, 128], patch_size=[16, 16], stride_size=[20, 20], in_chans=1, embed_dim=2048):
        super().__init__()
        self.num_x = (img_size[1] - patch_size[1]) // stride_size[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class SSE(nn.Module):
    def __init__(self, feature_dim=2048, num_head=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head

        self.w_q = nn.Linear(feature_dim, feature_dim, bias=False).apply(_init_vit_weights)
        self.w_k = nn.Linear(feature_dim, feature_dim, bias=False).apply(_init_vit_weights)
        self.w_v = nn.Linear(feature_dim, feature_dim, bias=False).apply(_init_vit_weights)

        self.layer_norm = nn.LayerNorm(feature_dim, eps=1e-6).apply(_init_vit_weights)
        self.mlp = Mlp(in_features=feature_dim, hidden_features=feature_dim * 2, act_layer=nn.ReLU, drop=0.).apply(
                _init_vit_weights)

    def forward(self, x):
        B, N, C = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q = q.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k = k.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v = v.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn = torch.matmul(q / ((C // self.num_head) ** 0.5), k.transpose(-1, -2))
        attn = F.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output = output.transpose(1, 2).contiguous().flatten(2)

        output = self.layer_norm(output + x)
        output = self.mlp(output)
        return output


class MutualCrossAttn(nn.Module):
    def __init__(self, feature_dim=2048, num_head=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_head = num_head

        self.layer_norm1 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)
        self.layer_norm2 = nn.LayerNorm(self.feature_dim, eps=1e-6).apply(_init_vit_weights)

    def forward(self, f_a, f_s):
        B, N, C = f_a.shape

        q1 = f_a
        k1 = v1 = f_s
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q1 = q1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k1 = k1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v1 = v1.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn1 = torch.matmul(q1 / ((C // self.num_head) ** 0.5),
                             k1.transpose(-1, -2))
        attn1 = F.softmax(attn1, dim=-1)
        output1 = torch.matmul(attn1, v1)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output1 = output1.transpose(1, 2).contiguous().flatten(2)

        q2 = f_s
        k2 = v2 = f_a
        # [B, N, C] -> [B, N, n, C//n] -> [B, n, N, C//n]
        q2 = q2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        k2 = k2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)
        v2 = v2.reshape(B, N, self.num_head, C // self.num_head).transpose(1, 2)

        attn2 = torch.matmul(q2 / ((C // self.num_head) ** 0.5),
                             k2.transpose(-1, -2))  # attn2 matrix is equal to the transpose of attn1
        attn2 = F.softmax(attn2, dim=-1)
        output2 = torch.matmul(attn2, v2)
        # [B, n, N, C//n] -> [B, N, n, C//n] -> [B, N, C]
        output2 = output2.transpose(1, 2).contiguous().flatten(2)

        output1 = self.layer_norm1(output1 + f_a)
        output2 = self.layer_norm2(output2 + f_s)
        return output1, output2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)
        self.layer_norm = nn.LayerNorm(in_features, eps=1e-6)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        x = self.layer_norm(residual + x)
        return x


class CAMC(nn.Module):
    def __init__(self, args, num_classes, pretrain=True, last_stride=1):
        super().__init__()
        self.num_classes = num_classes
        self.img_backbone = ResnetBackbone(pretrain, last_stride)
        self.get_heatmap = GetHeatmap(args)

        self.feature_dim = feature_dim = self.img_backbone.feature_dim
        num_multi_head = 32

        self.k_embed = nn.Sequential(
            PatchEmbed_overlap(img_size=[64, 32], patch_size=[4, 4], stride_size=[4, 4],
                               in_chans=self.get_heatmap.k, embed_dim=feature_dim),
            nn.ReLU(inplace=True)
        )

        self.sse = SSE(feature_dim=feature_dim, num_head=num_multi_head)

        embed_dim = feature_dim // 2
        self.co_attn_align = nn.Sequential(
            nn.Linear(2 * feature_dim, embed_dim).apply(weights_init_kaiming),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 2 * feature_dim).apply(weights_init_kaiming),
            nn.Sigmoid()
        )
        self.mutual_cross_attn = MutualCrossAttn(feature_dim=feature_dim, num_head=num_multi_head)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.bottleneck = nn.BatchNorm1d(2 * feature_dim).apply(weights_init_kaiming)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(2 * feature_dim, self.num_classes, bias=False).apply(weights_init_classifier)

    def forward(self, x):
        f_a = self.img_backbone(x)
        f_a = f_a.flatten(2).transpose(-1, -2)  # [B, hw, d]

        f_raw = self.get_heatmap(x)  # [B, k, H//4, W//4]
        f_k = self.k_embed(f_raw)  # [B, hw, d]
        f_s = self.sse(f_k)  # [B, hw, d]

        f_c = torch.concat((f_a, f_s), dim=-1)
        align_score = self.co_attn_align(f_c)
        aligned_p = align_score[:, :, 0: self.feature_dim] * f_a + f_a
        aligned_s = align_score[:, :, self.feature_dim:] * f_s + f_s

        p_s, s_p = self.mutual_cross_attn(aligned_p, aligned_s)

        feat = torch.concat((p_s, s_p), dim=-1)  # [B, hw, 2d]
        feat = self.gap(feat.transpose(-1, -2)).squeeze(-1)  # [B, 2d]
        feat_bn = self.bottleneck(feat)
        y = self.classifier(feat_bn)

        if not self.training:
            return feat_bn
        else:
            return y

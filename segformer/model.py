import torch
from torch import nn
from torch.nn.functional import interpolate

from segformer.backbones import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5, MixTransformer
from segformer.heads import SegFormerHead

model_urls = {
    # Complete SegFormer weights trained on ADE20K.
    'ade': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_512x512_ade_160k-d0c08cfd.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_512x512_ade_160k-1cd52578.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_512x512_ade_160k-fa162a4f.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_512x512_ade_160k-5abb3eb3.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_512x512_ade_160k-bb0fa50c.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_640x640_ade_160k-106a5e57.pth',
    },
    # Complete SegFormer weights trained on CityScapes.
    'city': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_1024x1024_city_160k-3e581249.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_1024x1024_city_160k-e415b121.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_1024x1024_city_160k-9793f658.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_1024x1024_city_160k-732b9fde.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_1024x1024_city_160k-1836d907.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_1024x1024_city_160k-2ca4dff8.pth',
    },
    # Backbone-only SegFormer weights trained on ImageNet.
    'imagenet': {
        'segformer_b0': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b0_backbone_imagenet-eb42d485.pth',
        'segformer_b1': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b1_backbone_imagenet-357971ac.pth',
        'segformer_b2': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b2_backbone_imagenet-3c162bb8.pth',
        'segformer_b3': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b3_backbone_imagenet-0d113e32.pth',
        'segformer_b4': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b4_backbone_imagenet-b757a54d.pth',
        'segformer_b5': 'https://github.com/anibali/segformer/releases/download/v0.0.0/segformer_b5_backbone_imagenet-d552b33d.pth',
    },
}


class SegFormer(nn.Module):
    def __init__(self, backbone: MixTransformer, decode_head: SegFormerHead):
        super().__init__()
        self.backbone = backbone
        self.decode_head = decode_head

    @property
    def align_corners(self):
        return self.decode_head.align_corners

    @property
    def num_classes(self):
        return self.decode_head.num_classes

    # def forward(self, x):
    #     image_hw = x.shape[2:]
    #     feats = self.backbone(x)
    #     x = self.decode_head(feats, x)  # 这里多传一个原图 x
    #     x = interpolate(x, size=image_hw, mode='bilinear', align_corners=self.align_corners)
    #     return x
    def forward(self, x):
        image_hw = x.shape[2:]
        x = self.backbone(x)
        x = self.decode_head(x)
        x = interpolate(x, size=image_hw, mode='bilinear', align_corners=self.align_corners)#上采样
        return x


def create_segformer_b0(num_classes):
    backbone = mit_b0()
    head = SegFormerHead(
        in_channels=(32, 64, 160, 256),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)


def create_segformer_b1(num_classes):
    backbone = mit_b1()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=256,
    )
    return SegFormer(backbone, head)
import torch
import torch.nn as nn
from torch.nn.functional import interpolate, relu, dropout
import torch.nn.functional as F
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        mid_channels = max(in_channels // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EEM(nn.Module):
    def __init__(self, in_channels=3, out_channels=256):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, dilation=1)
        self.conv2 = ConvBNReLU(out_channels, out_channels, dilation=2)
        self.conv3 = ConvBNReLU(out_channels, out_channels, dilation=5)
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.cbam(x)
        return x

class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        if dilation == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = dilation

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = ASPPConv(in_channels, out_channels, dilation=1)
        self.branch2 = ASPPConv(in_channels, out_channels, dilation=2)
        self.branch3 = ASPPConv(in_channels, out_channels, dilation=7)
        self.branch4 = ASPPConv(in_channels, out_channels, dilation=15)
        self.branch5 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x)
        ], dim=1)
        x = self.project(x)
        return x
class EEMASPPHead(nn.Module):
    def __init__(self, in_channels, num_classes, embed_dim=256, dropout_p=0.1, align_corners=False):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p
        self.align_corners = align_corners
        self.num_classes = num_classes

        # 4个编码器特征，各自接一个 ASPP
        self.aspp_blocks = nn.ModuleList([
            ASPP(ch, embed_dim) for ch in in_channels
        ])

        # 边缘分支
        self.eem = EEM(in_channels=3, out_channels=embed_dim)

        # 4个 ASPP 输出 + 1个 EEM 输出
        self.linear_fuse = nn.Conv2d(embed_dim * 5, embed_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.dropout = nn.Dropout2d(dropout_p)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.linear_fuse.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, features, image):
        # features = [f1, f2, f3, f4]
        feats_hw = features[0].shape[2:]   # 对齐到第1层特征大小

        aspp_outs = []
        for feat, aspp in zip(features, self.aspp_blocks):
            y = aspp(feat)
            if y.shape[2:] != feats_hw:
                y = interpolate(y, size=feats_hw, mode='bilinear', align_corners=self.align_corners)
            aspp_outs.append(y)

        # EEM 边缘分支
        edge_feat = self.eem(image)
        edge_feat = interpolate(edge_feat, size=feats_hw, mode='bilinear', align_corners=self.align_corners)

        # 拼接 O1 + O2
        x = torch.cat(aspp_outs + [edge_feat], dim=1)

        x = self.linear_fuse(x)
        x = self.bn(x)
        x = relu(x, inplace=True)
        x = self.dropout(x)
        x = self.linear_pred(x)

        return x

# def create_segformer_b2(num_classes):
#     backbone = mit_b2()
#     head = EEMASPPHead(
#         in_channels=(64, 128, 320, 512),
#         dropout_p=0.1,
#         num_classes=num_classes,
#         align_corners=False,
#         embed_dim=256,
#     )
#     return SegFormer(backbone, head)
def create_segformer_b2(num_classes):   # 原来的
    backbone = mit_b2()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b3(num_classes):
    backbone = mit_b3()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b4(num_classes):
    backbone = mit_b4()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def create_segformer_b5(num_classes):
    backbone = mit_b5()
    head = SegFormerHead(
        in_channels=(64, 128, 320, 512),
        dropout_p=0.1,
        num_classes=num_classes,
        align_corners=False,
        embed_dim=768,
    )
    return SegFormer(backbone, head)


def _load_pretrained_weights_(model, progress):
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    # state_dict = torch.load("./weights/segformer_b5.pth", map_location='cpu')
    state_dict = torch.load("./weights/mit_b2.pth", map_location='cpu')
    # state_dict = torch.hub.load_state_dict_from_url(model_url, progress=progress)
    del_keys = ['head.weight', 'head.bias']
    for k in del_keys:
        del state_dict[k]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('decode_head'):
            if k.endswith('.proj.weight'):
                k = k.replace('.proj.weight', '.weight')
                v = v[..., None, None]
            elif k.endswith('.proj.bias'):
                k = k.replace('.proj.bias', '.bias')
            elif '.linear_fuse.conv.' in k:
                k = k.replace('.linear_fuse.conv.', '.linear_fuse.')
            elif '.linear_fuse.bn.' in k:
                k = k.replace('.linear_fuse.bn.', '.bn.')

            if '.linear_c4.' in k:
                k = k.replace('.linear_c4.', '.layers.0.')
            elif '.linear_c3.' in k:
                k = k.replace('.linear_c3.', '.layers.1.')
            elif '.linear_c2.' in k:
                k = k.replace('.linear_c2.', '.layers.2.')
            elif '.linear_c1.' in k:
                k = k.replace('.linear_c1.', '.layers.3.')
        else:
            if 'patch_embed1.' in k:
                k = k.replace('patch_embed1.', 'stages.0.patch_embed.')
            elif 'patch_embed2.' in k:
                k = k.replace('patch_embed2.', 'stages.1.patch_embed.')
            elif 'patch_embed3.' in k:
                k = k.replace('patch_embed3.', 'stages.2.patch_embed.')
            elif 'patch_embed4.' in k:
                k = k.replace('patch_embed4.', 'stages.3.patch_embed.')
            elif 'block1.' in k:
                k = k.replace('block1.', 'stages.0.blocks.')
            elif 'block2.' in k:
                k = k.replace('block2.', 'stages.1.blocks.')
            elif 'block3.' in k:
                k = k.replace('block3.', 'stages.2.blocks.')
            elif 'block4.' in k:
                k = k.replace('block4.', 'stages.3.blocks.')
            elif 'norm1.' in k:
                k = k.replace('norm1.', 'stages.0.norm.')
            elif 'norm2.' in k:
                k = k.replace('norm2.', 'stages.1.norm.')
            elif 'norm3.' in k:
                k = k.replace('norm3.', 'stages.2.norm.')
            elif 'norm4.' in k:
                k = k.replace('norm4.', 'stages.3.norm.')

            if '.mlp.dwconv.dwconv.' in k:
                k = k.replace('.mlp.dwconv.dwconv.', '.mlp.conv.')

            if '.mlp.' in k:
                k = k.replace('.mlp.', '.ffn.')
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)


def segformer_b0_ade(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the ADE20K segmentation task.
    """
    model = create_segformer_b0(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b0'], progress=progress)
    return model


def segformer_b1_ade(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the ADE20K segmentation task.
    """
    model = create_segformer_b1(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b1'], progress=progress)
    return model


def segformer_b2_ade(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the ADE20K segmentation task.
    """
    model = create_segformer_b2(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b2'], progress=progress)
    return model


def segformer_b3_ade(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the ADE20K segmentation task.
    """
    model = create_segformer_b3(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b3'], progress=progress)
    return model


def segformer_b4_ade(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the ADE20K segmentation task.
    """
    model = create_segformer_b4(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b4'], progress=progress)
    return model


def segformer_b5_ade(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the ADE20K segmentation task.
    """
    model = create_segformer_b5(num_classes=150)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['ade']['segformer_b5'], progress=progress)
    return model


def segformer_b0_city(pretrained=True, progress=True):
    """Create a SegFormer-B0 model for the CityScapes segmentation task.
    """
    model = create_segformer_b0(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b0'], progress=progress)
    return model


def segformer_b1_city(pretrained=True, progress=True):
    """Create a SegFormer-B1 model for the CityScapes segmentation task.
    """
    model = create_segformer_b1(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b1'], progress=progress)
    return model


def segformer_b2_city(pretrained=True, progress=True):
    """Create a SegFormer-B2 model for the CityScapes segmentation task.
    """
    model = create_segformer_b2(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b2'])
    return model


def segformer_b3_city(pretrained=True, progress=True):
    """Create a SegFormer-B3 model for the CityScapes segmentation task.
    """
    model = create_segformer_b3(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b3'], progress=progress)
    return model


def segformer_b4_city(pretrained=True, progress=True):
    """Create a SegFormer-B4 model for the CityScapes segmentation task.
    """
    model = create_segformer_b4(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b4'], progress=progress)
    return model


def segformer_b5_city(pretrained=True, progress=True):
    """Create a SegFormer-B5 model for the CityScapes segmentation task.
    """
    model = create_segformer_b5(num_classes=19)
    if pretrained:
        _load_pretrained_weights_(model, model_urls['city']['segformer_b5'], progress=progress)
    return model


def segformer_b0(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B0 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b0(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b0'],
                                  progress=progress)
    return model


def segformer_b1(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B1 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b1(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b1'],
                                  progress=progress)
    return model


def segformer_b2(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B2 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b2(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, progress=progress)
    return model


def segformer_b3(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B3 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b3(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b3'],
                                  progress=progress)
    return model


def segformer_b4(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B4 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b4(num_classes=num_classes)
    if pretrained:
        _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b4'],
                                  progress=progress)
    return model


def segformer_b5(pretrained=False, progress=True, num_classes=150):
    """Create a SegFormer-B5 model.

    Args:
        pretrained: Download backbone weights pretrained on ImageNet data if true.
        progress: Display the download progress of pretrained weights if true.
        num_classes: Number of output classes;.
    """
    model = create_segformer_b5(num_classes=num_classes)
    if pretrained:
        # _load_pretrained_weights_(model.backbone, model_urls['imagenet']['segformer_b5'],
        #                           progress=progress)
        _load_pretrained_weights_(model.backbone, progress=progress)
    return model

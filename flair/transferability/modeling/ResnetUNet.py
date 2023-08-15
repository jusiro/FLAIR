"""
Adapting ResNet to segmentation via decoder and skip connections, similar to UNet models.

Adapted from:
https://github.com/marshuang80/gloria/blob/main/gloria/models/unet.py
"""

import torch.nn as nn
import torch
import torchvision


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        with_nonlinearity=True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels), ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of:
        Upsample->ConvBlock->ConvBlock
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        up_conv_in_channels=None,
        up_conv_out_channels=None,
        upsampling_method="conv_transpose",
    ):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(
                up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2
            )
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x, concatenate=True):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torchvision.transforms.Resize((down_x.shape[-2], down_x.shape[-1]))(x)
        if concatenate:
            x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class ResnetUNet(nn.Module):
    FEATS = [64, 256, 512, 1024, 2048]
    # depth=6

    def __init__(self, pretrained_encoder=None, depth=6, update_bn=True):
        super().__init__()
        self.DEPTH = depth
        self.update_bn = update_bn

        # Load backbone
        if pretrained_encoder is None:
            weights = 'IMAGENET1K_V1'
            resnet = torchvision.models.resnet50(weights=weights)
        else:
            resnet = pretrained_encoder.model

        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(ResnetUNet.FEATS[self.DEPTH-2], ResnetUNet.FEATS[self.DEPTH-2])
        up_blocks.append(UpBlock(2048, 1024))
        up_blocks.append(UpBlock(1024, 512))
        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(
            UpBlock(
                in_channels=128 + 64,
                out_channels=128,
                up_conv_in_channels=256,
                up_conv_out_channels=128,
            )
        )
        up_blocks.append(
            UpBlock(
                in_channels=64 + 3,
                out_channels=64,
                up_conv_in_channels=128,
                up_conv_out_channels=64,
            )
        )

        self.up_blocks = nn.ModuleList(up_blocks)

    def forward(self, x):

        if self.training:
            training = True
            if not self.update_bn:
                self.eval()
        else:
            training = False

        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            if i > (self.DEPTH - 1):
                continue
            x = block(x)
            pre_pools[f"layer_{i}"] = x

        if training:
            self.train()

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks[(len(self.up_blocks) - self.DEPTH + 1):], 1):
            key = f"layer_{self.DEPTH - 1 - i}"
            # print(key)
            if key == "layer_0":
                x = block(x, pre_pools[key], concatenate=True)
            else:
                x = block(x, pre_pools[key])
        output_feature_map = x
        del pre_pools
        return output_feature_map

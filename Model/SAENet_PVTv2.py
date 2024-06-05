import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.PVTv2 import pvt_v2_b4
from Model.Modules import ConvBNReLU, PPM, SqueezeExcitation


class ConvBlock(nn.Module):
    def __init__(self, channel, dilation=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )
        # Dilation Conv
        self.dconv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channel)
        )
        # Asymmetric conv
        self.asconv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(channel)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channel * 2, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.cat((self.dconv(x), self.asconv(x)), dim=1)
        x = self.out_conv(x)
        return x


class DEMS(nn.Module):
    """
    Detail Enhanced Multi-Scale Module
    """

    def __init__(self, channel):
        super(DEMS, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = ConvBlock(channel, dilation=1)
        self.branch1 = ConvBlock(channel, dilation=3)
        self.branch2 = ConvBlock(channel, dilation=5)
        self.branch3 = ConvBlock(channel, dilation=7)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 3, channel, kernel_size=1),
            nn.BatchNorm2d(channel)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x + x1)
        x3 = self.branch3(x + x2)
        x_cat = self.conv_cat(torch.cat((x1, x2, x3), dim=1))
        x = self.relu(x0 + x_cat)
        return x


class WeightGenerator(nn.Module):
    def __init__(self, channel):
        super(WeightGenerator, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap_mlp = ConvBNReLU(in_channels=channel, out_channels=1, kernel_size=1)
        self.gmp_mlp = ConvBNReLU(in_channels=channel, out_channels=1, kernel_size=1)

    def forward(self, x):
        x1 = self.gap_mlp(self.gap(x))
        x2 = self.gmp_mlp(self.gmp(x))
        return x1 + x2


class WeightStageFeature(nn.Module):
    def __init__(self, channel):
        super(WeightStageFeature, self).__init__()
        self.wg1 = WeightGenerator(channel)
        self.wg2 = WeightGenerator(channel)
        self.wg3 = WeightGenerator(channel)
        self.wg4 = WeightGenerator(channel)

    def forward(self, x1, x2, x3, x4):
        w1 = self.wg1(x1)  # [B, 1, 1, 1]
        w2 = self.wg2(x2)  # [B, 1, 1, 1]
        w3 = self.wg3(x3)  # [B, 1, 1, 1]
        w4 = self.wg4(x4)  # [B, 1, 1, 1]
        w = torch.cat((w1, w2, w3, w4), dim=1)
        w = torch.softmax(w, dim=1)
        x1 = w[:, 0, :, :].unsqueeze(1) * x1
        x2 = w[:, 1, :, :].unsqueeze(1) * x2
        x3 = w[:, 2, :, :].unsqueeze(1) * x3
        x4 = w[:, 3, :, :].unsqueeze(1) * x4
        return x1, x2, x3, x4


class WDSA(nn.Module):
    """
    Weighted Dense Semantic Aggregation Module
    """

    def __init__(self, channel):
        super(WDSA, self).__init__()
        # stage 1
        self.f1_1_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 2, reduction=4)
        self.f2_1_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f3_1_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f4_1_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f5_1_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        # stage 2
        self.f1_2_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f2_2_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 2, reduction=4)
        self.f3_2_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f4_2_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f5_2_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        # stage 3
        self.f1_3_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f2_3_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f3_3_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 2, reduction=4)
        self.f4_3_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f5_3_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        # stage 4
        self.f1_4_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f2_4_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f3_4_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f4_4_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 2, reduction=4)
        self.f5_4_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        # stage 5
        self.f1_5_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f2_5_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f3_5_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f4_5_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 8, reduction=4)
        self.f5_5_conv = SqueezeExcitation(in_channels=channel, out_channels=channel // 2, reduction=4)
        # stage selection
        self.wf1 = WeightStageFeature(channel=channel // 8)
        self.wf2 = WeightStageFeature(channel=channel // 8)
        self.wf3 = WeightStageFeature(channel=channel // 8)
        self.wf4 = WeightStageFeature(channel=channel // 8)
        self.wf5 = WeightStageFeature(channel=channel // 8)


    def forward(self, f1, f2, f3, f4, f5):
        ''' feature Selection '''
        # stage 1 feature
        f1_1 = F.interpolate(self.f1_1_conv(f1), size=f1.shape[2:], mode='bilinear', align_corners=True)
        f2_1 = F.interpolate(self.f2_1_conv(f2), size=f1.shape[2:], mode='bilinear', align_corners=True)
        f3_1 = F.interpolate(self.f3_1_conv(f3), size=f1.shape[2:], mode='bilinear', align_corners=True)
        f4_1 = F.interpolate(self.f4_1_conv(f4), size=f1.shape[2:], mode='bilinear', align_corners=True)
        f5_1 = F.interpolate(self.f5_1_conv(f5), size=f1.shape[2:], mode='bilinear', align_corners=True)
        # stage 2 feature
        f1_2 = F.interpolate(self.f1_2_conv(f1), size=f2.shape[2:], mode='bilinear', align_corners=True)
        f2_2 = F.interpolate(self.f2_2_conv(f2), size=f2.shape[2:], mode='bilinear', align_corners=True)
        f3_2 = F.interpolate(self.f3_2_conv(f3), size=f2.shape[2:], mode='bilinear', align_corners=True)
        f4_2 = F.interpolate(self.f4_2_conv(f4), size=f2.shape[2:], mode='bilinear', align_corners=True)
        f5_2 = F.interpolate(self.f5_2_conv(f5), size=f2.shape[2:], mode='bilinear', align_corners=True)
        # stage 3 feature
        f1_3 = F.interpolate(self.f1_3_conv(f1), size=f3.shape[2:], mode='bilinear', align_corners=True)
        f2_3 = F.interpolate(self.f2_3_conv(f2), size=f3.shape[2:], mode='bilinear', align_corners=True)
        f3_3 = F.interpolate(self.f3_3_conv(f3), size=f3.shape[2:], mode='bilinear', align_corners=True)
        f4_3 = F.interpolate(self.f4_3_conv(f4), size=f3.shape[2:], mode='bilinear', align_corners=True)
        f5_3 = F.interpolate(self.f5_3_conv(f5), size=f3.shape[2:], mode='bilinear', align_corners=True)
        # stage 4 feature
        f1_4 = F.interpolate(self.f1_4_conv(f1), size=f4.shape[2:], mode='bilinear', align_corners=True)
        f2_4 = F.interpolate(self.f2_4_conv(f2), size=f4.shape[2:], mode='bilinear', align_corners=True)
        f3_4 = F.interpolate(self.f3_4_conv(f3), size=f4.shape[2:], mode='bilinear', align_corners=True)
        f4_4 = F.interpolate(self.f4_4_conv(f4), size=f4.shape[2:], mode='bilinear', align_corners=True)
        f5_4 = F.interpolate(self.f5_4_conv(f5), size=f4.shape[2:], mode='bilinear', align_corners=True)
        # stage 5 feature
        f1_5 = F.interpolate(self.f1_5_conv(f1), size=f5.shape[2:], mode='bilinear', align_corners=True)
        f2_5 = F.interpolate(self.f2_5_conv(f2), size=f5.shape[2:], mode='bilinear', align_corners=True)
        f3_5 = F.interpolate(self.f3_5_conv(f3), size=f5.shape[2:], mode='bilinear', align_corners=True)
        f4_5 = F.interpolate(self.f4_5_conv(f4), size=f5.shape[2:], mode='bilinear', align_corners=True)
        f5_5 = F.interpolate(self.f5_5_conv(f5), size=f5.shape[2:], mode='bilinear', align_corners=True)
        ''' Stage Selection '''
        f2_1, f3_1, f4_1, f5_1 = self.wf1(f2_1, f3_1, f4_1, f5_1)
        f1_2, f3_2, f4_2, f5_2 = self.wf2(f1_2, f3_2, f4_2, f5_2)
        f1_3, f2_3, f4_3, f5_3 = self.wf3(f1_3, f2_3, f4_3, f5_3)
        f1_4, f2_4, f3_4, f5_4 = self.wf4(f1_4, f2_4, f3_4, f5_4)
        f1_5, f2_5, f3_5, f4_5 = self.wf5(f1_5, f2_5, f3_5, f4_5)

        # aggregation
        f1 = torch.cat((f1_1, f2_1, f3_1, f4_1, f5_1), dim=1)
        f2 = torch.cat((f1_2, f2_2, f3_2, f4_2, f5_2), dim=1)
        f3 = torch.cat((f1_3, f2_3, f3_3, f4_3, f5_3), dim=1)
        f4 = torch.cat((f1_4, f2_4, f3_4, f4_4, f5_4), dim=1)
        f5 = torch.cat((f1_5, f2_5, f3_5, f4_5, f5_5), dim=1)

        return f1, f2, f3, f4, f5


class GuidedAttention(nn.Module):
    def __init__(self, in_channels):
        super(GuidedAttention, self).__init__()

        self.in_channel = in_channels
        self.trans1 = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.trans2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, f):
        """
        x: main feature
        f: guided feature
        """
        # spatial attention
        sa = self.sigmoid(self.trans1(f)) * x
        # channel attention
        ca = self.softmax(self.trans2(self.gap(sa))) * self.in_channel * sa
        return ca + x


class EBM(nn.Module):
    """
    Explicit Boundary Modeling Module
    """

    def __init__(self, in_channels, out_channels):
        super(EBM, self).__init__()

        self.transition = ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.conv = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=3, padding=1)
        # foreground branch
        self.fam = nn.Sequential(
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )
        self.fga = GuidedAttention(in_channels=out_channels)
        # background branch
        self.bam = nn.Sequential(
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            ConvBNReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        )
        self.bga = GuidedAttention(in_channels=out_channels)
        # edge
        self.eam = ConvBNReLU(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.reconv = nn.Conv2d(in_channels=out_channels, out_channels=1, kernel_size=3, padding=1)
        # background guided attention
        self.ga = GuidedAttention(in_channels=out_channels)

    def forward(self, x):
        x = self.transition(x)
        # coarse map
        cmap = self.conv(x)
        # foreground branch
        fx = x * torch.sigmoid(cmap)
        fx = self.fam(fx)
        # background branch
        bx = x * (1 - torch.sigmoid(cmap))
        bx = self.bam(bx)
        # edge
        ef = fx * bx
        ef = self.eam(ef)
        emap = self.reconv(ef)
        # foreground branch
        fx = self.fga(fx, ef)
        # background branch
        bx = self.bga(bx, ef)
        # output
        out = self.ga(fx, bx)
        return out, cmap, emap


class SAENet(nn.Module):

    def __init__(self, channel=64):
        super(SAENet, self).__init__()

        self.encoder = pvt_v2_b4(pretrain=True)
        self.ppm = PPM(in_channels=channel, out_channels=channel)
        # reduction
        self.re_conv1 = ConvBNReLU(in_channels=64, out_channels=channel, kernel_size=3)
        self.re_conv2 = ConvBNReLU(in_channels=128, out_channels=channel, kernel_size=3)
        self.re_conv3 = ConvBNReLU(in_channels=320, out_channels=channel, kernel_size=3)
        self.re_conv4 = ConvBNReLU(in_channels=512, out_channels=channel, kernel_size=3)
        self.re_conv5 = ConvBNReLU(in_channels=channel * 3, out_channels=channel, kernel_size=3)
        # Weighted Dense Semantic Aggregation
        self.wdsa = WDSA(channel=channel)
        # Detail Enhanced Multi-Scale Extraction
        self.dems1 = DEMS(channel)
        self.dems2 = DEMS(channel)
        self.dems3 = DEMS(channel)
        self.dems4 = DEMS(channel)
        self.dems5 = DEMS(channel)
        # edge enhancement
        self.ebm1 = EBM(in_channels=2 * channel, out_channels=channel)
        self.ebm2 = EBM(in_channels=2 * channel, out_channels=channel)
        self.ebm3 = EBM(in_channels=2 * channel, out_channels=channel)
        self.ebm4 = EBM(in_channels=2 * channel, out_channels=channel)
        self.ebm5 = EBM(in_channels=channel, out_channels=channel)
        # out conv
        self.out_conv = nn.Conv2d(channel, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        # channel reduction to 64
        x1 = self.re_conv1(x1)
        x2 = self.re_conv2(x2)
        x3 = self.re_conv3(x3)
        x4 = self.re_conv4(x4)
        x5 = self.re_conv5(torch.cat((F.interpolate(x2, size=x4.shape[2:], mode='bilinear', align_corners=True),
                                      F.interpolate(x3, size=x4.shape[2:], mode='bilinear', align_corners=True),
                                      x4), dim=1))
        x5 = self.ppm(x5)

        # multi-level multi-scale feature extraction
        f1, f2, f3, f4, f5 = self.wdsa(x1, x2, x3, x4, x5)
        f1 = self.dems1(f1)
        f2 = self.dems2(f2)
        f3 = self.dems3(f3)
        f4 = self.dems4(f4)
        f5 = self.dems5(f5)

        # edge enhancement
        F5, cmap5, emap5 = self.ebm5(f5)

        f4 = torch.cat((f4, F5), dim=1)
        F4, cmap4, emap4 = self.ebm4(f4)

        F4_up = F.interpolate(F4, size=f3.shape[2:], mode='bilinear', align_corners=True)
        f3 = torch.cat((f3, F4_up), dim=1)
        F3, cmap3, emap3 = self.ebm3(f3)

        F3_up = F.interpolate(F3, size=f2.shape[2:], mode='bilinear', align_corners=True)
        f2 = torch.cat((f2, F3_up), dim=1)
        F2, cmap2, emap2 = self.ebm2(f2)

        F2_up = F.interpolate(F2, size=f1.shape[2:], mode='bilinear', align_corners=True)
        f1 = torch.cat((f1, F2_up), dim=1)
        F1, cmap1, emap1 = self.ebm1(f1)

        out = self.out_conv(F1)

        # upsampling coarse maps and edge maps to gt size
        size = (out.shape[2] * 4, out.shape[3] * 4)
        out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        cmap1 = F.interpolate(cmap1, size=size, mode='bilinear', align_corners=True)
        cmap2 = F.interpolate(cmap2, size=size, mode='bilinear', align_corners=True)
        cmap3 = F.interpolate(cmap3, size=size, mode='bilinear', align_corners=True)
        cmap4 = F.interpolate(cmap4, size=size, mode='bilinear', align_corners=True)
        cmap5 = F.interpolate(cmap5, size=size, mode='bilinear', align_corners=True)
        emap1 = F.interpolate(emap1, size=size, mode='bilinear', align_corners=True)
        emap2 = F.interpolate(emap2, size=size, mode='bilinear', align_corners=True)
        emap3 = F.interpolate(emap3, size=size, mode='bilinear', align_corners=True)
        emap4 = F.interpolate(emap4, size=size, mode='bilinear', align_corners=True)
        emap5 = F.interpolate(emap5, size=size, mode='bilinear', align_corners=True)

        return out, cmap1, cmap2, cmap3, cmap4, cmap5, emap1, emap2, emap3, emap4, emap5


if __name__ == '__main__':
    model = SAENet(channel=64)
    model.eval()

    in_tensor = torch.randn(size=(2, 3, 384, 384))
    outs = model(in_tensor)
    for i in outs:
        print(i.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pvtv2_new import pvt_v2_b2
from networks.pvtv2_new import pvt_v2_encoder
from einops import rearrange


class Curvature(torch.nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1 / 16, 5 / 16, -1 / 16], [5 / 16, -1, 5 / 16], [-1 / 16, 5 / 16, -1 / 16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio

    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B * C, 1, H, W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p = p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio * C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)

        return selected


class Doubleconv(nn.Module):
    def __init__(self, in_chan, out_chan, mid_channels=None):
        super(Doubleconv, self).__init__()
        if mid_channels is None:
            mid_channels = out_chan
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.conv1 = nn.Conv2d(in_channels=in_chan, out_channels=mid_channels, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=out_chan, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_chan)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class Up(nn.Module):
    def __init__(self, in_chan, out_chan, bilinear=True):
        super(Up, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Doubleconv(in_chan + out_chan, out_chan, out_chan)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_chan, out_channels=out_chan, kernel_size=2, stride=2)
            self.conv = Doubleconv(out_chan * 2, out_chan)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class Fusion_layer(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2, x3):
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x1_0 = self.conv1(x1_0)

        x2_0 = x1 * x3
        x2_0 = torch.cat((x2_0, x1), dim=1)
        x2_0 = self.conv2(x2_0)

        x = torch.cat((x1_0, x2_0), dim=1)
        x = self.conv3(x)

        x3_0 = x1 * x2 * x3
        x = torch.cat((x3_0, x), dim=1)
        x = self.conv4(x)

        return x

class Fusion_layer_ET(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer_ET, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2):
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x = self.conv1(x1_0)

        return x

class Fusion_layer_ET0(nn.Module):
    def __init__(self, dim):
        super(Fusion_layer_ET0, self).__init__()
        self.dim = dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim * 3, dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )


    def forward(self, x1, x2):
        x1_0 = x1 * x2
        x1_0 = torch.cat((x1_0, x1), dim=1)
        x = self.conv1(x1_0)


        return x

class Final_Up(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(Final_Up, self).__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim * dim_scale ** 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, C, H, W
        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C)
        x = self.expand(x)
        x = x.view(B, H, W, -1)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class multiPI_TransBTS(nn.Module):
    def __init__(self, ratio):
        super(multiPI_TransBTS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1)
        self.backbone0 = pvt_v2_encoder()  # [64, 128, 320, 512]
        self.backbone1 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone2 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone3 = pvt_v2_b2()  # [64, 128, 320, 512]
        self.backbone4 = pvt_v2_b2()  # [64, 128, 320, 512]

        path = './networks/pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone1.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone0.load_state_dict(model_dict, False)
        self.backbone1.load_state_dict(model_dict)
        self.backbone2.load_state_dict(model_dict)
        self.backbone3.load_state_dict(model_dict)
        self.backbone4.load_state_dict(model_dict)

        self.up11 = Up(512, 320, False)
        self.up12 = Up(320, 128, False)
        self.up13 = Up(128, 64, False)
        self.up21 = Up(512, 320, False)
        self.up22 = Up(320, 128, False)
        self.up23 = Up(128, 64, False)

        self.up01 = Up(512, 320, False)
        self.up02 = Up(320, 128, False)
        self.up03 = Up(128, 64, False)
        self.fusion_layer1 = Fusion_layer(320)
        self.fusion_layer2 = Fusion_layer(128)
        self.fusion_layer3 = Fusion_layer(64)
        self.fusion_layer1_ET = Fusion_layer_ET(320)
        self.fusion_layer2_ET = Fusion_layer_ET(128)
        # self.fusion_layer3_ET = Fusion_layer_ET(64)
        self.fusion_layer3_ET0 = Fusion_layer_ET0(64)
        self.up0 = Final_Up((4, 4), 64, 4)
        self.up1 = Final_Up((4, 4), 64, 4)
        self.up2 = Final_Up((4, 4), 64, 4)

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )
        self.out1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )
        self.out2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )
        self.ratio = ratio
        self.ife2 = Curvature(self.ratio[0])

    def forward(self, x):
        # backbone
        x1 = x[:, 0, None, :, :]
        x2 = x[:, 1, None, :, :]
        x3 = x[:, 2, None, :, :]
        x4 = x[:, 3, None, :, :]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv1(x3)
        x4 = self.conv2(x4)

        pvt1 = self.backbone1(x1)  # Flair
        pvt2 = self.backbone2(x2)  # T1
        pvt3 = self.backbone3(x3)  # T1ce
        pvt4 = self.backbone4(x4)  # T2


        # -- stage encoder fusion
        x_WT,x_TC,x_ET = self.backbone0(pvt1, pvt2, pvt3, pvt4)

        # WT

        x0 = self.fusion_layer1(x_WT[2], pvt1[2], pvt4[2])
        logits = self.up01(x_WT[3], x0)
        x0 = self.fusion_layer2(x_WT[1], pvt1[1], pvt4[1])
        logits = self.up02(logits, x0)
        x0 = self.fusion_layer3(x_WT[0], pvt1[0], pvt4[0])
        logits = self.up03(logits, x0)

        logits = self.up0(logits)
        logits_0 = self.out(logits)

        # TC

        x0 = self.fusion_layer1(x_TC[2], pvt3[2], pvt4[2])
        logits = self.up01(x_TC[3], x0)
        x0 = self.fusion_layer2(x_TC[1], pvt3[1], pvt4[1])
        logits = self.up02(logits, x0)
        x0 = self.fusion_layer3(x_TC[0], pvt3[0], pvt4[0])
        logits = self.up03(logits, x0)

        logits = self.up0(logits)
        logits_1 = self.out(logits)


        # ET

        x0 = self.fusion_layer1_ET(x_ET[2], pvt3[2])
        logits = self.up01(x_ET[3], x0)
        x0 = self.fusion_layer2_ET(x_ET[1], pvt3[1])
        logits = self.up02(logits, x0)

        x_ET[0] = torch.cat([x_ET[0], self.ife2(x_ET[0])], dim=1)
        pvt3[0] = torch.cat([pvt3[0], self.ife2(pvt3[0])], dim=1)

        x0 = self.fusion_layer3_ET0(x_ET[0], pvt3[0])
        logits = self.up03(logits, x0)

        logits = self.up0(logits)
        logits_2 = self.out(logits)

        logits = torch.cat([logits_0, logits_1, logits_2], dim=1)


        return logits, logits_0, logits_1, logits_2

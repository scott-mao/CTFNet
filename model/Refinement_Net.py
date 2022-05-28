import torch
import torch.nn as nn




def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=(kernel_size,kernel_size),
        padding=(kernel_size // 2), bias=bias, stride=(stride,stride))



#multi-level Feature Fusion block
class MFFB(nn.Module):
    def __init__(self, in_channels,height =3,  reduction=8, bias=False):
        super(MFFB, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, d, (1,1), padding=0, bias=bias),
            nn.ReLU(),
        )

        self.fc_a = nn.Sequential(
            nn.Conv2d(d, in_channels, kernel_size=(1,1), stride=(1,1), bias=bias),
            nn.ReLU(),
        )
        self.fc_b = nn.Sequential(
            nn.Conv2d(d, in_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            nn.ReLU(),
        )
        self.fc_c = nn.Sequential(
            nn.Conv2d(d, in_channels, kernel_size=(1, 1), stride=(1, 1), bias=bias),
            nn.ReLU(),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats_a,inp_feats_b,inp_feats_c):
        batch_size = inp_feats_a.shape[0]
        n_feats = inp_feats_a.shape[1]

        inp_feats = torch.cat((inp_feats_a,inp_feats_b,inp_feats_c), dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_a = torch.sum(inp_feats, dim=1)
        # print(feats_a.shape)
        feats_u = self.avg_pool(feats_a)
        feats_z = self.conv_du(feats_u)
        attention_vector_a = self.fc_a(feats_z)
        attention_vector_b = self.fc_b(feats_z)
        attention_vector_c = self.fc_c(feats_z)
        attention_vectors = torch.cat((attention_vector_a,attention_vector_b,attention_vector_c), dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=(stride,stride), padding=padding,
                              dilation=(dilation,dilation), groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale


##Channel Attention (CA)
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, (1,1), padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, (1,1), padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##Dual Attention(DAB)
class DAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.ReLU(), res_scale=1):
        super(DAB, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)
        self.SA = spatial_attn_layer()
        self.CA = ca_layer(n_feat, reduction, bias=bias)
        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=(1,1), bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


##down/up sampling
class DownSample(nn.Module):
    def __init__(self, in_channels, stride,bias=False):
        super(DownSample, self).__init__()

        self.down = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1,1), stride=(1,1), padding=0, bias=bias),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, in_channels, (3,3), stride=(1,1), padding=1, bias=bias),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, in_channels, (4,4), stride=(stride,stride), padding=((4-stride)//2,(4-stride)//2), bias=bias),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, in_channels * stride, (1,1), stride=(1,1), padding=0, bias=bias))

    def forward(self, x):
        out = self.down(x)

        return out




class UpSample(nn.Module):
    def __init__(self, in_channels, stride,bias=False):
        super(UpSample, self).__init__()

        self.up1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1,1), stride=(1,1), padding=0, bias=bias),
                                 nn.ReLU(),
                                 nn.ConvTranspose2d(in_channels, in_channels, (3,3), stride=(stride,stride), padding=((4-stride)//2,(4-stride)//2), output_padding=(1,1),
                                                    bias=bias),
                                 nn.ReLU(),
                                 nn.Conv2d(in_channels, in_channels // stride, (1,1), stride=(1,1), padding=0, bias=bias))

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=stride, mode='bilinear', align_corners=bias),
                                 nn.Conv2d(in_channels, in_channels // stride, (1,1), stride=(1,1), padding=0, bias=bias))

    def forward(self, x):
        up1 = self.up1(x)
        up2 = self.up2(x)
        out = up1 + up2
        return out


##########################################################################
##---------- Feature Fusion Detail Enhancement(FFDE) ----------
class FFDE(nn.Module):
    def __init__(self, n_feat, height, width, bias):
        super(FFDE, self).__init__()
        self.n_feat = n_feat
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=(3,3),padding=(1,1), bias=bias, stride=(1,1))
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=(3, 3), padding=(1, 1), bias=bias, stride=(1, 1))
        self.DAB_1 = DAB(n_feat=n_feat)
        self.DAB_2 = DAB(n_feat = n_feat*2)
        self.DAB_3 = DAB(n_feat = n_feat*4)
        self.fusion_1 = MFFB(in_channels=n_feat)
        self.fusion_2 = MFFB(in_channels=n_feat*2)
        self.fusion_3 = MFFB(in_channels=n_feat*4)
        self.fusion = MFFB(in_channels=n_feat)
        self.down2 = DownSample(self.n_feat, stride = 2)
        self.down2_DAB2 = DownSample(self.n_feat*2,  stride=2)
        self.down4 = DownSample(self.n_feat, stride=4)
        self.down2_DAB4 = DownSample(self.n_feat * 4,  stride=2)
        self.up2 = UpSample(int(self.n_feat*2),stride=2)
        self.up2_DAB3 = UpSample(int(self.n_feat*4),stride=2)
        self.up4 = UpSample(int(self.n_feat * 4),  stride=4)

    def forward(self, x):
        x1 = self.conv1(x)
        x1_down2 = self.down2(x1)
        x1_down4 = self.down4(x1)
        DAB1 = self.DAB_1(x1)
        DAB2 = self.DAB_2(x1_down2)
        DAB3 = self.DAB_3(x1_down4)
        DAB1_d4 = self.down4(DAB1)
        DAB1_d2 = self.down2(DAB1)
        DAB2_d2 = self.down2_DAB2(DAB2)
        DAB2_u2 = self.up2(DAB2)
        DAB3_u2 = self.up2_DAB3(DAB3)
        DAB3_u4 = self.up4(DAB3)
        MF_DAB1 = self.fusion_1(DAB1,DAB2_u2,DAB3_u4)
        MF_DAB2 = self.fusion_2(DAB1_d2, DAB2, DAB3_u2)
        MF_DAB3 = self.fusion_3(DAB1_d4, DAB2_d2, DAB3)
        MF_DAB2_up2 = self.up2(MF_DAB2)
        MF_DAB3_up4 = self.up4(MF_DAB3)
        fusion_DAB = self.fusion(MF_DAB1,MF_DAB2_up2,MF_DAB3_up4)
        out = self.conv2(fusion_DAB)
        out+=x
        return out








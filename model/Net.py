import torch
import torch.nn as nn
import conf
from model.Refinement_Net import FFDE
import torch.nn.functional as F
contact_channel = conf.contact_channel
eps = conf.eps

class FineNet(nn.Module):
    def __init__(self, in_channels=6, out_channels=3, n_feat=64, kernel_size=3,  n_FFDE=3, height=3,
                 width=2, bias=False):
        super(FineNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=(kernel_size,kernel_size), padding=(kernel_size - 1) // 2,
                                 bias=bias)
        self.conv_in2 = nn.Conv2d(n_feat, n_feat, kernel_size=(kernel_size, kernel_size),
                                 padding=(kernel_size - 1) // 2,
                                 bias=bias)
        modules_body = [FFDE(n_feat, height, width, bias) for _ in range(n_FFDE)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out2 = nn.Conv2d(n_feat, out_channels, kernel_size=(kernel_size, kernel_size),
                                  padding=(kernel_size - 1) // 2,
                                  bias=bias)
        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=(kernel_size,kernel_size), padding=(kernel_size - 1) // 2,
                                  bias=bias)

    def forward(self, x,x_coarse):
        x_coarse.detach()
        x = torch.cat((x,x_coarse),dim=1)
        h = self.ReLU(self.conv_in(x))
        h = self.ReLU(self.conv_in2(h))
        h = self.body(h)
        h = self.ReLU(self.conv_out(h))
        h = self.ReLU(self.conv_out2(h))
        h += x_coarse
        return h


class CoarseNet(nn.Module):
    def __init__(self, rgb_channel = 3,channels = 16,illu_channel=1):
        super(CoarseNet, self).__init__()
        self.noise_net = nn.Sequential(
            nn.Conv2d(rgb_channel, channels,  kernel_size=(3,3),stride=(1,1), padding=1,),
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, (3,3) ,(1,1), 1),
            nn.ReLU(),
            nn.Conv2d(channels*2, channels, (3,3) ,(1,1), 1),
            nn.ReLU(),
            nn.Conv2d(channels, rgb_channel, (3,3) ,(1,1), 1),
        )
        self.illu_net = nn.Sequential(
            nn.Conv2d(illu_channel, channels, (3,3) ,(1,1), 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels*2, (3,3) ,(1,1), 1),
            nn.ReLU(),
            nn.Conv2d(channels*2, channels, (3,3) ,(1,1), 1),
            nn.ReLU(),
            nn.Conv2d(channels, illu_channel, (3,3) ,(1,1), 1),
        )


    def forward(self,img_tensor):

        illu_in = torch.max(img_tensor,dim=1)[0].unsqueeze(1)
        illu = torch.sigmoid(self.illu_net(illu_in))
        noise = torch.tanh(self.noise_net(img_tensor))
        I_res = (img_tensor-noise)/(illu+eps)
        I_res = torch.clamp(I_res,min=0,max=1)
        return  I_res,illu,noise
#



class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h0 = F.relu(self.conv1_1(X), inplace=True)
        h1 = F.relu(self.conv1_2(h0), inplace=True)
        h2 = F.max_pool2d(h1, kernel_size=2, stride=2)
        h3 = F.relu(self.conv2_1(h2), inplace=True)
        h4 = F.relu(self.conv2_2(h3), inplace=True)
        h5 = F.max_pool2d(h4, kernel_size=2, stride=2)
        h6 = F.relu(self.conv3_1(h5), inplace=True)
        h7 = F.relu(self.conv3_2(h6), inplace=True)
        h8 = F.relu(self.conv3_3(h7), inplace=True)
        h9 = F.max_pool2d(h8, kernel_size=2, stride=2)
        h10 = F.relu(self.conv4_1(h9), inplace=True)
        h11 = F.relu(self.conv4_2(h10), inplace=True)
        conv4_3 = self.conv4_3(h11)
        result = F.relu(conv4_3, inplace=True)

        return result

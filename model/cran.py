# Unofficial implementations of CRAN, from Ast-363 github.
# Some problems.
import imp
from model import common

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# Using CDRR would result in disaster

def make_model(args, parent=False):
    return RCAN(args)

class Conv2d_CG(nn.Conv2d):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
        super(Conv2d_CG, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # just uncomment this region if you want to use CDRR
        # weights for gate convolution network (context descriptor relationshop reasoning)
        self.weight_g = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        self.weight_r = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_g)
        nn.init.kaiming_normal_(self.weight_r)

        # weight for affinity matrix
        self.weight_affinity_1 = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        self.weight_affinity_2 = Parameter(torch.zeros(in_channels, in_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_affinity_1)
        nn.init.kaiming_normal_(self.weight_affinity_2)
        
        # weight & bias for content-gated-convolution
        self.weight_conv = Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=True).cuda()
        self.bias_conv = Parameter(torch.zeros(out_channels), requires_grad=True)
        nn.init.kaiming_normal_(self.weight_conv)

        self.stride = stride
        self.padding= padding
        self.dilation = dilation
        self.groups = groups

        # for convolutional layers with a kernel size  of 1, just use traditional convolution
        if kernel_size == 1:
            self.ind = True
        else:
            self.ind = False
            self.oc = out_channels
            self.ks = kernel_size

            # target spatial size of the pooling layer
            ws = kernel_size
            self.avg_pool = nn.AdaptiveAvgPool2d((ws, ws))

            # the dimension of latent representation
            self.num_lat = int((kernel_size * kernel_size)/2 + 1)

            # the context encoding module
            self.ce = nn.Linear(ws*ws, self.num_lat, False)
            self.ce_bn = nn.BatchNorm1d(in_channels)
            self.ci_bn2 = nn.BatchNorm1d(in_channels)

            self.act = nn.ReLU(inplace=True)

            # the number of groups in the channel interaction module
            if in_channels // 16:
                self.g = 16
            else:
                self.g = in_channels
            
            # the channel interacting module
            self.ci = nn.Linear(self.g, out_channels // (in_channels // self.g), bias=False)
            self.ci_bn = nn.BatchNorm1d(out_channels)

            # the gate decoding module (spatial interaction)
            self.gd = nn.Linear(self.num_lat, kernel_size*kernel_size, False)
            self.gd2 = nn.Linear(self.num_lat, kernel_size*kernel_size, False)

            # used to prepare the input feature map to patches
            self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

            # sigmoid function
            self.sig = nn.Sigmoid()

    def forward(self, x):
        # for convolutional layers with a kernel size of 1, just use the traditional convolution
        if self.ind:
            return F.conv2d(x, self.weight_conv, self.bias_conv, self.stride, self.padding, self.dilation, self.groups)
        else:
            b, c, h, w = x.size()                   # x: batch x n_feat(=64) x h_patch x w_patch
            weight = self.weight_conv

            # allocate global information
            gl = self.avg_pool(x).view(b, c, -1)    # gl: batch x n_feat x 3 x 3 -> batch x n_feat x 9

            # context-encoding module
            out = self.ce(gl)                       # out: batch x n_feat x 5


            # just uncomment this region if you want to use CDRR
            # Conext Descriptor Relationship Reasoning
            weighted_out_1 = torch.matmul(self.weight_affinity_1, out)                      # weighted_out: batch x n_feat x 5
            weighted_out_2 = torch.matmul(self.weight_affinity_2, out)
            affinity = torch.bmm(weighted_out_1.permute(0, 2, 1), weighted_out_2)           # affinity: batch x 5 x 5
            out_1 = torch.matmul(affinity, out.permute(0, 2, 1))                        # out_1: batch x 5 x n_feat
            out_2 = torch.matmul(out_1, self.weight_g)                                  # out_2: batch x 5 x n_feat
            out_3 = torch.matmul(out_2, self.weight_r)                                  # out_3: batch x 5 x n_feat
            out_4 = out_3.permute(0, 2, 1)                                              # out_4: batch x n_feat x 5
            out_5 = torch.mul(out_4, out)                                               # out_5: batch x n_feat x 5
            out = out + out_5                                                                # out: batch x n_feat x 5


            # use different bn for following two branches
            ce2 = out                               # ce2: batch x n_feat x 5
            out = self.ce_bn(out)
            out = self.act(out)                     # out: batch x n_feat x 5 (just batch normalization)

            # gate decoding branch 1 (spatial interaction)
            out = self.gd(out)                      # out: batch x n_feat x 9 (5 --> 9 = 3x3)

            # channel interacting module
            if self.g > 3:
                oc = self.ci(self.act(self.ci_bn2(ce2).view(b, c//self.g, self.g, -1).transpose(2,3))).transpose(2,3).contiguous()
            else:
                oc = self.ci(self.act(self.ci_bn2(ce2).transpose(2,1))).transpose(2,1).contiguous() 
            oc = oc.view(b, self.oc, -1)
            oc = self.ci_bn(oc)
            oc = self.act(oc)                       # oc: batch x n_feat x 5 (after grouped linear layer)

            # gate decoding branch 2 (spatial interaction)
            oc = self.gd2(oc)                       # oc: batch x n_feat x 9 (5 --> 9 = 3x3)
            
            # produce gate (equation (4) in the CRAN paper)
            out = self.sig(out.view(b, 1, c, self.ks, self.ks) + oc.view(b, self.oc, 1, self.ks, self.ks))  # out: batch x out_channel x in_channel x kernel_size x kernel_size (same dimension as conv2d weight)

            # unfolding input feature map to patches
            x_un = self.unfold(x)
            b, _, l = x_un.size()

            # gating
            out = (out * weight.unsqueeze(0)).view(b, self.oc, -1)

            # currently only handle square input and output
            #return torch.matmul(out, x_un).view(b, self.oc, int(np.sqrt(l)), int(np.sqrt(l)))
            out = torch.matmul(out, x_un)
            return out.view(b, self.oc, h, w)
## Context Reasoning Attention Block (CRAB)
class CRABLayer(nn.Module):
    def __init__(self, n_feat, res_scale=1):
        super(CRABLayer, self).__init__()
        self.res_scale = res_scale

        self.conv_CRAC = nn.Sequential(
            Conv2d_CG(n_feat, n_feat),
            nn.ReLU(inplace=True),
            Conv2d_CG(n_feat, n_feat)
        )
    
    def forward(self, x):
        res = self.conv_CRAC(x).mul(self.res_scale)
        res += x
        return res


## Residual Block (RB)
class RB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1):
    
        super(RB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CRABLayer(n_feat))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale
    
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RB(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(inplace=True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        if len(args.scale) == 1:
            scale = args.scale[0]
        else:
            scale = args.scale
        act = nn.ReLU(inplace=True)
        
        # RGB mean for DIV2K
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))

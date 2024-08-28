# -*- coding:utf-8 -*-
import torch
import math
import torch.nn as nn

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
    return init_fun

class Evolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        self.input_conv.apply(weights_init('kaiming'))
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False
    def forward(self, input, mask):
        output = self.input_conv(input)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)
        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)
        return output, new_mask
class EVOActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True,  activ='relu',
                 conv_bias=False):
        super().__init__()
        self.conv = Evolution(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, mask):
        h, h_mask = self.conv(input, mask)
        if hasattr(self, 'bn'):
            h = self.bn(h)
        if hasattr(self, 'activation'):
            h = self.activation(h)
        return h, h_mask


class STNET(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.evolution = EVOActiv(1, 1, activ='leaky')
        self.bn = nn.BatchNorm2d(1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.t = args.t

    def forward(self,input, mask):
        size = input.size()
        output1, mask1 = self.evolution(input, mask)
        output1 = output1 * mask1


        output2 = input.view(self.t, -1,  size[1], size[2] * size[3]).permute(3, 2, 0, 1)
        mask2 = mask.view(self.t, -1,  size[1], size[2] * size[3]).permute(3, 2, 0, 1)
        output3, mask3 = self.evolution(output2, mask2)
        output3 = output3 * mask3
        output4 = output3.view(size[2], size[3], size[1], -1).permute(3, 2, 0, 1)
        mask4 = mask3.view(size[2], size[3], size[1], -1).permute(3, 2, 0, 1)
        output5 = torch.cat((output1, output4), dim=1)
        output5 = self.conv2(output5)
        output5 = self.conv1(output5)
        output5 = self.bn(output5)
        output5 = self.activation(output5)
        new_mask = mask1 + mask4
        no_update_holes = new_mask.int() == 0
        mask6 = torch.ones_like(new_mask)
        mask6 = mask6.masked_fill_(no_update_holes, 0.0)

        return output5, mask6




class STEINet(torch.nn.Module):
    def __init__(self, args):
        super(STEINet,self).__init__()
        self.stnet = STNET(args)

    def forward(self,input, mask):
        number = 0
        output_result = input
        output_mask = mask
        while 0 in output_mask.int():
            output_result, output_mask = self.stnet(output_result, output_mask)
            output_result = output_result * output_mask
            number = number+1

        return output_result, number


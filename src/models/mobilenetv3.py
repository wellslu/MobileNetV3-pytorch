import mlconfig
import math
from torch import nn
import torch.nn.functional as F
import torch

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
    
class ConvBNHSwish(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            h_swish()
        ]
        super(ConvBNHSwish, self).__init__(*layers)

class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, _make_divisible(in_size // reduction, 8), kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(_make_divisible(in_size // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(_make_divisible(in_size // reduction, 8), in_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_size),
            h_sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

# class SeModule(nn.Module):
#     def __init__(self, channel, reduction=4):
#         super(SeModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#                 nn.Linear(channel, _make_divisible(channel // reduction, 8)),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(_make_divisible(channel // reduction, 8), channel),
#                 h_sigmoid()
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y

class InvertedResidual(nn.Module):
    def __init__(self, inp, expand_size, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(inp, expand_size, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(expand_size),
            h_swish() if use_hs else nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, expand_size, kernel_size, stride, kernel_size//2, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            SeModule(expand_size) if use_se else nn.Identity(), 
            h_swish() if use_hs else nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(oup),
            h_swish() if use_hs else nn.ReLU(inplace=True)
        )
        
        self.shortcut = False
        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(oup)
                                          )

    def forward(self, x):
        out = self.conv(x)
        out = out + self.shortcut(x) if self.shortcut else out
        return out
        
@mlconfig.register
class MobileNetV3(nn.Module):

    def __init__(self, num_classes=2, shallow=False):
        super(MobileNetV3, self).__init__()
        
        self.features = nn.Sequential(*self.get_layers())
        self.avg_pool = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280), 
            nn.BatchNorm1d(1280), 
            h_swish(), 
            nn.Linear(1280, num_classes), 
            nn.Softmax(dim=1)
        )
        self._initialize_weights()
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    @staticmethod
    def get_layers():
        cfgs = [
           # k,  ex,  oup,  SE, HS, s
            [3,   16,   16,  0,  0, 1],
            [3,   64,   24,  0,  0, 2],
            [3,   72,   24,  0,  0, 1],
            [5,   72,   40,  1,  0, 2],
            [5,  120,   40,  1,  0, 1],
            [5,  120,   40,  1,  0, 1],
            [3,  240,   80,  0,  1, 2],
            [3,  200,   80,  0,  1, 1],
            [3,  184,   80,  0,  1, 1],
            [3,  184,   80,  0,  1, 1],
            [3,  480,  112,  1,  1, 1],
            [3,  672,  112,  1,  1, 1],
            [5,  672,  160,  1,  1, 2],
            [5,  960,  160,  1,  1, 1],
            [5,  960,  160,  1,  1, 1],
            ]
#         cfgs = [
#            # k,  ex,  oup,  SE, HS, s
#             [3,   16,   16,  1,  0, 2],
#             [3,   72,   24,  0,  0, 2],
#             [3,   88,   24,  0,  0, 1],
#             [5,   96,   40,  1,  1, 2],
#             [5,  240,   40,  1,  1, 1],
#             [5,  240,   40,  1,  1, 1],
#             [5,  120,   48,  1,  1, 1],
#             [5,  144,   48,  1,  1, 1],
#             [5,  288,   96,  1,  1, 2],
#             [5,  576,   96,  1,  1, 1],
#             [5,  576,   96,  1,  1, 1],
#             ]
        
        layers = [ConvBNHSwish(3, 16, kernel_size=3, stride=2, padding=1)]
        input_channel = _make_divisible(16, 8)
        for kernal, exp_size, output_channel, use_se, use_hs, stride in cfgs:
            output_channel = _make_divisible(output_channel, 8)
            exp_size = _make_divisible(exp_size, 8)
            layers.append(InvertedResidual(input_channel, exp_size, output_channel, kernal, stride, use_se, use_hs))
            input_channel = output_channel
        layers.append(ConvBNHSwish(input_channel, 960, kernel_size=1, stride=1, padding=0))
        return layers
    
    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


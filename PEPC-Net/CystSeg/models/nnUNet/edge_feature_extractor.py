from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from .simple_conv_blocks import ConvDropoutNormReLU


class DualPathEdgeFeatureExtractor(nn.Module):
    def __init__(self, features, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super(DualPathEdgeFeatureExtractor, self).__init__()

        self.n_stages = len(features_per_stage)

        fuses = []

        for i in range(self.n_stages):
            fuses.append(SFGate(features, features_per_stage[i], conv_bias, norm_op,
                                norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
        self.fuses = nn.ModuleList(fuses)

        if self.n_stages % 2 == 0:
            self.bottom_up_index = [i * 2 for i in range(self.n_stages // 2)]
            self.top_down_index = [self.n_stages - 1 - i * 2 for i in range(self.n_stages // 2)]
            self.middle_index = None
        else:
            self.bottom_up_index = [i for i in range(self.n_stages // 2)]
            self.top_down_index = [self.n_stages - 1 - i for i in range(self.n_stages // 2)]
            self.middle_index = self.n_stages // 2

    def forward(self, x, skips):
        explicit_x_ret = []

        for i, j in zip(self.bottom_up_index, self.top_down_index):
            x = self.fuses[i](x, skips[i]) + self.fuses[j](x, skips[j])
            explicit_x_ret.append(x)

        if self.middle_index is not None:
            x = self.fuses[self.middle_index](x, skips[self.middle_index])
            explicit_x_ret.append(x)

        return explicit_x_ret


class SFGate(nn.Module):
    def __init__(self, in_channel_1: int, in_channel_2: int, conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None):
        super(SFGate, self).__init__()
        self.conv1 = ConvDropoutNormReLU(nn.Conv3d, in_channel_2, in_channel_1, 3, 1, conv_bias, norm_op,
                                         norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.conv2 = nn.Sequential(
            nn.Conv3d(2 * in_channel_1, 2, kernel_size=3, padding=1),
            nn.Softmax(dim=1))

    def forward(self, inputs1, inputs2):
        if inputs1.shape != inputs2.shape:
            up_inputs2 = nn.functional.interpolate(self.conv1(inputs2), size=inputs1.size()[2:])
        else:
            up_inputs2 = inputs2
        salient_map = self.conv2(torch.cat((inputs1, up_inputs2), dim=1))
        salient_map = torch.split(salient_map, split_size_or_sections=1, dim=1)
        out = inputs1 * salient_map[0] + up_inputs2 * salient_map[1]
        return out


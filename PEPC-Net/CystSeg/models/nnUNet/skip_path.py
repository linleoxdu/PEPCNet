from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd
from .simple_conv_blocks import ConvDropoutNormReLU


class ImplicitFeaturePath(nn.Module):
    def __init__(self,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        super().__init__()
        assert len(features_per_stage) == len(strides)
        self.n_stages = len(features_per_stage)

        transpconvs = []
        convs = []

        for i in range(1, len(features_per_stage)):
            layer = []
            layer.append(nn.ConvTranspose3d(features_per_stage[i], features_per_stage[i - 1],
                                                  strides[i], strides[i], bias=conv_bias))
            layer.append(nonlin(** nonlin_kwargs))
            transpconvs.append(nn.Sequential(*layer))
            convs.append(ConvDropoutNormReLU(nn.Conv3d, features_per_stage[i-1] * 2, features_per_stage[i-1],
                                             3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                             nonlin, nonlin_kwargs))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.convs = nn.ModuleList(convs)

    def forward(self, skips, middle_xs):
        implicit_x_ret = []
        for i in range(1, self.n_stages):
            implicit_x = self.convs[i - 1](torch.cat((skips[i - 1], skips[i - 1] - self.transpconvs[i - 1](middle_xs[i - 1])), dim=1))
            implicit_x_ret.append(implicit_x)
        return implicit_x_ret


class ImplicitFeaturePath2(nn.Module):
    def __init__(self,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 ):
        super().__init__()
        assert len(features_per_stage) == len(strides)
        self.n_stages = len(features_per_stage)

        transpconvs = []
        convs = []

        for i in range(1, len(features_per_stage)):
            layer = []
            layer.append(nn.ConvTranspose3d(features_per_stage[i-1], features_per_stage[i - 1],
                                                  strides[i], strides[i], bias=conv_bias))
            layer.append(nonlin(** nonlin_kwargs))
            transpconvs.append(nn.Sequential(*layer))
            convs.append(ConvDropoutNormReLU(nn.Conv3d, features_per_stage[i-1] * 2, features_per_stage[i-1],
                                             3, 1, conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                             nonlin, nonlin_kwargs))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.convs = nn.ModuleList(convs)

    def forward(self, skips, middle_xs):
        implicit_x_ret = []
        for i in range(1, self.n_stages):
            implicit_x = self.convs[i - 1](torch.cat((skips[i - 1], skips[i - 1] - self.transpconvs[i - 1](middle_xs[i - 1])), dim=1))
            implicit_x_ret.append(implicit_x)
        return implicit_x_ret

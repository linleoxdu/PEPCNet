from typing import Union, Type, List, Tuple
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from .plain_conv_encoder import ModifiedConvEncoder
from .unet_decoder import Decoder
from .helper import convert_conv_op_to_dim
from .skip_path import ImplicitFeaturePath
from .edge_feature_extractor import DualPathEdgeFeatureExtractor
from .simple_conv_blocks import ConvDropoutNormReLU


class PEPC_Net(nn.Module):
    def __init__(self,
                 edge_branch_channels: int,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.n_stages = n_stages
        self.encoder = ModifiedConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                           n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                           dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                           nonlin_first=nonlin_first)
        self.implicit_feature_path = ImplicitFeaturePath(features_per_stage, strides, conv_bias, norm_op,
                                                         norm_op_kwargs,
                                                         dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
        self.edge_conv = ConvDropoutNormReLU(nn.Conv3d, input_channels, edge_branch_channels, 3, 1, conv_bias,
                                             norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin,
                                             nonlin_kwargs)
        self.explicit_feature_path = DualPathEdgeFeatureExtractor(edge_branch_channels, features_per_stage, conv_bias,
                                                                  norm_op, norm_op_kwargs, dropout_op,
                                                                  dropout_op_kwargs,
                                                                  nonlin, nonlin_kwargs)
        edge_supervision = []
        if n_stages % 2 == 0:
            edge_supervision_nums = n_stages // 2
        else:
            edge_supervision_nums = n_stages // 2 + 1

        for i in range(edge_supervision_nums):
            edge_supervision.append(nn.Conv3d(edge_branch_channels, 1, 1))
        self.edge_supervision = nn.ModuleList(edge_supervision)

        converts = []
        for i in range(self.n_stages - 1):
            sub_strides = strides[: i + 1]
            stride = [1, 1, 1]
            for ss in sub_strides:
                stride[0] *= ss[0]
                stride[1] *= ss[1]
                stride[2] *= ss[2]

            converts.append(ConvDropoutNormReLU(nn.Conv3d, edge_branch_channels, features_per_stage[i], stride,
                                                stride, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                                dropout_op_kwargs, nonlin, nonlin_kwargs))
        self.converts = nn.ModuleList(converts)

        self.decoder = Decoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                               nonlin_first=nonlin_first, conv_bias=conv_bias, norm_op=norm_op,
                               norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs)

    def forward(self, x):
        skips, middle_xs = self.encoder(x)
        implicit_xs = self.implicit_feature_path(skips, middle_xs)
        explicit_xs = self.explicit_feature_path(self.edge_conv(x), skips)
        converted_explicit_xs = []
        for i in range(self.n_stages - 1):
            converted_explicit_xs.append(self.converts[i](explicit_xs[-1]))
        if self.training:
            return [self.edge_supervision[i](ex) for i, ex in enumerate(explicit_xs)], \
                   self.decoder(skips, implicit_xs, converted_explicit_xs)
        else:
            return self.decoder(skips, implicit_xs, converted_explicit_xs)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(
            self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                   "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                   "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size)

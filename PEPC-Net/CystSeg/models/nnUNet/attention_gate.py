import torch
from torch import nn
from torch.nn import functional as F
from .simple_conv_blocks import ConvDropoutNormReLU


class _AttentionGateND(nn.Module):
    def __init__(self, in_channels, gating_channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 inter_channels=None, dimension=3, mode='concatenation', sub_sample_factor=(2, 2, 2)):
        super(_AttentionGateND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        self.nonlin = nonlin(**nonlin_kwargs)

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if norm_op_kwargs is None:
            norm_op_kwargs = {}

        if dimension == 3:
            conv_nd = nn.Conv3d
            norm = norm_op(num_features=in_channels, **norm_op_kwargs)
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            norm = norm_op(num_features=in_channels, **norm_op_kwargs)
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            norm,
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=True)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        # f = F.relu(theta_x + phi_g, inplace=True)
        f = self.nonlin(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        # return W_y, sigm_psi_f
        return W_y

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class AttentionGate3D(_AttentionGateND):
    def __init__(self, in_channels, gating_channels, norm_op, norm_op_kwargs, nonlin, nonlin_kwargs,
                 inter_channels=None, mode='concatenation', sub_sample_factor=(2, 2, 2)):
        super(AttentionGate3D, self).__init__(in_channels,
                                              gating_channels,
                                              norm_op,
                                              norm_op_kwargs,
                                              nonlin,
                                              nonlin_kwargs,
                                              inter_channels,
                                              3,
                                              mode,
                                              sub_sample_factor,
                                              )


class ChannelAttention(nn.Module):
    def __init__(self, planes, nonlin, nonlin_kwargs):
        super(ChannelAttention, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.globalMaxPool = nn.AdaptiveMaxPool3d((1, 1, 1))

        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes)

        self.nonlin = nonlin(**nonlin_kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        # For global average pool
        out1 = self.globalAvgPool(x)
        out1 = out1.view(out1.size(0), -1)
        out1 = self.fc1(out1)
        out1 = self.nonlin(out1)
        out1 = self.fc2(out1)
        out1 = self.sigmoid(out1)
        out1 = out1.view(out1.size(0), out1.size(1), 1, 1, 1)
        out1 = out1 * x

        # For global maximum pool
        out2 = self.globalMaxPool(x)
        out2 = out2.view(out2.size(0), -1)
        out2 = self.fc1(out2)
        out2 = self.nonlin(out2)
        out2 = self.fc2(out2)
        out2 = self.sigmoid(out2)
        out2 = out2.view(out2.size(0), out2.size(1), 1, 1, 1)
        out2 = out2 * x

        out = out1 + out2
        out = residual + out
        out = self.nonlin(out)

        return out


class EdgeAwareBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                 dropout_op_kwargs, nonlin, nonlin_kwargs, nonlocal_mode, sub_sample_factor):
        super(EdgeAwareBlock, self).__init__()
        self.gate_1 = AttentionGate3D(in_channels=in_size, gating_channels=gate_size, norm_op=norm_op,
                                      norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                                      inter_channels=inter_size, mode=nonlocal_mode,
                                      sub_sample_factor=sub_sample_factor)
        self.gate_2 = AttentionGate3D(in_channels=in_size, gating_channels=gate_size, norm_op=norm_op,
                                      norm_op_kwargs=norm_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                                      inter_channels=inter_size, mode=nonlocal_mode,
                                      sub_sample_factor=sub_sample_factor)
        self.combine_1 = ChannelAttention(in_size * 2, nonlin, nonlin_kwargs)
        self.combine_2 = ConvDropoutNormReLU(nn.Conv3d, in_size * 2, in_size, 1, 1, conv_bias, norm_op, norm_op_kwargs,
                                             dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)

    def forward(self, inputs1, inputs2, gating_signal):
        gate_1 = self.gate_1(inputs1, gating_signal)
        gate_2 = self.gate_2(inputs2, gating_signal)

        return self.combine_2(self.combine_1(torch.cat([gate_1, gate_2], 1)))

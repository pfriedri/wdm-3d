from abc import abstractmethod

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import checkpoint, conv_nd, linear, avg_pool_nd, zero_module, normalization, timestep_embedding
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    A wavelet upsampling layer with an optional convolution on the skip connections used to perform upsampling.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.resample_2d = resample_2d

        self.use_freq = use_freq
        self.idwt = IDWT_3D("haar")

        # Grouped convolution on 7 high frequency subbands (skip connections)
        if use_conv:
            self.conv = conv_nd(dims, self.channels * 7, self.out_channels * 7, 3, padding=1, groups=7)

    def forward(self, x):
        if isinstance(x, tuple):
            skip = x[1]
            x = x[0]
        assert x.shape[1] == self.channels

        if self.use_conv:
            skip = self.conv(th.cat(skip, dim=1) / 3.) * 3.
            skip = tuple(th.chunk(skip, 7, dim=1))

        if self.use_freq:
            x = self.idwt(3. * x, skip[0], skip[1], skip[2], skip[3], skip[4], skip[5], skip[6])
        else:
            if self.dims == 3 and self.resample_2d:
                x = F.interpolate(
                    x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
                )
            else:
                x = F.interpolate(x, scale_factor=2, mode="nearest")

        return x, None


class Downsample(nn.Module):
    """
    A wavelet downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, resample_2d=True, use_freq=True):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims

        self.use_freq = use_freq
        self.dwt = DWT_3D("haar")

        stride = (1, 2, 2) if dims == 3 and resample_2d else 2

        if use_conv:
            self.op = conv_nd(dims, self.channels, self.out_channels, 3, stride=stride, padding=1)
        elif self.use_freq:
            self.op = self.dwt
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        if self.use_freq:
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.op(x)
            x = (LLL / 3., (LLH, LHL, LHH, HLL, HLH, HHL, HHH))
        else:
            x = self.op(x)
        return x


class WaveletDownsample(nn.Module):
    """
    Implements the wavelet downsampling blocks used to generate the input residuals.

    :param in_ch: number of input channels.
    :param out_ch: number of output channels (should match the feature size of the corresponding U-Net level)
    """
    def __init__(self, in_ch=None, out_ch=None):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.conv = conv_nd(3, self.in_ch * 8, self.out_ch, 3, stride=1, padding=1)
        self.dwt = DWT_3D('haar')

    def forward(self, x):
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = self.dwt(x)
        x = th.cat((LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH), dim=1) / 3.
        return self.conv(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels via up- or downsampling.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels, otherwise out_channels = channels.
    :param use_conv: if True and out_channels is specified, use a spatial convolution instead of a smaller 1x1
                     convolution to change the channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    :param num_groups: if specified, the number of groups in the (adaptive) group normalization layers.
    :param use_freq: specifies if frequency aware up- or downsampling should be used.
    :param z_emb_dim: the dimension of the z-embedding.

    """

    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=True, use_scale_shift_norm=False,
                 dims=2, use_checkpoint=False, up=False, down=False, num_groups=32, resample_2d=True, use_freq=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint
        self.up = up
        self.down = down
        self.num_groups = num_groups
        self.use_freq = use_freq


        # Define (adaptive) group normalization layers
        self.in_layers = nn.Sequential(
            normalization(channels, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # Check if up- or downsampling should be performed by this ResBlock
        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
            self.x_upd = Upsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
        elif down:
            self.h_upd = Downsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
            self.x_upd = Downsample(channels, False, dims, resample_2d=resample_2d, use_freq=self.use_freq)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # Define the timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )

        # Define output layers including (adaptive) group normalization
        self.out_layers = nn.Sequential(
            normalization(self.out_channels, self.num_groups),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )

        # Define skip branch
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)


    def forward(self, x, temb):
        # Make sure to pipe skip connections
        if isinstance(x, tuple):
            hSkip = x[1]
        else:
            hSkip = None

        # Forward pass for ResBlock with up- or downsampling
        if self.updown:
            if self.up:
                x = x[0]
            h = self.in_layers(x)

            if self.up:
                h = (h, hSkip)
                x = (x, hSkip)

            h, hSkip = self.h_upd(h)   # Updown in main branch (ResBlock)
            x, xSkip = self.x_upd(x)   # Updown in skip-connection (ResBlock)

        # Forward pass for standard ResBlock
        else:
            if isinstance(x, tuple):  # Check for skip connection tuple
                x = x[0]
            h = self.in_layers(x)

        # Common layers for both standard and updown ResBlocks
        emb_out = self.emb_layers(temb)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)

        else:
            h = h + emb_out  # Add timestep embedding
            h = self.out_layers(h)  # Forward pass out layers

        # Add skip connections
        out = self.skip_connection(x) + h
        out = out, hSkip

        return out



class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
            num_groups=32,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                    channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels, num_groups)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class WavUNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which attention will take place. May be a set,
                                  list, or tuple. For example, if this contains 4, then at 4x downsampling, attention
                                  will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially increased efficiency.
    """

    def __init__(self, image_size, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                 dropout=0, channel_mult=(1, 2, 4, 8), conv_resample=True, dims=2, num_classes=None,
                 use_checkpoint=False, use_fp16=False, num_heads=1, num_head_channels=-1, num_heads_upsample=-1,
                 use_scale_shift_norm=False, resblock_updown=False, use_new_attention_order=False, num_groups=32,
                 bottleneck_attention=True, resample_2d=True, additive_skips=False, decoder_device_thresh=0,
                 use_freq=False, progressive_input='residual'):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        # self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        # self.num_heads = num_heads
        # self.num_head_channels = num_head_channels
        # self.num_heads_upsample = num_heads_upsample
        self.num_groups = num_groups
        self.bottleneck_attention = bottleneck_attention
        self.devices = None
        self.decoder_device_thresh = decoder_device_thresh
        self.additive_skips = additive_skips
        self.use_freq = use_freq
        self.progressive_input = progressive_input

        #############################
        # TIMESTEP EMBEDDING layers #
        #############################
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim))

        ###############
        # INPUT block #
        ###############
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        input_pyramid_channels =in_channels
        ds = 1

        ######################################
        # DOWNWARD path - Feature extraction #
        ######################################
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):                                         # Adding Residual blocks
                layers = [
                    ResBlock(
                        channels=ch,
                        emb_channels=time_embed_dim,
                        dropout=dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        use_freq=self.use_freq,
                    )
                ]
                ch = mult * model_channels  # New input channels = channel_mult * base_channels
                                            # (first ResBlock performs channel adaption)

                if ds in attention_resolutions:                                     # Adding Attention layers
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            num_groups=self.num_groups,
                        )
                    )

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            # Adding downsampling operation
            out_ch = ch
            layers = []
            layers.append(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=out_ch,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        down=True,
                        num_groups=self.num_groups,
                        resample_2d=resample_2d,
                        use_freq=self.use_freq,
                    )
                    if resblock_updown
                    else Downsample(
                        ch,
                        conv_resample,
                        dims=dims,
                        out_channels=out_ch,
                        resample_2d=resample_2d,
                    )
                )
            self.input_blocks.append(TimestepEmbedSequential(*layers))

            layers = []
            if self.progressive_input == 'residual':
                layers.append(WaveletDownsample(in_ch=input_pyramid_channels, out_ch=out_ch))
                input_pyramid_channels = out_ch

            self.input_blocks.append(TimestepEmbedSequential(*layers))

            ch = out_ch
            input_block_chans.append(ch)
            ds *= 2
            self._feature_size += ch

        self.input_block_chans_bk = input_block_chans[:]

        #########################
        # LATENT/ MIDDLE blocks #
        #########################
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
                use_freq=self.use_freq,
            ),
            *([AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                num_groups=self.num_groups,
            )] if self.bottleneck_attention else [])
            ,
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                num_groups=self.num_groups,
                resample_2d=resample_2d,
                use_freq=self.use_freq,
            ),
        )
        self._feature_size += ch

        #################################
        # UPWARD path - feature mapping #
        #################################
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks+1):                                          # Adding Residual blocks
                if not i == num_res_blocks:
                    mid_ch = model_channels * mult

                    layers = [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=mid_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                            use_freq=self.use_freq,
                        )
                    ]
                    if ds in attention_resolutions:                                         # Adding Attention layers
                        layers.append(
                            AttentionBlock(
                                mid_ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=num_head_channels,
                                use_new_attention_order=use_new_attention_order,
                                num_groups=self.num_groups,
                            )
                        )
                    ch = mid_ch
                else:                                                                       # Adding upsampling operation
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            mid_ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            num_groups=self.num_groups,
                            resample_2d=resample_2d,
                            use_freq=self.use_freq,
                        )
                        if resblock_updown
                        else Upsample(
                            mid_ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            resample_2d=resample_2d
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                mid_ch = ch

        ################
        # Out ResBlock #
        ################
        self.out_res = nn.ModuleList([])
        for i in range(num_res_blocks):
            layers = [
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=ch,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    num_groups=self.num_groups,
                    resample_2d=resample_2d,
                    use_freq=self.use_freq,
                )
            ]
            self.out_res.append(TimestepEmbedSequential(*layers))

        ################
        # OUTPUT block #
        ################
        self.out = nn.Sequential(
            normalization(ch, self.num_groups),
            nn.SiLU(),
            conv_nd(dims, model_channels, out_channels, 3, padding=1),
        )

    def to(self, *args, **kwargs):
        """
        we overwrite the to() method for the case where we
        distribute parts of our model to different devices
        """
        if isinstance(args[0], (list, tuple)) and len(args[0]) > 1:
            assert not kwargs and len(args) == 1
            # distribute to multiple devices
            self.devices = args[0]
            # move first half to first device, second half to second device
            self.input_blocks.to(self.devices[0])
            self.time_embed.to(self.devices[0])
            self.middle_block.to(self.devices[0])  # maybe devices 0
            for k, b in enumerate(self.output_blocks):
                if k < self.decoder_device_thresh:
                    b.to(self.devices[0])
                else:  # after threshold
                    b.to(self.devices[1])
            self.out.to(self.devices[0])
            print(f"distributed UNet components to devices {self.devices}")

        else:  # default behaviour
            super().to(*args, **kwargs)
            if self.devices is None:  # if self.devices has not been set yet, read it from params
                p = next(self.parameters())
                self.devices = [p.device, p.device]

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param zemb: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        hs = []  # Save skip-connections here
        input_pyramid = x
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))  # Gen sinusoidal timestep embedding
        h = x
        self.hs_shapes = []

        for module in self.input_blocks:
            if not isinstance(module[0], WaveletDownsample):
                h = module(h, emb)  # Run a downstream module
                skip = None
                if isinstance(h, tuple):  # Check for skip features (tuple of high frequency subbands) and store in hs
                    h, skip = h
                hs.append(skip)
                self.hs_shapes.append(h.shape)
            else:
                input_pyramid = module(input_pyramid, emb)
                input_pyramid = input_pyramid + h
                h = input_pyramid

        for module in self.middle_block:
            h = module(h, emb)
            if isinstance(h, tuple):
                h, skip = h

        for module in self.output_blocks:
            new_hs = hs.pop()
            if new_hs:
                skip = new_hs

            # Use additive skip connections
            if self.additive_skips:
                h = (h + new_hs) / np.sqrt(2)

            # Use frequency aware skip connections
            elif self.use_freq:  # You usually want to use the frequency aware upsampling
                if isinstance(h, tuple):  # Replace None with the stored skip features
                    l = list(h)
                    l[1] = skip
                    h = tuple(l)
                else:
                    h = (h, skip)

            # Use concatenation
            else:
                h = th.cat([h, new_hs], dim=1)

            h = module(h, emb)  # Run an upstream module

        for module in self.out_res:
            h = module(h, emb)

        h, _ = h
        return self.out(h)

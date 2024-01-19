import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.normal import Normal
from . import layers
from . import utils

def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding='same',
        )
        kaiming_init(module=self.main, distribution='normal')
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class ConvDropBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for affine feature detector.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1, dropout=0):
        super().__init__()

        Conv = getattr(nn, f'Conv{ndims}d')
        Drop = getattr(nn, f'Dropout{ndims}d')
        self.main = Conv(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding='same',
        )
        kaiming_init(module=self.main, distribution='normal')
        self.dropout  = Drop(dropout)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,    
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            conv_block: The set of layers used for a single Unet block
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for _, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class UnetAffine(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 add_features = None,
                 nb_levels=None,
                 max_pool=2,    
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False):
        """
        Parameters:
            conv_block: The set of layers used for a single Unet block
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_enc_convs = len(enc_nf)
        nb_dec_convs = len(dec_nf)
        if nb_dec_convs < nb_enc_convs:
            final_convs = dec_nf + add_features
            dec_nf = []
        else:
            final_convs = dec_nf[nb_dec_convs:]
            dec_nf = dec_nf[:nb_dec_convs]

        self.nb_enc_levels = int(nb_enc_convs / nb_conv_per_level) + 1
        self.nb_dec_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_enc_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_enc_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvDropBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)
        
        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_dec_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                
                convs.append(ConvDropBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_dec_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for _, nf in enumerate(final_convs):
            self.remaining.append(ConvDropBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_dec_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x

class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_resolution=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_resolution: Resolution (relative voxel size) of the flow field during
                vector integration. Default is 2.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
        """
        
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        unet_half_res = True if int_resolution == 2 else False
        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding='same')
    
        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
       
        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                'Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers (downsize)
        if int_steps > 0 and int_resolution > 1:
            # self.resize = layers.ResizeTransform(int_resolution, ndims)
            self.resize = layers.RescaleTransform(1 / int_resolution)
            
        else:
            self.resize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        self.integrate = layers.VecInt(int_steps=int_steps) if int_steps > 0 else None
        
        # resize to full res
        if int_steps > 0 and int_resolution > 1:
            self.fullsize = layers.RescaleTransform(int_resolution)
            
        else:
            self.fullsize = None
            
        # configure transformer
        self.transformer = layers.SpatialTransformer('linear', 'ij')

        
    def forward(self, source, target):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return integrated flow during inference. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)
    
        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        if self.resize:
            pos_flow = self.resize(pos_flow)

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None
    
        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None
                
        # warp image with flow field
        y_source = self.transformer([source, pos_flow])
        y_target = self.transformer([target, neg_flow]) if self.bidir else None

        # return warped image and deformation field
        results = {
            'y_source': y_source,
            'y_target': y_target,
            'flow': pos_flow,
        }

        return results


class VxmAffineFeatureDetector(nn.Module):
    """
    SynthMorph network for symmetric affine or rigid registration of two images.

    References:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.265325

    """
    def __init__(
        self,
        in_shape=None,
        input_model=None,
        num_chan=1,
         dropout=0,
        num_feat=64,
        enc_nf=[256] * 4,
        dec_nf=[256] * 0,
        add_nf=[256] * 4,
        per_level=1,
        unet_half_res=True,
        src_feats=1,
        trg_feats=1,
        half_res=True,
        weighted=True,
        rigid=False,
        make_dense=True,
        bidir=False,
        return_trans_to_mid_space=False,
        return_trans_to_half_res=False,
        return_moved=False,
        return_feat=False
    ) -> None:
        """
        Internally, the model computes transforms in a centered frame at full resolution. However,
        matrix transforms returned with `make_dense=False` operate on zero-based indices to
        facilitate composition, in particular when changing resolution. Thus, any subsequent
        `SpatialTransformer` or `ComposeTransform` calls require `shift_center=False`.

        While the returned transforms always apply to full-resolution images, you can use the flag
        `return_trans_to_half_res=True` to obtain transforms producing outputs at half resolution,
        for faster training. Careful: this requires setting the adequate output `shape` for
        `SpatialTransformer` when applying transforms.

        Parameters:
            in_shape: Spatial dimensions of a single input image
            input_model: Model whose outputs will be used as data inputs, and whose inputs will be
                used as inputs to the returned model, as an alternative to specifying `in_shape`.
            num_chan: Number of input-image channels.
            num_feat: Number of output feature maps giving rise to centers of mass.
            enc_nf: Number of convolutional encoder filters at each level, as an iterable. The
                model will downsample by a factor of 2 after each convolution.
            dec_nf: Number of convolutional decoder filters at each level, as an iterable. The
                model will upsample by a factor of 2 after each convolution.
            add_nf: Number of additional convolutional filters applied at the end, as an iterable.
                The model will maintain the resolution after these convolutions.
            per_level: Number of encoding and decoding convolution repeats.
            dropout: Spatial dropout rate applied after each convolution.
            half_res: For efficiency, halve the input-image resolution before registration.
            weighted: Fit transforms using weighted instead of ordinary least squares.
            rigid: Discard scaling and shear to return a rigid transform.
            make_dense: Return a dense displacement field instead of a matrix transform.
            bidir: In addition to the transform from image 1 to image 2, also return the inverse.
                The transforms apply to full-resolution images but may end half way and/or at half
                resolution, depending on `return_trans_to_mid_space`, `return_trans_to_half_res`.
                Also return pairs of moved images and feature maps, if requested.
            return_trans_to_mid_space: Return transforms from the input images to the mid-space.
                Careful: your loss inputs must reflect this choice, and training with large
                transforms may lead to NaN loss values. You can change this option after training.
            return_trans_to_half_res: Return transforms from input images at full resolution to
                output images at half resolution. You can change this option after training.
            return_moved: Append the transformed images to the model outputs.
            return_feat: Append the output feature maps to the model outputs.

        """
        super().__init__()

        # Dimensions
        shape_full = np.asarray(in_shape)
        shape_half = shape_full // 2
        ndims = len(shape_full)
        assert ndims in (2, 3), 'only 2D and 3D supported'
        assert not return_trans_to_half_res or half_res, 'only for `half_res=True`'
        
        # Layers
        conv = getattr(nn, f'Conv{ndims}d')
        maxpool = getattr(nn, f'MaxPool{ndims}d')
        
        dtype = torch.get_default_dtype()

        self.halfres_st = layers.SpatialTransformer(fill_value=0, shape=shape_half, shift_center=False)
        self.det = UnetAffine(
            inshape=shape_full,
            infeats=(src_feats + trg_feats),
            nb_features=[enc_nf, dec_nf],
            add_features=add_nf,
            nb_conv_per_level=per_level,
            half_res=unet_half_res,
        )

        # # Downsampling and upsampling operations
        # maxpool_scale = [2] * len(enc_nf)
        # self.pooling = [maxpool(kernel_size=ms, dtype=torch.float32) for ms in maxpool_scale]
        # self.upsampling = [nn.Upsample(scale_factor=ms, mode='nearest') for ms in maxpool_scale]

        # # Feature detector: encoder.
        # prev_nf = num_chan
        # self.encoder = nn.ModuleList()
        # for nf in enc_nf:
        #     blocks = nn.ModuleList()
        #     for _ in range(per_level):
        #         blocks.append(
        #             ConvDropBlock(
        #                 ndims=ndims, 
        #                 in_channels=prev_nf, 
        #                 out_channels=nf, 
        #                 dropout=dropout
        #             )
        #         )
        #         prev_nf = nf
        #     self.encoder.append(blocks)

        # # Feature detector: decoder
        # self.decoder = nn.ModuleList()
        # for nf in dec_nf:
        #     blocks = nn.ModuleList()
        #     for _ in range(per_level):
        #         blocks.append(
        #             ConvDropBlock(
        #                 ndims=ndims, 
        #                 in_channels=prev_nf, 
        #                 out_channels=nf, 
        #                 dropout=dropout
        #             )
        #         )
        #         prev_nf = nf
        #     self.decoder.append(blocks)
        
        # # Additional convolutions
        # self.remaining = nn.ModuleList()
        # for nf in add_nf:
        #     self.remaining.append(
        #         ConvDropBlock(
        #             ndims=ndims, 
        #             in_channels=prev_nf, 
        #             out_channels=nf, 
        #             dropout=dropout
        #         )
        #     )
        #     prev_nf = nf
        
        # self.det = nn.ModuleList(self.encoder + self.decoder + self.remaining)


        # Output features
        self.out = nn.ModuleList()
        self.out.append(conv(self.det.final_nf, num_feat, kernel_size=3, padding='same'))
        self.out.append(nn.ReLU())

        self.com = layers.CenterOfMass(axes=range(1, ndims + 1), normalize=True, shift_center=True, dtype=dtype)
        self.ls_fit = layers.LeastSquaresFit()
        self.params_to_aff = layers.ParamsToAffineMatrix(ndims=ndims)
        self.compose = layers.ComposeTransform(shift_center=False)
        
        shape_out = shape_half if return_trans_to_half_res else shape_full
        self.aff_to_dense = layers.AffineToDenseShift(shape_out, shift_center=False)
        self.out_st = layers.SpatialTransformer(shift_center=False, fill_value=0, shape=shape_out)
        

    def forward(self, source, target):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return integrated flow during inference. Default is False.
        """
        x = torch.cat([source, target], dim=1)
        x = self.det(x)
        x = self.out(x)

        return
    
    # Static transforms. Function names refer to effect on coordinates.
    def _tensor(self, x, shape, dtype=torch.float32):
        x = torch.tensor(x[None, :-1, :], dtype=dtype)
        return x.repeat(shape[0], 1, 1)

    def _cen(self, ndims, shape):
        mat = torch.eye(ndims + 1)
        mat[:-1, -1] = -0.5 * (shape - 1)
        return self._tensor(mat)
    
    def _un_cen(self, ndims, shape):
        mat = torch.eye(ndims + 1)
        mat[:-1, -1] = +0.5 * (shape - 1)
        return self._tensor(mat)

    def _scale(self, ndims, fact):
        mat = torch.diag(torch.tensor([fact] * ndims + [1.0]))
        return self._tensor(mat)

    
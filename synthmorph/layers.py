import warnings
import torch
import torch.nn as nn

# local modules
from . import utils


class SpatialTransformer(nn.Module):
    """
    ND spatial transformer layer

    Applies affine and dense transforms to images. A dense transform gives
    displacements (not absolute locations) at each voxel.

    If you find this layer useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which 
    was in turn transformed to be dense with the help of (affine) STN code 
    via https://github.com/kevinzakka/spatial-transformer-network.

    Since then, we've re-written the code to be generalized to any 
    dimensions, and along the way wrote grid and interpolation functions.
    """

    def __init__(
       self,
       interp_method='linear',
       indexing='ij',
       single_transform=False,
       fill_value=None,
       shift_center=True,
       **kwargs,
    ):
        self.interp_method = interp_method
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.built = False
        super().__init__(**kwargs)  
        

    def build(self, input_shape):
        # Only build once to avoid repetition
        if self.built:
            return  
    
        # sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = utils.is_affine_shape(input_shape[1][1:])

        # make sure inputs are reasonable shapes
        if self.is_affine:
            expected = (self.ndims, self.ndims + 1)
            actual = tuple(self.trfshape[-2:])
            if expected != actual:
                raise ValueError(f'Expected {expected} affine matrix, got {actual}.')
        else:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')

        # confirm built
        self.built = True


    def forward(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """

        inputs = [i.permute(0, 2, 3, 1) for i in inputs]    # convert to tf format
        input_shape = [i.shape for i in inputs]
        self.build(input_shape)
        
        # # necessary for multi-gpu models
        vol = torch.reshape(inputs[0], (-1, *self.imshape))
        trf = torch.reshape(inputs[1], (-1, *self.trfshape))
        
        
        # convert affine matrix to warp field
        if self.is_affine:
            fun = lambda x: utils.affine_to_dense_shift(x, vol.shape[1: -1],
                                                shift_center=self.shift_center,
                                                indexing=self.indexing)
            trf = torch.stack([fun(t) for t in torch.unbind(trf)])
    
        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = torch.split(trf, trf.shape[-1], dim=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = torch.cat(trf_lst, dim=-1)

        # map transform across batch
        out = None
        if self.single_transform:
            out = torch.stack([self._single_transform([v, trf[0, :]]) for v in torch.unbind(vol)])
        else:
            out = torch.stack([self._single_transform([v, t]) for v, t in zip(torch.unbind(vol), torch.unbind(trf))])
        
        return out.permute(0, 3, 1, 2)  # convert back to torch format
        

    def _single_transform(self, inputs):    
        return utils.transform(inputs[0], inputs[1], interp_method=self.interp_method,
                               fill_value=self.fill_value)


class VecInt(nn.Module):
    """
    Vector integration layer

    Enables vector integration via several methods (ode or quadrature for
    time-dependent vector fields and scaling-and-squaring for stationary fields)

    If you find this function useful, please cite:

      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.
    """

    def __init__(self,
                 indexing='ij',
                #  method='ss',
                 int_steps=7,
                #  out_time_pt=1,
                #  ode_args=None,
                #  odeint_fn=None,
                 **kwargs):
    
        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing
        # self.method = method
        self.int_steps = int_steps
        self.inshape = None
        self.built = False
        # self.out_time_pt = out_time_pt
        # self.ode_args = ode_args
        # self.odeint_fn = odeint_fn  # if none then will use a tensorflow function
       
        # if ode_args is None:
        #     self.ode_args = {'rtol': 1e-6, 'atol': 1e-12}
        super(self.__class__, self).__init__(**kwargs)
    
    def build(self, input_shape):
        if self.built:
            return

        trf_shape = input_shape
        if isinstance(input_shape[0], (list, tuple)):
            trf_shape = input_shape[0]
        self.inshape = trf_shape

        if trf_shape[-1] != len(trf_shape) - 2:
            raise Exception('transform ndims %d does not match expected ndims %d'
                            % (trf_shape[-1], len(trf_shape) - 2))
        self.built = True
    
    def forward(self, inputs):
        inputs = [torch.unsqueeze(i, 0) for i in inputs]
        inputs = [i.permute(0, 2, 3, 1) for i in inputs]    # convert to tf format
        input_shape = [i.shape for i in inputs]
        self.build(input_shape)

        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        loc_shift = inputs[0]

        # Necessary for multi-gpu models
        loc_shift = torch.reshape(loc_shift, [-1, *self.inshape[1:]])
        # if hasattr(inputs[0], 'shape'):
        #     loc_shift.shape = torch.Size([*inputs[0].shape])
        
        # prepare location shift
        if self.indexing == 'xy':
            # Shift the first two dimensions (0 and 1) by swapping them
            loc_shift_split = torch.split(loc_shift, loc_shift.shape[-1], dim=-1)
            loc_shift_lst = [loc_shift_split[1], loc_shift_split[0], *loc_shift_split[2:]]
            loc_shift = torch.cat(loc_shift_lst, dim=-1)
        
        # if len(inputs) > 1:
        #     assert self.out_time_pt is None, \
        #         'out_time_pt should be None if providing batch_based out_time_pt'

        # map transform across batch
        out_tensors = [loc_shift] + inputs[1:]
        # out = torch.stack([self._single_int(torch.unsqueeze(t, 0)) for t in out_tensors])
        out = torch.stack([self._single_int(t) for t in out_tensors])
        out = out.permute(0, 3, 1, 2)   # convert back to torch format
  
        # if hasattr(inputs[0], 'shape'):
        #     out.shape = inputs[0].shape

        return out

    def _single_int(self, inputs):
        vel = inputs[0]
        # out_time_pt = self.out_time_pt
        # if len(inputs) == 2:
            # out_time_pt = inputs[1]
        int_vec = utils.integrate_vec(vel, nb_steps=self.int_steps,
                                #    ode_args=self.ode_args,
                                #    out_time_pt=out_time_pt,
                                #    odeint_fn=self.odeint_fn
                                )

        return int_vec
    

class RescaleTransform(nn.Module):
    """ 
    Rescale transform layer

    Rescales a dense or affine transform. If dense, this involves resizing and
    rescaling the vector field.
    """
    
    def __init__(self, zoom_factor, interp_method='linear', **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.built = False
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        if self.built:
            return
        self.is_affine = utils.is_affine_shape(input_shape[1:])
        self.ndims = input_shape[-1] -1 if self.is_affine else input_shape[-1]
        self.built = True

    def forward(self, transform):
        transform = transform.permute(0, 2, 3, 1)   # convert to tf format
        input_shape = transform.shape
        self.build(input_shape)

        transform = transform.squeeze(0)    # remove batch if batch == 1
        out = None
        if self.is_affine:
            out = utils.rescale_affine(transform, self.zoom_factor)
        else:
            out = utils.rescale_dense_transform(transform, self.zoom_factor,
                                                    interp_method=self.interp_method)
        # If unbatched, add batch axis
        if len(out.shape) < 4:
            out = out.unsqueeze(0)

        out = out.permute(0, 3, 1, 2)   # convert back to torch format

        return out
    

class Resize(nn.Module):
    """
    N-D Resize Torch Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.built = False
        super(self.__class__, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume should be a *vol_shape x N
        """
        if self.built:
            return
        
        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')
        
        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        if not isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor = [self.zoom_factor] * self.ndims
        else:
            assert len(self.zoom_factor) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'\
                .format(len(self.zoom_factor), self.ndims)

        # confirm built
        self.built = True


    def forward(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """
        inputs = [torch.unsqueeze(i, 0) for i in inputs]
        inputs = [i.permute(0, 2, 3, 1) for i in inputs]    # convert to tf format
        input_shape = [i.shape for i in inputs]
        self.build(input_shape)
        
        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = torch.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        out = torch.stack([self._single_resize(t) for t in vol])
        indices = list(range(len(out.shape)))
        out = out.permute(0, -1, *indices[1:-1])
        return out

    def _single_resize(self, inputs):
        return utils.resize(inputs, self.zoom_factor, interp_method=self.interp_method)
    

class RescaleValues(nn.Module):
    """
    A simple Torch layer to rescale data values by a fixed factor
    """
    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(self.__class__, self).__init__(**kwargs)

    def forward(self, x):
        return x * self.resize


class ComposeTransform(nn.Module):
    """ 
    Composes a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = ComposeTransform()([A, B, C])
    """

    def __init__(self, interp_method='linear', shift_center=True, **kwargs) -> None:
        self.interp_method = interp_method
        self.shift_center = shift_center
        super().__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config.copy()
        config.update({
            'interp_method': self.interp_method,
            'shift_center': self.shift_center,
        })
        return config

    def build(self, input_type, **kwargs):
        # Sanity check on input
        if not isinstance(input_type, (list, tuple)):
            raise Exception('ComposeTransform must be called for a list of transforms.')
    
    def forward(self, transforms):
        """
        Parameters:
            transforms: List of affine or dense transforms to compose.
        """
        self.build(transforms)

        if len(transforms) == 1:
            return transforms[0]
        
        # Convert to TF shape
        transforms_shape_indices = list(range(transforms[0].ndim))
        transforms = [trf.permute(0, *transforms_shape_indices[2:], 1) for trf in transforms]
        
        composed = utils.compose(
            transforms,
            interp_method=self.interp_method,
            shift_center=self.shift_center
        )
        composed = composed.permute(0, -1, *transforms_shape_indices[1:-1]) # convert back to Torch format

        return composed
    

class ParamsToAffineMatrix(nn.Module):
    """
    Constructs an affine transformation matrix from translation, rotation, scaling and shearing
    parameters in 2D or 3D.

    If you find this layer useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self, ndims=3, deg=True, shift_scale=False, last_row=False, **kwargs):
        """
        Parameters:
            ndims: Dimensionality of transform matrices. Must be 2 or 3.
            deg: Whether the input rotations are specified in degrees.
            shift_scale: Add 1 to any specified scaling parameters. This may be desirable
                when the parameters are estimated by a network.
            last_row: Whether to return a full matrix, including the last row.
        """
        self.ndims = ndims
        self.deg = deg
        self.shift_scale = shift_scale
        self.last_row = last_row
        super().__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ndims': self.ndims,
            'deg': self.deg,
            'shift_scale': self.shift_scale,
            'last_row': self.last_row,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.ndims + int(self.last_row), self.ndims + 1)
    
    def forward(self, params):
        """
        Parameters:
            params: Parameters as a vector which corresponds to translations, rotations, scaling
                    and shear. The size of the last axis must not exceed (N, N+1), for N
                    dimensions. If the size is less than that, the missing parameters will be
                    set to the identity.
        """
        # Convert to TF shape
        params_shape_indices = list(range(params.ndim))
        params = params.permute(0, *params_shape_indices[2:], 1)

        mat = utils.params_to_affine_matrix(
            par=params,
            deg=self.deg,
            shift_scale=self.shift_scale,
            ndims=self.ndims,
            last_row=self.last_row
        )
        # Convert back to TF format
        mat_shape_indices = list(range(mat.ndim))
        mat = mat.permute(0, -1, *mat_shape_indices[1:-1])

        return mat


class AffineToDenseShift(nn.Module):
    """
    Converts an affine transform to a dense shift transform.
    """

    def __init__(self, shape, shift_center=True, **kwargs):
        """
        Parameters:
            shape: Target shape of dense shift
        """
        self.shape = shape
        self.ndims = len(shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'shift_center': self.shift_center,
        })
        return config
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape, self.ndims)

    def build(self, input_shape):
        utils.validate_affine_shape(input_shape)
    
    def forward(self, mat):
        """
        Parameters:
            mat: Affine matrices of shape (B, N, N+1).
        """
        self.build(mat.shape)

        # Convert to TF format
        mat_shape_indices = list(range(mat.ndim))
        mat = mat.permute(0, *mat_shape_indices[2:], 1)

        dense = utils.affine_to_dense_shift(mat, self.shape, shift_center=self.shift_center)

        # Convert back to Torch format
        dense_shape_indices = list(range(dense.ndim))
        dense = dense.permute(0, -1, *dense_shape_indices[1: -1])

        return dense


class CenterOfMass(nn.Module):
    """
    Compute the barycenter of extracted features along specified axes
    """
    def __init__(self, axes, normalize=True, shift_center=True, dtype=torch.float32, **kwargs):
        
        self.axes = axes
        self.normalize = normalize
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def forward(self, feat):
       
        # Convert to TF format
        feat_shape_indices = list(range(feat.ndim))
        feat = feat.permute(0, *feat_shape_indices[2:], 1)

        center = utils.barycenter(
            axes=self.axes,
            normalize=self.normalize,
            shift_center=self.shift_center
        )
        center_shape_indices = list(range(center.ndim))
        center = center.permute(0, -1, *center_shape_indices[1:, -1])

        return center

class LeastSquaresFit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, source, target, weights=None):
        # Convert to TF format
        inputs_shape_indices = list(range(source.ndim))
        source, target = [i.permute(0, *inputs_shape_indices[2:], 1) for i in [source, target]]

        aff = utils.fit_affine(source, target, weights=weights)

        # Convert back to Torch format
        aff_shape_indices = list(range(aff.ndim))
        aff = aff.permute(0, -1, *aff_shape_indices[1:, -1])

        return aff

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
       single_transform=False,
       fill_value=None,
       shift_center=True,
       shape=None,
       **kwargs
    ):
        """
        Parameters:
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
            single_transform: Use single transform for the entire image batch.
            fill_value: Value to use for points sampled outside the domain.
                If None, the nearest neighbors will be used.
            shift_center: Shift grid to image center when converting affine
                transforms to dense transforms. Assumes the input and output spaces are identical.
            shape: ND output shape used when converting affine transforms to dense
                transforms. Includes only the N spatial dimensions. If None, the
                shape of the input image will be used. Incompatible with `shift_center=True`.

        Notes:
            There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
            indexing. Due to inconsistencies in how some functions and layers handled xy-indexing,
            we removed it in favor of default ij-indexing to minimize the potential for confusion.

        """
        self.interp_method = interp_method
        self.single_transform = single_transform
        self.fill_value = fill_value
        self.shift_center = shift_center
        self.shape = shape
        super().__init__(**kwargs)  
        

    def build(self, input_shape):
    
        # Sanity check on input list
        if len(input_shape) > 2:
            raise ValueError('Spatial Transformer must be called on a list of length 2: '
                             'first argument is the image, second is the transform.')

        # Set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.imshape = input_shape[0][1:]
        self.trfshape = input_shape[1][1:]
        self.is_affine = utils.is_affine_shape(input_shape[1][1:])

        # Make sure transform has reasonable shape (is_affine_shape throws error otherwise)
        if not self.is_affine:
            image_shape = tuple(self.imshape[:-1])
            dense_shape = tuple(self.trfshape[:-1])
            if image_shape != dense_shape:
                warnings.warn(f'Dense transform shape {dense_shape} does not match '
                              f'image shape {image_shape}.')


    def forward(self, inputs):
        """
        Parameters
            inputs: List of [img, trf], where img is the ND moving image and trf
            is either a dense warp of shape [B, D1, ..., DN, N] or an affine matrix
            of shape [B, N, N+1].
        """
        input_shape = [i.shape for i in inputs]
        self.build(input_shape)
        
        # # necessary for multi-gpu models
        vol = torch.reshape(inputs[0], (-1, *self.imshape))
        trf = torch.reshape(inputs[1], (-1, *self.trfshape))

        # map transform across batch
        if self.single_transform:
            out = torch.stack([self._single_transform([v, trf[0, :]]) for v in torch.unbind(vol)])
        else:
            out = torch.stack([self._single_transform([v, t]) for v, t in zip(torch.unbind(vol), torch.unbind(trf))])
        
        return out


    def _single_transform(self, inputs):    
        return utils.transform(
            inputs[0], 
            inputs[1], 
            interp_method=self.interp_method,
            fill_value=self.fill_value,
            shift_center=self.shift_center,
            shape=self.shape,
        )


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
        """
        Parameters:
            zoom_factor: Scaling factor.
            interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        self.is_affine = utils.is_affine_shape(input_shape[1:])
        self.ndims = input_shape[-1] -1 if self.is_affine else input_shape[-1]

    def forward(self, transform):
        input_shape = transform.shape
        self.build(input_shape)

        if transform.shape[0] == 1:
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

    def __init__(self, interp_method='linear', shift_center=True, shape=None, **kwargs) -> None:
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

        # Find a way to vectorize this without installing new packages
        composed = []
        batch_size = transforms[0].shape[0]
        for b in range(batch_size):
            batch = [t[b, ...] for t in transforms]
            composed.append(self._compose(batch))

        return torch.stack(composed)
    
    def _compose(self, inputs):
        return utils.compose(
            inputs,
            interp_method=self.interp_method,
            shift_center=self.shift_center
        )
        

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

        mat = utils.params_to_affine_matrix(
            par=params,
            deg=self.deg,
            shift_scale=self.shift_scale,
            ndims=self.ndims,
            last_row=self.last_row
        )

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

        dense = utils.affine_to_dense_shift(mat, self.shape, shift_center=self.shift_center)

        return dense


class CenterOfMass(nn.Module):
    """
    Compute the barycenter of extracted features along specified axes
    """
    def __init__(self, axes, normalize=True, shift_center=True, dtype=torch.float32, **kwargs):
        
        self.axes = axes
        self.normalize = normalize
        self.shift_center = shift_center
        self.dtype = dtype
        super().__init__(**kwargs)

    def forward(self, feat):

        center = utils.barycenter(
            feat,
            axes=self.axes,
            normalize=self.normalize,
            shift_center=self.shift_center,
            dtype=self.dtype
        )

        return center

class LeastSquaresFit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, source, target, weights):
            
        aff = utils.fit_affine(source, target, weights=weights)

        return aff
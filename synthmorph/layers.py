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
                                                    
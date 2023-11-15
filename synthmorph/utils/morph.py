import itertools
import numpy as np
import torch
import torch.nn.functional as nnf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def interpn(vol, loc, interp_method='linear', fill_value=None, device=device):
    vol = vol.to(device)
    if isinstance(loc, (list, tuple)):
        loc = torch.stack(loc, dim=-1).to(device)
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = torch.unsqueeze(vol, -1)
    
    # Flatten and float location tensors
    if not loc.dtype.is_floating_point:
        target_loc_dtype = vol.dtype if vol.dtype.is_floating_point else torch.float32
        loc = loc.to(target_loc_dtype)
    elif vol.dtype.is_floating_point and vol.dtype != loc.dtype:
        loc = loc.to(vol.dtype)

    if isinstance(vol.shape, torch.Size):
        vol_shape = list(vol.shape)
    else:
        vol_shape = vol.shape

    max_loc = [d - 1 for d in list(vol.shape)]
    
    if interp_method == 'linear':
        loc0 = loc.floor()

        # Clip values
        clipped_loc = [loc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [loc0[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        # Get other end of point cube
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[f.to(torch.int32) for f in loc0lst], [f.to(torch.int32) for f in loc1]]

        # Compute the difference between the upper value and the original value
        # Differences are basically 1 - (pt - floor(pt))
        #  because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        # Note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]
       
        # Go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
    
        for c in cube_pts:
            subs = [locs[c[d]][d] for d in range(nb_dims)]
           
            idx = sub2ind2d(vol_shape[:-1], subs)
            vol_reshape = torch.reshape(vol, [-1, vol_shape[-1]])
            vol_val = vol_reshape[idx.to(torch.int64)]  # torch version of tf.gather()

            # Get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            wt = prod_n(wts_lst)
            wt = wt.unsqueeze(-1)
            # Compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest', \
            'method should be linear or nearest, got: %s' % interp_method
        roundloc = loc.round().to(torch.int32)
        roundloc = [roundloc[..., d].clamp(0, max_loc[d]) for d in range(nb_dims)]

        idx = sub2ind2d(vol_shape[:-1], roundloc)
        interp_vol = vol.reshape(-1, vol_shape[-1])[idx]

    if fill_value is not None:
        out_type = interp_vol.dtype
        fill_value = torch.tensor(fill_value, dtype=out_type)
        below = [loc[..., d] < 0 for d in range(nb_dims)]
        above = [loc[..., d] > max_loc[d] for d in range(nb_dims)]
        out_of_bounds = torch.any(torch.stack(below + above, dim=-1), dim=-1, keepdim=True)
        interp_vol *= torch.logical_not(out_of_bounds).to(out_type)
        interp_vol += out_of_bounds.to(out_type) * fill_value

    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol


def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])
    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


def prod_n(lst):
    """
    Alternative to torch.stacking and prod
    """
    prod = lst[0].clone()
    for p in lst[1:]:
        prod *= p
    return prod


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Warning: this uses the tf.meshgrid convention, of 'xy' indexing.

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [torch.arange(0, d, device=torch.device('cuda')) for d in volshape]
    grid = torch.meshgrid(*linvec, **kwargs)

    return grid


def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors

    """
    return torch.meshgrid(*args, indexing='ij', **kwargs)


def affine_to_dense_shift(matrix, shape, 
                          shift_center=True, 
                          indexing='ij'):
    """
    Transforms an affine matrix to a dense location shift.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply affine matrix to each index.
        3. Subtract grid.

    Parameters:
        matrix: affine matrix of shape (N, N+1).
        shape: ND shape of the target warp.
        shift_center: Shift grid to image center.
        indexing: Must be 'xy' or 'ij'.

    Returns:
        Dense shift (warp) of shape (*shape, N).
    """

    if isinstance(shape, torch.Size):
        shape = list(shape)

    if not torch.is_tensor(matrix) or not matrix.is_floating_point():
        matrix = matrix.float()

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')

    # list of volume ndgrid
    # N-long list, each entry of shape
    mesh = volshape_to_meshgrid(shape, indexing=indexing)
    mesh = [f if f.dtype == matrix.dtype else f.to(matrix.dtype) for f in mesh]

    if shift_center:
        mesh = [(f - (shape[i] - 1) / 2) for i, f in enumerate(mesh)]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [f.reshape(-1) for f in mesh]
    flat_mesh.append(torch.ones_like(flat_mesh[0], dtype=matrix.dtype))
    mesh_matrix = torch.stack(flat_mesh, dim=1).transpose(0, 1)  # 4 x nb_voxels

    # compute locations
    loc_matrix = torch.matmul(matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = loc_matrix[:ndims, :].transpose(0, 1)  # nb_voxels x N
    loc = loc_matrix.reshape(shape + [ndims])  # *shape x N

    # get shifts and return
    return loc - torch.stack(mesh, dim=ndims)


def is_affine_shape(shape):
    """
    Determins whether the given shape (single-batch) represents an
    affine matrix.

    Parameters:
        shape:  List of integers of the form [N, N+1], assuming an affine.
    """
    if len(shape) == 2 and shape[-1] != 1:
        validate_affine_shape(shape)
        return True
    return False


def validate_affine_shape(shape):
    """
    Validates whether the given input shape represents a valid affine matrix.
    Throws error if the shape is valid.

    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')


def transform(vol, loc_shift, interp_method='linear', 
              indexing='ij', fill_value=None):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in Torch

    Essentially interpolates volume vol at locations determined by loc_shift. 
    This is a spatial transform in the sense that at location [x] we now have the data from, 
    [x + shift] so we've moved data.

    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
        where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
        where C is the number of channels, and D is the dimentionality len(vol_shape)
        If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
        In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
        If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]
    """
    # Parse shapes.
    # location volshape, including channels if available
    vol = torch.squeeze(vol, 0)
    loc_shift = torch.squeeze(loc_shift, 0)
    loc_volshape = loc_shift.shape[:-1]

    if isinstance(loc_volshape, torch.Size):
        loc_volshape = list(loc_volshape)

    # Volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        f'Dimension check failed for transform(): {nb_dims}D volume (shape {vol.shape[:-1]}) called with {loc_shift.shape[-1]}D transform'

    # Location should be mesh and delta
    mesh = volshape_to_meshgrid(loc_volshape, indexing=indexing) # Volume mesh
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # If channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])
    
    # Test single
    return interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def integrate_vec(vec, nb_steps):
    """
    Integrate (stationary of time-dependent) vector field (N-D Tensor) in Torch.
    Currrently only supports scaling and squaring.

    Parameters:
        vec: the Tensor field to integrate. 

    Returns:
        int_vec: integral of vector field.
    """ 

    assert nb_steps >= 0, 'nb_steps should be >= 0, found: %d' % nb_steps
    vec = vec / (2**nb_steps)
    for _ in range(nb_steps):
        vec += transform(vec, vec)
    disp = vec
    
    return disp


def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """
    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)
    
    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    
    # Skip resize for zoom_factor of 1
    if all(z == 1 for z in zoom_factor):
        return vol
    
    if not isinstance(vol_shape[0], int):
        vol_shape = list(vol_shape)
    
    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [torch.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range (ndims)]
    grid = ndgrid(*lin)
    grid = [g.to('cuda') for g in grid]

    return interpn(vol, grid, interp_method=interp_method)


def rescale_affine(mat, factor):
    """
    Rescales affine matrix by some factor.

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
        factor: Zoom factor.
    """
    scaled_translation = torch.unsqueeze(mat[..., -1] * factor, dim=-1)
    scaled_matrix = torch.cat((mat[..., :-1], scaled_translation), dim=-1)
    return scaled_matrix


def rescale_dense_transform(transform, factor, interp_method='linear'):
    """
    Rescales a dense transform. this involves resizing and rescaling the vector field.

    Parameters:
        transform: A dense warp of shape [..., D1, ..., DN, N].
        factor: Scaling factor.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
    """

    def single_batch(trf):
        if factor < 1:
            trf = resize(trf, factor, interp_method=interp_method)
            trf *= factor
        else:
            # Multiply first to save memory (multiply in smaller space)
            trf *= factor
            trf = resize(trf, factor, interp_method=interp_method)

        return trf
    
    # enable batched or non-batched input
    if len(transform.shape) > (transform.shape[-1] + 1):
        rescaled = torch.stack([single_batch(t) for t in transform], 0)
    else:
        rescaled = single_batch(transform)

    return rescaled


def gaussian_kernel(sigma,
                    windowsize=None,
                    indexing='ij',
                    separate=False,
                    random=False,
                    min_sigma=0,
                    dtype=torch.float32,
                    seed=None):
    '''
    Construct an N-dimensional Gaussian kernel.

    Parameters:
        sigma: Standard deviations, scalar or list of N scalars.
        windowsize: Extent of the kernel in each dimension, scalar or list.
        indexing: Whether the grid is constructed with 'ij' or 'xy' indexing.
            Ignored if the kernel is separated.
        separate: Whether the kernel is returned as N separate 1D filters.
        random: Whether each standard deviation is uniformily sampled from the
            interval [min_sigma, sigma).
        min_sigma: Lower bound of the standard deviation, only considered for
            random sampling.
        dtype: Data type of the output. Should be floating-point.
        seed: Integer for reproducible randomization. It is possible that this parameter only
            has an effect if the function is wrapped in a Lambda layer.

    Returns:
        ND Gaussian kernel where N is the number of input sigmas. If separated,
        a list of 1D kernels will be returned.

    For more information see:
        https://github.com/adalca/mivt/blob/master/src/gaussFilt.m

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    # Data type.
    assert dtype.is_floating_point, f'{dtype.name} is not a real floating-point type'
    np_dtype = numpy_torch_dtype(dtype)

    # Kernel width.
    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    if not isinstance(min_sigma, (list, tuple)):
        min_sigma = [min_sigma] * len(sigma)
    sigma = [max(f, np.finfo(np_dtype).eps) for f in sigma]
    min_sigma = [max(f, np.finfo(np_dtype).eps) for f in min_sigma]

    # Kernel size.
    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]
    if not isinstance(windowsize, (list, tuple)):
        windowsize = [windowsize]
    if len(sigma) != len(windowsize):
        raise ValueError(f'sigma {sigma} and width {windowsize} differ in length')

    # Precompute grid.
    center = [(w - 1) / 2 for w in windowsize]
    mesh = [np.arange(w) - c for w, c in zip(windowsize, center)]
    mesh = [-0.5 * x**2 for x in mesh]
    if not separate:
        mesh = np.meshgrid(*mesh, indexing=indexing)
    mesh = [torch.tensor(m, dtype=dtype, device=device) for m in mesh]     # tf.constant

    # Exponents.
    if random:
        seeds = np.random.default_rng(seed).integers(np.iinfo(int).max, size=len(sigma))
        max_sigma = sigma
        sigma = []
        for a, b, s in zip(min_sigma, max_sigma, seeds):
            rng = torch.Generator(device=device).manual_seed(s.item())
            sigma_val = torch.rand(size=(1,),generator=rng, dtype=dtype, device=device)
            sigma_val = a + (b - a) * sigma_val
            sigma.append(sigma_val)

    exponent = [m / s**2 for m, s in zip(mesh, sigma)]

    # Kernel.
    if not separate:
        exponent = [torch.sum(torch.stack(exponent), axis=0)]
    kernel = [torch.exp(x) for x in exponent]
    kernel = [x / torch.sum(x) for x in kernel]

    return kernel if len(kernel) > 1 else kernel[0]


def separable_conv(x,
                   kernels,
                   axis=None,
                   batched=False,
                   padding='SAME',
                   strides=None,
                   dilations=None):
    '''
    Efficiently apply 1D kernels along axes of a tensor with a trailing feature
    dimension. The same filters will be applied across features.

    Inputs:
        x: Input tensor with trailing feature dimension.
        kernels: A single kernel or a list of kernels, as tensors or NumPy arrays.
            If a single kernel is passed, it will be applied along all specified axes.
        axis: Spatial axes along which to apply the kernels, starting from zero.
            A value of None means all spatial axes.
        padding: Whether padding is to be used, either "VALID" or "SAME".
        strides: Optional output stride as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.
        dilations: Optional filter dilation rate as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.

    Returns:
        Tensor with the same type as the input.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    if isinstance(padding, str):
        padding = padding.lower()

    # Shape.
    if not batched:
        x = torch.unsqueeze(x, dim=0)
    shape_space = x.shape[1:-1]
    num_dim = len(shape_space)

    # Axes.
    if np.isscalar(axis):
        axis = [axis]
    axes_space = range(num_dim)
    if axis is None:
        axis = axes_space
    assert all(ax in axes_space for ax in axis), 'non-spatial axis passed'

    # Conform strides and dilations.
    ones = np.ones(num_dim, np.int32)
    f = map(lambda x: 1 if x is None else x, (strides, dilations))
    f = map(np.ravel, f)
    f = map(np.ndarray.tolist, f)
    f = map(lambda x: x * len(axis) if len(x) == 1 else x, f)
    f = map(lambda x: [(*ones[:ax], x[i], *ones[ax + 1:]) for i, ax in enumerate(axis)], f)
    strides, dilations = f
    assert len(strides) == len(axis), 'number of strides and axes differ'
    assert len(dilations) == len(axis), 'number of dilations and axes differ'

    # Conform kernels.
    if not isinstance(kernels, (tuple, list)):
        kernels = [kernels]
    if len(kernels) == 1:
        kernels = kernels.copy() * len(axis)
    assert len(kernels) == len(axis), 'number of kernels and axes differ'

    # Merge features and batches.
    ind = np.arange(num_dim + 2)
    forward = (0, ind[-1], *ind[1:-1])
    backward = (0, *ind[2:], 1)
    x = torch.permute(x, forward)
    shape_bc = x.shape[:2]
    merge_shape = np.concatenate(
        (
            np.prod(shape_bc, keepdims=True),
            shape_space,
            [1],
        ),
        axis=0,
    )
    x = torch.reshape(x, shape=tuple(merge_shape))

    # Convolve.
    for ax, k, s, d in zip(axis, kernels, strides, dilations):
        width = np.prod(k.shape)
        k = torch.reshape(k, shape=(*ones[:ax], width, *ones[ax + 1:], 1, 1))
        x = nnf.conv2d(x, k, padding=padding, stride=s, dilation=d)     # tf.nn.convolution

    # Restore dimensions.
    restore_shape = np.concatenate((shape_bc, tuple(x.shape)[1:-1]), axis=0)
    x = torch.reshape(x, shape=tuple(restore_shape))
    x = torch.permute(x, backward)

    return x if batched else x[0, ...]


def numpy_torch_dtype(dtype):
    """
    Source:
        https://github.com/pytorch/pytorch/blob/v2.1.0/torch/testing/_internal/common_utils.py#L1057
        Lines 1221 - 1234
    """
    numpy_to_torch_dtype_dict = {
        np.bool_      : torch.bool,
        np.uint8      : torch.uint8,
        np.int8       : torch.int8,
        np.int16      : torch.int16,
        np.int32      : torch.int32,
        np.int64      : torch.int64,
        np.float16    : torch.float16,
        np.float32    : torch.float32,
        np.float64    : torch.float64,
        np.complex64  : torch.complex64,
        np.complex128 : torch.complex128
    }

    # Dict of torch dtype -> NumPy dtype
    torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}
    torch2numpy = lambda x: torch_to_numpy_dtype_dict[x]
    
    result = numpy_to_torch_dtype_dict.get(dtype, torch2numpy(dtype))
    assert result is not None, "Dtype not found"
    
    return result
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


def volshape_to_meshgrid(volshape, device=device, **kwargs):
    """
    Compute Tensor meshgrid from a volume size

    Warning: this uses the 'xy' indexing convention

    Parameters:
        volshape: the volume size
        **kwargs: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [torch.arange(0, d, device=device) for d in volshape]
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
                          warp_right=None):
    """
    Convert N-dimensional (ND) matrix transforms to dense displacement fields.

    Algorithm:
        1. Build and (optionally) shift grid to center of image.
        2. Apply matrices to each index coordinate.
        3. Subtract grid.

    Parameters:
        matrix: Affine matrix of shape (..., M, N + 1), where M is N or N + 1. Can have any batch
            dimensions.
        shape: ND shape of the output space.
        shift_center: Shift grid to image center.
        warp_right: Right-compose the matrix transform with a displacement field of shape
            (..., *shape, N), with batch dimensions broadcastable to those of `matrix`.

    Returns:
        Dense shift (warp) of shape (..., *shape, N).
    """

    if isinstance(shape, torch.Size):
        shape = list(shape)

    if not torch.is_tensor(matrix) or not matrix.is_floating_point():
        matrix = torch.as_tensor(matrix, torch.float32)

    # check input shapes
    ndims = len(shape)
    if matrix.shape[-1] != (ndims + 1):
        matdim = matrix.shape[-1] - 1
        raise ValueError(f'Affine ({matdim}D) does not match target shape ({ndims}D).')

    # Coordinate grid
    mesh = (torch.arange(s, dtype=matrix.dtype) for s in shape)
    if shift_center:
        mesh = (m - 0.5 * (s - 1) for m, s in zip(mesh, shape))
    mesh = [torch.reshape(m, shape=(-1,)) for m in ndgrid(*mesh)]
    mesh = torch.stack(mesh)
    out = mesh

    # Optionally right-compose with warp field
    if warp_right is not None:
        if not torch.is_tensor(warp_right) or warp_right.dtype != matrix.dtype:
            warp_right = torch.as_tensor(warp_right, dtype=matrix.dtype)
        flat_shape = torch.cat((torch.as_tensor(warp_right.shape[:-1 - ndims]), (-1, ndims)), axis=0)
        warp_right = torch.reshape(warp_right, flat_shape)  # ... x nb_voxels x N
        out += torch.transpose(warp_right, -1, -2)    # ... x N x nb_voxels

    # Compute locations, subtract grid to obtain shift
    out = matrix[..., :ndims, :-1] @ out + matrix[..., :ndims, -1:]     # ... x N x nb_voxels
    out = torch.transpose(out - mesh, -1, -2)
  
    # Restore shape
    shape = np.concatenate((tuple(matrix.shape[:-2]), (*shape, ndims)), axis=0)
    return torch.reshape(out, tuple(shape.astype(np.int32)))    # ... x in_shape x N


def is_affine_shape(shape):
    """
    Determines whether the given shape (single-batch) represents an
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
    Throws error if the shape is invalid.

    Parameters:
        shape: List of integers of the form [..., N, N+1].
    """
    ndim = shape[-1] - 1
    actual = tuple(shape[-2:])
    if ndim not in (2, 3) or actual != (ndim, ndim + 1):
        raise ValueError(f'Affine matrix must be of shape (2, 3) or (3, 4), got {actual}.')


def make_square_affine(mat):
    """
    Converts a [N, N+1] affine matrix to square shape [N+1, N+1].

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
    """
    validate_affine_shape(mat.shape)
    if mat.shape[-2] == mat.shape[-1]:
        return mat

    # Support dynamic shapes by storing in tensors
    shape_input = tuple(mat.shape)
    shape_batch = shape_input[:-2]
    shape_zeros = np.concatenate((shape_batch, (1,), shape_input[-2:-1]), axis=0).astype(np.int32)
    shape_one = np.concatenate((shape_batch, (1, 1)), axis=0).astype(np.int32)

    # Append last row
    zeros = torch.zeros(size=tuple(shape_zeros), dtype=mat.dtype)
    one = torch.ones(size=tuple(shape_one), dtype=mat.dtype)
    row = torch.cat((zeros, one), dim=-1)
    return torch.cat((mat, row), dim=-2)


def invert_affine(mat):
    """
    Compute the multiplicative inverse of an affine matrix.

    Parameters:
        mat: Affine matrix of shape [..., N, N+1].
    """
    rows = mat.shape[-2]
    return torch.inverse(make_square_affine(mat))[..., :rows, :]


def transform(
    vol, 
    loc_shift, 
    interp_method='linear',
    fill_value=None,
    shift_center=True,
    shape=None,
):
    """Apply affine or dense transforms to images in N dimensions.

    Essentially interpolates the input ND tensor at locations determined by
    loc_shift. The latter can be an affine transform or dense field of location
    shifts in the sense that at location x we now have the data from x + dx, so
    we moved the data.

    Parameters:
        vol: tensor or array-like structure  of size vol_shape or
            (*vol_shape, C), where C is the number of channels.
        loc_shift: Affine transformation matrix of shape (N, N+1) or a shift
            volume of shape (*new_vol_shape, D) or (*new_vol_shape, C, D),
            where C is the number of channels, and D is the dimensionality
            D = len(vol_shape). If the shape is (*new_vol_shape, D), the same
            transform applies to all channels of the input tensor.
        interp_method: 'linear' or 'nearest'.
        fill_value: Value to use for points sampled outside the domain. If
            None, the nearest neighbors will be used.
        shift_center: Shift grid to image center when converting affine
            transforms to dense transforms. Assumes the input and output spaces are identical.
        shape: ND output shape used when converting affine transforms to dense
            transforms. Includes only the N spatial dimensions. If None, the
            shape of the input image will be used. Incompatible with `shift_center=True`.

    Returns:
        Tensor whose voxel values are the values of the input tensor
        interpolated at the locations defined by the transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    Keywords:
        interpolation, sampler, resampler, linear, bilinear
    """
    if shape is not None and shift_center:
        raise ValueError('`shape` option incompatible with `shift_center=True`')
    
    # Convert data type 
    ftype = torch.float32
    if not torch.is_tensor(vol) or not vol.is_floating_point():
        vol = torch.as_tensor(vol, dtype=ftype)
    if not torch.is_tensor(loc_shift) or not loc_shift.is_floating_point():
        loc_shift = torch.as_tensor(loc_shift, dtype=ftype)

    # Convert affine to location shift
    if is_affine_shape(loc_shift.shape):
        loc_shift = affine_to_dense_shift(
            loc_shift,
            shape=vol.shape[:-1] if shape is None else shape,
            shift_center=shift_center,
        )

    # Parse spatial location shape, including channels if available
    # vol = torch.squeeze(vol, 0)
    # loc_shift = torch.squeeze(loc_shift, 0)
    loc_volshape = loc_shift.shape[:-1]

    if isinstance(loc_volshape, torch.Size):
        loc_volshape = list(loc_volshape)

    # Volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        f'Dimension check failed for transform(): {nb_dims}D volume (shape {vol.shape[:-1]}) ' \
        f'called with {loc_shift.shape[-1]}D transform'

    # Location should be mesh and delta
    mesh = volshape_to_meshgrid(loc_volshape, indexing='ij')    # volume mesh
    mesh = list(mesh)
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = torch.as_tensor(m, dtype=loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # If channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])
    
    # Test single
    return interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)


def compose(transforms, interp_method='linear', shift_center=True, shape=None):
    """
    Compose a single transform from a series of transforms.

    Supports both dense and affine transforms, and returns a dense transform unless all
    inputs are affine. The list of transforms to compose should be in the order in which
    they would be individually applied to an image. For example, given transforms A, B,
    and C, to compose a single transform T, where T(x) = C(B(A(x))), the appropriate
    function call is:

    T = compose([A, B, C])

    Parameters:
        transforms: List of affine and/or dense transforms to compose.
        interp_method: Interpolation method. Must be 'linear' or 'nearest'.
        shift_center: Shift grid to image center.

    Returns:
        Composed affine or dense transform.

    Notes:
        There used to be an argument for choosing between matrix ('ij') and Cartesian ('xy')
        indexing. Due to inconsistencies in how some functions and layers handled xy-indexing, we
        removed it in favor of default ij-indexing to minimize the potential for confusion.

    """
    if len(transforms) < 2:
        raise ValueError("Compose transform must include 2 or more transforms")
    
    curr = transforms[-1]
    for tr in reversed(transforms[:-1]):
        # Dense warp on left: interpolate.
        if not is_affine_shape(tr.shape):
            if is_affine_shape(curr.shape):
                curr = affine_to_dense_shift(curr, shape=tr.shape[:-1], shift_center=shift_center)
            curr += transform(vol=tr, loc_shift=curr, interp_method=interp_method)
        
        # Matrix on left, dense warp on right: matrix-vector product.
        elif not is_affine_shape(curr.shape):
            curr = affine_to_dense_shift(tr,
                                         shape=curr.shape[:-1],
                                         shift_center=shift_center,
                                         warp_right=curr)
            
         # No dense warp: matrix product.
        else:
            tr = make_square_affine(tr)
            curr = make_square_affine(curr)
            curr = torch.matmul(tr, curr)[:-1]

    return curr


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
    x_shape_indices = np.arange(len(x.shape))

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

    x = x.permute(0, -1, *x_shape_indices[1:-1])    # Torch format
    # Convolve.
    for ax, k, s, d in zip(axis, kernels, strides, dilations):
        width = np.prod(k.shape)
        k = torch.reshape(k, shape=(*ones[:ax], width, *ones[ax + 1:], 1, 1))
        x = nnf.conv2d(x, k, padding=padding, stride=s, dilation=d)     # tf.nn.convolution    
    x = x.permute(0, *x_shape_indices[2:], 1)    # TF format

    # Restore dimensions.
    restore_shape = np.concatenate((shape_bc, tuple(x.shape)[1:-1]), axis=0)
    x = torch.reshape(x, shape=tuple(restore_shape))
    x = torch.permute(x, backward)

    return x if batched else x[0, ...]


def barycenter(x: torch.Tensor, axes=None, normalize=False, shift_center=False, dtype=torch.float32):
    """
    Compute barycenter along specified axes.

    Arguments:
        x:
            Input tensor of any type. Will be cast to FP32 if needed.
        axes:
            Axes along which to compute the barycenter. None means all axes.
        normalize:
            Normalize grid dimensions to unit length.
        shift_center:
            Shift grid to image center.
        dtype:
            Output data type. The computation always uses single precision.

    Returns:
        Center of mass of the specified data type.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    compute_dtype = torch.float32
    if x.dtype != compute_dtype:
        x = x.to(compute_dtype)
    
    # Move reduction axes to end
    axes_all = range(len(x.shape))
    if axes is None:
        axes = axes_all
    axes_sub = tuple(ax for ax in axes_all if ax not in axes)
    if axes_sub:
        x = x.permute(*axes_sub, *axes)

    num_dim = len(axes)
    vol_shape = x.shape[-num_dim:]

    # Coordinate grid
    grid = (torch.arange(f, dtype=x.dtype) for f in vol_shape)
    if shift_center:
        grid = (g - (v - 1) / 2 for g, v in zip(grid, vol_shape))
    if normalize:
        grid = (g / v for g, v in zip(grid, vol_shape))
    grid = torch.meshgrid(*grid, indexing='ij')
    grid = torch.stack(grid, axis=-1)

    # Reduction
    axes_red = tuple(axes_all[-num_dim:])
    x = x.unsqueeze(axis=-1)
    x = divide_no_nan(
        torch.sum(grid * x, dim=axes_red),
        torch.sum(x, dim=axes_red)
    )

    return x.to(dtype=dtype) if dtype != compute_dtype else x


def angles_to_rotation_matrix(ang, deg=True, ndims=3, device=device):
    """
    Construct N-dimensional rotation matrices from angles, where N is 2 or
    3. The direction of rotation for all axes follows the right-hand rule: the
    thumb being the rotation axis, a positive angle defines a rotation in the
    direction pointed to by the other fingers. Rotations are intrinsic, that
    is, applied in the body-centered frame of reference. The function supports
    inputs with or without batch dimensions.

    In 3D, rotations are applied in the order ``R = X @ Y @ Z``, where X, Y,
    and Z are matrices defining rotations about the x, y, and z-axis,
    respectively.

    Arguments:
        ang: Array-like input angles of shape (..., M), specifying rotations
            about the first M axes of space. M must not exceed N. Any missing
            angles will be set to zero. Lists and tuples will be stacked along
            the last dimension.
        deg: Interpret `ang` as angles in degrees instead of radians.
        ndims: Number of spatial dimensions. Must be 2 or 3.

    Returns:
        mat: Rotation matrices of shape (..., N, N) constructed from `ang`.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')
    
    if isinstance(ang, (list, tuple)):
        ang = torch.stack(ang, dim=-1)
    
    if not isinstance(par, torch.Tensor) or not par.dtype.is_floating_point():
        par = torch.as_tensor(par, dtype=torch.float32, device=device)
    
    # Add dimension to scalar
    if len(par.shape) == 0:
        par = par.reshape(shape=(1,))
    
    # Validate shape
        num_ang = 1 if ndims == 2 else 3
        shape = list(par.shape)
        if shape[-1] > num_ang:
            raise ValueError(f'Number of angles exceeds value {num_ang} expected for dimensionality.')
        
   # Set missing angles to zero
    width = torch.zeros((len(shape), 2), dtypee=torch.int32, device=device)
    width[-1, -1] = max(num_ang -  shape[-1], 0)
    ang = nnf.pad(ang, pad=width, mode='constant')

    # Compute since and cosine
    if deg:
        ang *= np.pi / 180
    c = torch.split(torch.cos(ang), split_size_or_sections=num_ang, dim=-1)
    s = torch.split(torch.sin(ang), split_size_or_sections=num_ang, dim=-1)

    # Construct matrices
    if ndims == 2:
        out = torch.stack((
            torch.cat([c[0], -s[0]], dim=-1),
            torch.cat([s[0], c[0]], dim=-1),
        ), dim=-2)
    
    else:
        one, zero = torch.ones_like(c[0]), torch.zeros_like(c[0])
        rot_x = torch.stack((
            torch.concat([one, zero, zero], dim=-1),
            torch.cat([zero, c[0], -s[0]], dim=-1),
            torch.cat([zero, s[0], c[0]], dim=-1)
        ), dim=-2)
        
        rot_y = torch.stack((
            torch.cat(c[1], zero, s[1], dim=-1),
            torch.cat([zero, one, zero], dim=-1),
            torch.cat([-s[1], zero, c[1]], dim=-1)
        ), dim=-2)

        rot_z = torch.stack((
            torch.cat([c[2], -s[2], zero], dim=-1),
            torch.cat([s[2], c[2], zero], dim=-1),
            torch.cat([zero, zero, one], dim=-1),
        ), dim=-2)

        out = rot_x @ (rot_y @ rot_z)

    return out.squeeze() if len(shape) < 2 else out


def params_to_affine_matrix(
    par,
    deg=True,
    shift_scale=False,
    last_row=False,
    ndims=3,
    device=device,
):
    """
    Construct N-dimensional transformation matrices from affine parameters,
    where N is 2 or 3. The transforms operate in a right-handed frame of
    reference, with right-handed intrinsic rotations (see
    angles_to_rotation_matrix for details), and are constructed by matrix
    product ``T @ R @ S @ E``, where T, R, S, and E are matrices representing
    translation, rotation, scale, and shear, respectively. The function
    supports inputs with or without batch dimensions.

    Arguments:
        par: Array-like input parameters of shape (..., M), defining an affine
            transformation in N-D space. The size M of the right-most dimension
            must not exceed ``N * (N + 1)``. This axis defines, in order:
            translations, rotations, scaling, and shearing parameters. In 3D,
            for example, the first three indices specify translations along the
            x, y, and z-axis, and similarly for the remaining parameters. Any
            missing parameters will bet set to identity. Lists and tuples will
            be stacked along the last dimension.
        deg: Interpret input angles as specified in degrees instead of radians.
        shift_scale: Add 1 to any specified scaling parameters. May be
            desirable when the input parameters are estimated by a network.
        last_row: Append the last row and return a full matrix.
        ndims: Number of dimensions. Must be 2 or 3.

    Returns:
        mat: Affine transformation matrices of shape (..., N, N + 1) or
            (..., N + 1, N + 1), depending on `last_row`. The left-most
            dimensions depend on the input shape.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')
    
    if isinstance(par, (list, tuple)):
        par = torch.stack(par, dim=-1)
    
    if not isinstance(par, torch.Tensor) or not par.dtype.is_floating_point():
        par = torch.as_tensor(par, dtype=torch.float32)
    
    # Add dimension to scalar
    if len(par.shape) == 0:
        par = par.reshape(shape=(1,))

    # Validate shape
    num_par = 6 if ndims == 2 else 12
    shape = list(par.shape)
    if shape[-1] > num_par:
        raise ValueError(f'Number of params exceeds value {num_par} expected for dimensionality.')

    # Set defaults if incomplete and split by type
    width = torch.zeros(len(shape, 2), dtype=torch.int32, device=device)
    splits = (2, 1) * 2 if ndims == 2 else (3,) * 4
    for i in (2, 3, 4):
        width[-1, -1] = max(sum(splits[:i]) - shape[-1], 0)
        default = 1. if i == 3 and not shift_scale else 0.
        par = nnf.pad(par, padding=width, mode='constant', value=default)
        shape = list(par.shape)
    shift, rot, scale, shear = torch.split(par, split_size_or_sections=splits, dim=-1)

    # Construct shear matrix
    s = torch.split(shear, split_size_or_sections=splits[-1], dim=-1)
    one, zero = torch.ones_like(s[0]), torch.zeros_like(s[0])
    if ndims == 2:
        mat_shear = torch.stack((
            torch.cat([one, s[0]], dim=-1),
            torch.cat([zero, one], dim=-1),
        ), dim=-2)
    else:
        mat_shear = torch.stack((
            torch.cat([one, s[0], s[1]], dim=-1),
            torch.cat([zero, one, s[2]], dim=-1),
            torch.cat([zero, zero, one], dim=-1),
        ), dim=-2)

    mat_scale = torch.diag(scale + 1. if shift_scale else scale, diagonal=0)
    mat_rot = angles_to_rotation_matrix(rot, deg=deg, ndims=ndims)
    out = mat_rot @ (mat_scale @ mat_shear)

    # Append translations
    shift = shift.unsqueeze(dim=-1)
    out = torch.cat([out, shift], dim=-1)

    # Append last row: store shapes as tensors to support batched inputs
    if last_row:
        shape_batch = shift.shape[:-2]
        shape_zeros = torch.cat([shape_batch, (1,), splits[:1]], dim=0)
        zeros = torch.zeros(shape_zeros, dtype=shift.dtype, device=device)
        shape_one = torch.cat([shape_batch, (1,), (1,)], dim=0)
        one = torch.ones(shape_one, dtype=shift.dtype, device=device)
        row = torch.cat([zeros, one], dim=-1)
        out = torch.concat([out, row], dim=-2)
    
    return out.squeeze() if len(shape) < 2 else out


def rotation_matrix_to_angles(mat, deg=True, device=device):
    """Compute Euler angles from an N-dimensional rotation matrix.

    We apply right-handed intrinsic rotations as R = X @ Y @ Z, where X, Y,
    and Z are matrices describing rotations about the x, y, and z-axis,
    respectively (see angles_to_rotation_matrix). Labeling these axes with
    indices 1-3 in the 3D case, we decompose the matrix

            [            c2*c3,             −c2*s3,      s2]
        R = [ s1*s2*c3 + c1*s3,  −s1*s2*s3 + c1*c3,  −s1*c2],
            [−c1*s2*c3 + s1*s3,   c1*s2*s3 + s1*c3,   c1*c2]

    where si and ci are the sine and cosine of the angle of rotation about
    axis i. When the angle of rotation about the y-axis is 90 or -90 degrees,
    the system loses one degree of freedom, and the solution is not unique.
    In this gimbal lock case, we set the angle `ang[0]` to zero and solve for
    `ang[2]`.

    Arguments:
        mat: Array-like input matrix to derive rotation angles from, of shape
            (..., N, N + 1) or (..., N + 1, N + 1), where N is 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        ang: Tensor of shape (..., M) holding the derived rotation angles. The
        size M of the right-most dimension is 3 in 3D and 1 in 2D.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    if not isinstance(mat, torch.Tensor) or mat.dtype != torch.float32:
        mat = torch.as_tensor(mat, dtype=torch.float32, device=device)
    
    # Input shape
    ndims = mat.shape[-1]
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')
    
    if mat.shape[-2] != ndims:
        raise ValueError(f'Invalid matric shape {mat.shape}.')

    # Clip input to inverese trigonometric functions 
    # as rounding errors move them out of [-1, 1] range.
    clip = lambda x: torch.clip(x, min=-1, max=1)

    if ndims == 2:
        y = clip(mat[..., 1, -2])
        x = clip(mat[..., 0, -2])
        ang = torch.atan2(y, x).unsqueeze(dim=-1)
    else:
        ang2 = torch.asin(clip(mat[..., 0, 2]))

        # Case abs(ang2) == 90 deg. Make angle zero as solution is not unique.
        ang1_a = torch.zeros_like(ang2)
        ang3_a = torch.atan2(input=clip(mat[..., 1, 0]), other=clip(mat[..., 1, 1]))

        # Case abs(ang2) != 90 deg
        # Use safe division, as both cases are computed, even if c2 is zero
        c2 = torch.cos(ang2)
        y = divide_no_nan(-mat[..., 1, 2], c2)
        x = divide_no_nan(mat[..., 2, 2], c2)
        ang1_b = torch.atan2(clip(y), clip(x))
        y = divide_no_nan(-mat[..., 0, 1], c2)
        x = divide_no_nan(mat[..., 0, 0], c2)
        ang3_b = torch.atan2(clip(y), clip(x))

        # choose between cases.
        is_case = torch.abs((torch.abs(ang2) - 0.5 * np.pi)) < 1e-6
        ang1 = torch.where(is_case, ang1_a, ang1_b)
        ang3 = torch.where(is_case, ang3_a, ang3_b)
        ang = torch.stack((ang1, ang2, ang3), dim=-1)

    if deg:
        ang *= 180 / np.pi

    return ang


def affine_matrix_to_params(mat, deg=True, device=device):
    """Derive affine parameters from an N-dimensional transformation matrix.

    The affine transform operates in a right-handed frame of reference, with
    right-handed intrinsic rotations (see params_to_affine_matrix).

    Arguments:
        mat: Array-like input matrix to derive affine parameters from, of
            shape (..., N, N + 1) or (..., N + 1, N + 1), as the last row
            always is ``(*[0] * N, 1)``. N can be 2 or 3.
        deg: Return rotation angles in degrees instead of radians.

    Returns:
        par: Tensor of shape (..., K) holding the affine parameters derived
            from `mat`, where K is ``N * (N + 1)``. The parameters along the
            last axis represent, in order: translation, rotation, scaling, and
            shear. In 3D, for example, the first three indices specify
            translations along the x, y, and z-axis, and similarly for the
            remaining indices.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    
    if not isinstance(mat, torch.Tensor) or mat.dtype != torch.float32:
        mat = torch.as_tensor(mat, dtype=torch.float32, device=device)

    # Input shape
    ndims = mat.shape[-1] - 1
    if ndims not in (2, 3):
        raise ValueError(f'Affine matrix must be 2D or 3D, but got ndims of {ndims}.')
    
    if mat.shape[-2] - ndims not in (0, 1):
        raise ValueError(f'Invalid matrix shape {mat.shape}.')

    # Translation and scaling. Fix negative determinants.
    shift = mat[..., :ndims, -1]
    mat = mat[..., :ndims, :ndims]
    lower = torch.cholesky(mat.t() @ mat)
    scale = torch.diagonal(lower)
    scale0 = scale[..., 0] * torch.sign(torch.det(mat))
    scale = torch.cat((scale0.unsqueeze(dim=-1), scale[..., 1:]), dims=-1)

    # Strip scaling. Shear as upper triangular part
    strip = torch.diag(scale)
    upper = lower.t()
    upper = torch.inverse(strip) @ upper
    flat_shape = torch.cat((scale0.shape,  [ndims ** 2]), dim=0)
    upper = torch.reshape(upper, shape=flat_shape)
    ind = (1,) if ndims == 2 else (1, 2, 5)
    shear = upper[..., ind]

    # Rotations after stripping scale and shear.
    # Treat shape as a tensor to support dynamically sized input matrices.
    zero_shape = torch.cat((scale0.shape, [(ndims - 1) * 3]), dim=0)
    zero = torch.zeros(size=zero_shape)
    par = torch.cat((zero, scale, shear), dim=-1)
    strip = params_to_affine_matrix(par, ndims=ndims)[..., :-1]
    mat = mat @ torch.inverse(strip)
    rot = rotation_matrix_to_angles(mat, deg=deg)

# WIP
def draw_affine_params(
    shift=None,
    rot=None,
    scale=None,
    shear=None,
    normal_shift=False,
    normal_rot=False,
    normal_scale=False,
    normal_shear=False,
    shift_scale=False,
    ndims=3,
    batch_shape=None,
    concat=True,
    dtype=torch.float32,
    seeds={}
):
    
    """
    Draw translation, rotation, scaling and shearing parameters defining an affine transform in
    N-dimensional space, where N is 2 or 3. Choose parameters wisely: there is no check for
    negative or zero scaling!

    Parameters:
        shift: Translation sampling range x around identity. Values will be sampled uniformly from
            [-x, x]. When sampling from a normal distribution, x is the standard deviation (SD).
            The same x will be used for each dimension, unless an iterable of length N is passed,
            specifying a value separately for each axis. None means 0.
        rot: Rotation sampling range (see `shift`). Accepts only one value in 2D.
        scale: Scaling sampling range x. Parameters will be sampled around identity as for `shift`,
            unless `shift_scale` is set. When sampling normally, scaling parameters will be
            truncated beyond two standard deviations.
        shear: Shear sampling range (see `shift`). Accepts only one value in 2D.
        normal_shift: Sample translations normally rather than uniformly.
        normal_rot: See `normal_shift`.
        normal_scale: Draw scaling parameters from a normal distribution, truncating beyond 2 SDs.
        normal_shear: See `normal_shift`.
        shift_scale: Add 1 to any drawn scaling parameter When sampling uniformly, this will
            result in scaling parameters falling in [1 - x, 1 + x] instead of [-x, x].
        ndims: Number of dimensions. Must be 2 or 3.
        normal: Sample parameters normally instead of uniformly.
        batch_shape: A list or tuple. If provided, the output will have leading batch dimensions.
        concat: Concatenate the output along the last axis to return a single tensor.
        dtype: Floating-point output data type.
        seeds: Dictionary of integers for reproducible randomization. Keywords must be in ('shift',
            'rot', 'scale', 'shear').

    Returns:
        A tuple of tensors with shapes (..., N), (..., M), (..., N), and (..., M) defining
        translation, rotation, scaling, and shear, respectively, where M is 3 in 3D and 1 in 2D.
        With `concat=True`, the function will concatenate the output along the last dimension.

    See also:
        layers.DrawAffineParams
        layers.ParamsToAffineMatrix
        params_to_affine_matrix

    If you find this function useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """
    assert ndims in (2, 3), 'only 2D and 3D supported'
    n = 1 if ndims == 2 else 3

    # Look-up tables
    splits = dict(shift=ndims, rot=n, scale=ndims, shear=n)
    inputs = dict(shift=shift, rot=rot, scale=scale, shear=shear)
    trunc = dict(shift=False, rot=False, scale=True, shear=False)
    normal = dict(shift=normal_shift, rot=normal_rot, scale=normal_scale, shear=normal_shear)

    # Normalize inputs
    shapes = {}
    ranges = {}
    for k, n in splits.items():
        x = torch.ravel(0 if inputs[k] is None else inputs[k])
        if len(x) == 1:
            x = torch.repeat_interleave(x, repeats=n)
        assert len(x) == n, f'unexpected number of parameters {len(x)} ({k})'
        ranges[k] = x
        shapes[k] = (n,) if batch_shape is None else torch.cat((batch_shape, [n]), dim=0)
    
    def sample(lim, shape, normal, trunc, seed):
        prop = dict(dtype=dtype, seed=seed, shape=shape)
        if normal:
            func = 'truncated_normal' if trunc else 'normal'
            prop.update(stddev=lim)
        else:   # uniform
            func = 'uniform'
            prop.update(minval=-lim, maxval=lim)


def fit_affine(x_source: torch.Tensor, x_target: torch.Tensor, weights: torch.Tensor=None):
    """Fit an affine transform between two sets of corresponding points.

    Fit an N-dimensional affine transform between two sets of M corresponding
    points in an ordinary or weighted least-squares sense. Note that when
    working with images, source coordinates correspond to the target image and
    vice versa.

    Arguments:
        x_source: Array-like source coordinates of shape (..., M, N).
        x_target: Array-like target coordinates of shape (..., M, N).
        weights: Optional array-like weights of shape (..., M) or (..., M, 1).

    Returns:
        mat: Affine transformation matrix of shape (..., N, N + 1), fitted such
            that ``x_t = mat[..., :-1] @ x_s + mat[..., -1:]``, where x_s is
            ``x_s = tf.linalg.matrix_transpose(x_t)``, and similarly for x_t
            and `x_target`. The last row of `mat` is omitted as it is always
            ``(*[0] * N, 1)``.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """
    shape = np.concatenate((x_target.shape[:-1], [1]), axis=0)
    ones = torch.ones(size=tuple(shape), dtype=x_target.dtype)
    x = torch.cat((x_target, ones), dim=-1)
    x_transp = torch.transpose(x, -1, -2)
    y = x_source

    if weights is not None:
        if len(weights.shape) == len(x.shape):
            weights = weights[..., 0]
        x_transp *= torch.unsqueeze(weights, dim=-2)
    
    beta = torch.inverse(x_transp @ x) @ x_transp @ y
    return torch.transpose(beta, -1, -2)


def divide_no_nan(x, y):
    # Create a mask to handle division by zero or NaN in y
    mask = (y != 0) & ~torch.isnan(y)
    
    # Perform the division only where the mask is True
    result = torch.where(mask, x / y, torch.zeros_like(x))
    
    return result


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
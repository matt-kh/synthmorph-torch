from typing import List
import torch
import torch.nn.functional as nnf
import cv2
import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

device = "cuda" if torch.cuda.is_available() else "cpu"

# local code
from . import utils


def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                modulate=None,
                dtype=torch.float32,
                seed=None,
                device=device):
    '''
    Generate Perlin noise by drawing from Gaussian distributions at different
    resolutions, upsampling and summing. 

    Parameters:
        out_shape: List defining the output shape. In N-dimensional space, it
            should have N+1 elements, the last one being the feature dimension.
        scales: List of relative resolutions at which noise is sampled normally.
            A scale of 2 means half resolution relative to the output shape.
        min_std: Minimum standard deviation (SD) for drawing noise volumes.
        max_std: Maximum SD for drawing noise volumes.
        modulate: Boolean. Whether the SD for each scale is drawn from [0, max_std].
            The argument is deprecated: use min_std instead.
        dtype: Output data type.
        seed: Integer for reproducible randomization. This may only have an
            effect if the function is wrapped in a Lambda layer.
    '''
    out_shape_np = np.asarray(out_shape, dtype=np.int32)
    if isinstance(scales, (int)):
        scales = [scales]

    if not modulate:
        min_std = max_std
    if modulate is not None:
        warnings.warn('Argument modulate to ne.utils.augment.draw_perlin is deprecated '
                      'and will be removed in the future. Use min_std instead.')
        
    rand = np.random.default_rng(seed)
    seed = lambda: rand.integers(np.iinfo(int).max).item()
    rng = torch.Generator(device=device).manual_seed(seed())
    
    out = torch.zeros(out_shape, dtype=dtype, device=device)
    for scale in scales:
        sample_shape = np.ceil(out_shape_np[:-1] / scale)
        sample_shape = np.int32((*sample_shape, out_shape_np[-1]))

        std = torch.rand(size=(), dtype=dtype, generator=rng, device=device)
        std = min_std + (max_std - min_std) * std
        gauss = torch.empty(size=tuple(sample_shape), dtype=dtype, device=device).normal_(std=std, generator=rng)

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        out += gauss if scale == 1 else utils.resize(gauss, zoom[:-1])

    # Transform to Torch format
    # indices = list(range(len(out.shape)))
    # out = out.permute(-1, *indices[:-1])

    return out.detach().cpu().numpy()

def draw_perlin_np(out_shape: List[int] = [256, 256, 16],
                   scales: List[int] = [32, 64],
                   min_std=0,
                   max_std=1,
                   dtype=np.float32,
                   seed=None):
    """_summary_

    Args:
        out_shape (List[int], optional): _description_. Defaults to [256, 256, 16].
        scales (List[int], optional): _description_. Defaults to [32, 64].
        min_std (int, optional): _description_. Defaults to 0.
        max_std (int, optional): _description_. Defaults to 1.
        dtype (_type_, optional): _description_. Defaults to np.float32.

    Returns:
        _type_: _description_
    """
    if seed is not None:
        np.random.seed(seed)

    out_shape = np.array(out_shape, dtype=np.int32)
    out = np.zeros(out_shape, dtype=dtype)

    for scale in scales:
        # assume last dimension = channel
        sample_shape = np.ceil(out_shape[:-1] / scale)
        sample_shape = np.int32([*sample_shape, out_shape[-1]])


        std = np.random.uniform(min_std, max_std)
        gauss = np.random.normal(0, std, sample_shape)

        out += resize_np(gauss, out_shape)

    return out


def compute_points(src: List[int], target: List[int] = None):
    if target is None:
        target = src

    assert len(src) == len(target), "len(src) != len(target)"
    lins = []
    for s, t in zip(src, target):
        l = np.linspace(0., s - 1., t)
        lins.append(l)

    return lins


def preprocess_interpn(vol: np.ndarray):
    for i, s in enumerate(vol.shape):
        if s == 1:
            vol = np.concatenate([vol, vol], axis=i)

    return vol

def resize_np(vol: np.ndarray,
              target_shapes: List[int],
              interp_method='linear'):
    assert len(target_shapes) == len(vol.shape), "target_shapes must match len(vol.shape)"
    vol = preprocess_interpn(vol)

    grids = compute_points(vol.shape, target_shapes)
    out_grids = np.meshgrid(*grids, indexing='ij')
    out_grids = np.stack(out_grids, -1)

    points = compute_points(vol.shape)
    inter = interpn(points, vol, out_grids, method=interp_method)

    return inter


def transform(vol: np.ndarray,
              loc_shift: np.ndarray,
              interp_method='linear',
              fill_value=None):
    original_shape = vol.shape

    vol = preprocess_interpn(vol)
    loc_shift = preprocess_interpn(loc_shift)

    points = compute_points(vol.shape)
    out_grids = np.meshgrid(*points, indexing='ij')
    out_grids = np.stack(out_grids, -1)

    cross_channel = np.zeros([*out_grids.shape[:-1], 1])
    loc_shift = np.concatenate([loc_shift, cross_channel], axis=len(out_grids.shape) -1)

    # breakpoint()
    out_grids = out_grids + loc_shift

    inter = interpn(points, vol, out_grids, method=interp_method,
                    # there is a possibility for out_grid to be out of bound.
                    # at this point, we need to extrapolate the value.
                    bounds_error=False, fill_value=fill_value)

    index = [slice(None)] * len(original_shape) 
    for i, s in enumerate(original_shape):
        if s == 1:
            index[i] = 0

    inter = inter[tuple(index)]
    return inter

def generate_map(size: List[int] = [256, 256],
                 nLabel: int = 16,
                 random=np.random.RandomState(None),
                 device=device
):

    seed1 = random.randint(0, 2**31 - 1)
    seed2 = random.randint(0, 2**31 - 1)

    num_dim = len(size)
    out = draw_perlin([*size, nLabel], [32, 64], max_std=1, seed=seed1, device=device)
    warp = draw_perlin([*size, nLabel, num_dim], [16, 32, 64], max_std=16, seed=seed2, device=device)
    out = torch.tensor(out, device=device)
    warp = torch.tensor(warp, device=device)
    # out_indices = np.arange(len(out.shape))
    # warp_indices = np.arange(len(warp.shape))
    # out = out.permute(*out_indices[1:], 0)
    # warp = warp.permute(*warp_indices[1: -1], -1, 0)
    deform = utils.transform(out, warp)

    map = torch.argmax(deform, dim=-1)
    return map.to(torch.uint8).detach().cpu().numpy()


def generate_map_np(size: List[int] = [256, 256],
                 nLabel: int = 16,
                 random=np.random.RandomState(None)):

    seed1 = random.randint(0, 2**31 - 1)
    seed2 = random.randint(0, 2**31 - 1)

    num_dim = len(size)
    out = draw_perlin([*size, nLabel], [32, 64], max_std=1, seed=seed1)
    warp = draw_perlin([*size, nLabel, num_dim], [16, 32, 64], max_std=16, seed=seed2)

    deform = transform(out, warp)

    map = np.argmax(deform, axis=-1)
    return np.uint8(map)


def vec_intergral(vec: np.ndarray, nsteps=5):
    scale = 1.0 / 2 ** nsteps
    vec = vec * scale
    for _ in range(nsteps):
        vec_field = np.expand_dims(vec, -1)
        vec_field = np.repeat(vec_field, vec.shape[-1], axis=-1)
        vec = vec + transform(vec, vec_field)

    return vec


def map_to_image(label_map: np.ndarray):
    rand_gen = np.random.default_rng()

    num_dim = len(label_map.shape)
    shape = np.array(label_map.shape)
   
    warp_shape = [*(shape // 2), num_dim]
    warp_res =  np.int32(np.array([8, 16, 32]) / 2)
    warp_field = draw_perlin(warp_shape, warp_res, 
                             min_std=3, max_std=2)

    warp_field = vec_intergral(warp_field, nsteps=5)
    warp_field = warp_field * 2
    warp_field = cv2.resize(warp_field, label_map.shape)

    label_map = np.expand_dims(label_map, 2)
    warp_field = np.expand_dims(warp_field, 2)
    label_map = transform(label_map, warp_field, interp_method='nearest')

    label_map = label_map.astype(np.int32)
    out = random_label(label_map)

    # randomly zero out some background
    rand_flip = rand_gen.uniform(low=0, high=1, size=out.shape)
    rand_flip = rand_flip < 0.2 ** 16
    out *= 1 - rand_flip
    
    out = gaussian_filter(out, 1)   # gaussian blur

    bias_field = draw_perlin([*shape, 1],
                             scales=[40],
                             min_std=0.3, max_std=0.3)
    bias_field = np.squeeze(bias_field, -1)
    out *= np.exp(bias_field)

    # Intensity manipulation
    out = np.clip(out, 0, 255)
    out = minmax_norm(out)
    gamma = rand_gen.normal(loc=0.0, scale=0.25, size=out.shape)
    out = np.power(out, np.exp(gamma))
   
    out = np.expand_dims(out, -1)

    return label_map, out


def random_label(label_map, low_mean=25, max_mean=255, low_std=5, max_std=25,
                random=np.random.RandomState(None)):
    out = random.uniform(size=label_map.shape)

    for j in np.unique(label_map):
        mean = random.uniform(low_mean, max_mean)
        std = random.uniform(low_std, max_std)

        out[label_map == j] = out[label_map == j] * std + mean

    return out


def minmax_norm(x, axis=None):
    """
    Min-max normalize tensor using a safe division.
    Arguments:
        x: Tensor to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.
    Returns:
        Normalized tensor.
    """
    x_min = np.min(x, axis=axis, keepdims=True)
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.divide(x - x_min, x_max - x_min, out=np.zeros_like(x), where=(x_max - x_min) != 0)


def conform(x, in_shape, device):
    x = x.astype(np.float32)
    x = x.squeeze()
    x = minmax_norm(x)
    x = torch.from_numpy(x)
    x = utils.resize(x, zoom_factor=[o / i for o, i in zip(in_shape, x.shape)])
    x = x.view(1, *in_shape, 1)
    return x.to(device)


def post_predict(x):
    x = torch.squeeze(x)
    x = x.cpu()
    x = x.detach().numpy()
    return x


class SynthMorphDataset(Dataset):
    def __init__(
        self,
        size: int,
        input_size=(256,256),
        num_labels: int=26,
    ):
        self.input_size = input_size
        self.num_labels = num_labels
        self.size = size
        self.label_maps = [self.generate_label() for _ in tqdm(range(size))]

    def generate_label(self):
        label_map = generate_map(
                size=self.input_size,
                nLabel=self.num_labels,
            )

        return label_map
    
    def prepare_data(self):
        pass

    def __len__(self):
        return self.size

    def __getitem__(self, index: int):
 
        label = self.label_maps[index]      

        fixed_map, fixed = map_to_image(label)
        moving_map, moving =  map_to_image(label)
        

        fixed = torch.tensor(fixed, device=device).permute(2, 0, 1)
        moving = torch.tensor(moving, device=device).permute(2, 0, 1)
        fixed_map = torch.tensor(fixed_map, dtype=torch.int64, device=device),
        moving_map = torch.tensor(moving_map, dtype=torch.int64, device=device)
    
        # preprocess label map
        fixed_map = nnf.one_hot(fixed_map[0], num_classes=self.num_labels)
        fixed_map = fixed_map.permute(2, 0, 1)
        fixed_map = fixed_map.contiguous()
        
        moving_map = nnf.one_hot(moving_map, num_classes=self.num_labels)
        moving_map = moving_map.permute(2, 0, 1)
        moving_map = moving_map.contiguous()

        results = {
            "fixed": fixed.to(torch.float32),
            "moving": moving.to(torch.float32),
            "fixed_map": fixed_map,
            "moving_map": moving_map
        }
        
        return results

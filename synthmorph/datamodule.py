import warnings
from typing import List
import torch
import torch.nn.functional as nnf
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

# local code
from . import utils
from . import layers

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    indices = list(range(len(out.shape)))
    out = out.permute(-1, *indices[:-1])

    return out


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
    
    out_indices = np.arange(len(out.shape))
    warp_indices = np.arange(len(warp.shape))
    out = out.permute(*out_indices[1:], 0)
    warp = warp.permute(*warp_indices[1: -1], -1, 0)
    deform = utils.transform(out, warp)

    map = torch.argmax(deform, dim=-1)
    return map.to(torch.uint8)


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
    
    if axis == None:
        # Treated as fattened, 1D tensor
        torchmin = lambda x: torch.min(x)
        torchmax = lambda x: torch.max(x)
    else:
        # Operates on specified axis, and maintain shape
        torchmin = lambda x: torch.min(x, dim=axis, keepdim=True).values
        torchmax = lambda x: torch.max(x, dim=axis, keepdim=True).values

    x_min = torchmin(x)
    x_max = torchmax(x)
    result = torch.where((x_max - x_min) != 0, (x - x_min) / (x_max - x_min), torch.zeros_like(x))
    return result


def conform(x, in_shape, device):
    x = x.astype(np.float32)
    x = x.squeeze()
    x = torch.from_numpy(x)
    x = minmax_norm(x)
    x = utils.resize(x, zoom_factor=[o / i for o, i in zip(in_shape, x.shape)])
    x = x.view(1, *in_shape, 1)
    return x.to(device)


def post_predict(x):
    x = torch.squeeze(x)
    x = x.cpu()
    x = x.detach().numpy()
    return x

    
def labels_to_image(
    labels,
    out_label_list=None,
    out_shape=None,
    num_chan=1,
    mean_min=None,
    mean_max=None,
    std_min=None,
    std_max=None,
    zero_background=0.2,
    warp_res=[16],
    warp_std=0.5,
    warp_modulate=True,
    bias_res=40,
    bias_std=0.3,
    bias_modulate=True,
    blur_std=1,
    blur_modulate=True,
    normalize=True,
    gamma_std=0.25,
    dc_offset=0,
    one_hot=True,
    seeds={},
    return_vel=False,
    return_def=False,
    id=0,
    device=device
):
    """
    Augment label maps and synthesize images from them.

    Parameters:
        out_label_list (optional): List of labels in the output label maps. If
            a dictionary is passed, it will be used to convert labels, e.g. to
            GM, WM and CSF. All labels not included will be converted to
            background with value 0. If 0 is among the output labels, it will be
            one-hot encoded. Defaults to the input labels.
        out_shape (optional): List of the spatial dimensions of the outputs.
            Inputs will be symmetrically cropped or zero-padded to fit.
            Defaults to the input shape.
        num_chan (optional): Number of image channels to be synthesized.
            Defaults to 1.
        mean_min (optional): List of lower bounds on the means drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            25 for all other labels.
        mean_max (optional): List of upper bounds on the means drawn to generate
            the intensities for each label. Defaults to 225 for each label.
        std_min (optional): List of lower bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 0 for the background and
            5 for all other labels.
        std_max (optional): List of upper bounds on the SDs drawn to generate
            the intensities for each label. Defaults to 25 for each label.
            25 for all other labels.
        zero_background (float, optional): Probability that the background is set
            to zero. Defaults to 0.2.
        warp_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the SVF is drawn.
            Defaults to 16.
        warp_std (float, optional): Upper bound on the SDs used when drawing
            the SVF. Defaults to 0.5.
        warp_modulate (bool, optional): Whether to draw the SVF with random SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        bias_res (optional): List of factors N determining the
            resultion 1/N relative to the inputs at which the bias field is
            drawn. Defaults to 40.
        bias_std (float, optional): Upper bound on the SDs used when drawing
            the bias field. Defaults to 0.3.
        bias_modulate (bool, optional): Whether to draw the bias field with
            random SDs. If disabled, each batch will use the maximum SD.
            Defaults to True.
        blur_std (float, optional): Upper bound on the SD of the kernel used
            for Gaussian image blurring. Defaults to 1.
        blur_modulate (bool, optional): Whether to draw random blurring SDs.
            If disabled, each batch will use the maximum SD. Defaults to True.
        normalize (bool, optional): Whether the image is min-max normalized.
            Defaults to True.
        gamma_std (float, optional): SD of random global intensity
            exponentiation, i.e. gamma augmentation. Defaults to 0.25.
        dc_offset (float, optional): Upper bound on global DC offset drawn and
            added to the image after normalization. Defaults to 0.
        one_hot (bool, optional): Whether output label maps are one-hot encoded.
            Only the specified output labels will be included. Defaults to True.
        seeds (dictionary, optional): Integers for reproducible randomization.
        return_vel (bool, optional): Whether to append the half-resolution SVF
            to the model outputs. Defaults to False.
        return_def (bool, optional): Whether to append the combined displacement
            field to the model outputs. Defaults to False.
        id (int, optional): Model identifier used to create unique layer names
            for including this model multiple times. Defaults to 0.
    """
    np_rng = np.random.default_rng(None)
    default_seed = lambda: np_rng.integers(np.iinfo(int).max).item()
    rng = lambda x : torch.Generator(device=device).manual_seed(x)

    
    num_dim = len(labels.shape)
    in_shape = labels.shape
    if out_shape is None:
        out_shape = in_shape
    in_shape, out_shape = map(np.asarray, (in_shape, out_shape))
    batch_size = in_shape[0] if num_dim > 2 else 1
    # Reshape
    if num_dim > 2:
        labels = labels.view(batch_size, num_chan, *in_shape[1:-1])
    else:
        labels = labels.view(batch_size, num_chan, *in_shape)

    # Transform labels into [0, 1, ..., N-1].
    labels = labels.to(dtype=torch.int32, device=device)
    in_label_list = labels.unique()
    num_in_labels = len(in_label_list)

    in_lut = torch.zeros(size=(torch.max(in_label_list) + 1,), dtype=torch.int32, device=device)
    for i, lab in enumerate(in_label_list):
        in_lut[lab] = i
    labels = in_lut[labels] # tf.gather(in_lut, indices=labels)

    if warp_std > 0:
        # Velocity field.
        vel_shape = (*out_shape // 2, num_dim)
        vel_scale = np.asarray(warp_res) / 2
        vel_draw = lambda x: draw_perlin(
            vel_shape, scales=vel_scale,
            min_std=0 if warp_modulate else warp_std, max_std=warp_std,
            seed=seeds.get('warp')
        )
        # One per batch.
        vel_field = torch.stack([vel_draw(l) for l in labels])
        # Deformation field.
        def_field = layers.VecInt(int_steps=5)(vel_field)
        def_field = layers.RescaleValues(2)(def_field)
        def_field = layers.Resize(2, interp_method='linear')(def_field)
        # Resampling.
        labels = layers.SpatialTransformer(interp_method='nearest', fill_value=0)([labels, def_field])

    labels = labels.to(torch.int32)
    # Intensity means and standard deviations.
    if mean_min is None:
        mean_min = [0] + [25] * (num_in_labels - 1)
    if mean_max is None:
        mean_max = [225] * num_in_labels
    if std_min is None:
        std_min = [0] + [5] * (num_in_labels - 1)
    if std_max is None:
        std_max = [25] * num_in_labels
    as_torch_tensor = lambda x: torch.as_tensor(x, device=device)
    m0, m1, s0, s1 = map(as_torch_tensor, (mean_min, mean_max, std_min, std_max))

    mean = torch.rand(
        size=(batch_size, num_chan, num_in_labels),
        generator=rng(seeds.get('mean', default_seed())),
        device=device,
    )
    mean = m0 + (m1 - m0) * mean

    std = torch.rand(
        size=(batch_size, num_chan, num_in_labels),
        generator=rng(seeds.get('std', default_seed())),
        device=device,
    )
    std = s0 + (s1 - s0) * std

    # Synthetic image.
    image = torch.empty(size=labels.shape, device=device).normal_(generator=rng(seeds.get('noise', default_seed())))
    indices = torch.concat([labels + i * num_in_labels for i in range(num_chan)], dim=-1)
    gather = lambda x: torch.reshape(x[0], (-1,))[x[1]]
    mean = gather([mean, indices])
    std = gather([std, indices])
    image = image * std + mean

    # Zero background.
    if zero_background > 0:
        rand_flip = torch.rand(
            size=(batch_size, *[1] * num_dim, num_chan),
            generator=rng(seeds.get('background', default_seed())),
            device=device,
        )
        rand_flip = torch.lt(rand_flip, zero_background)    # tf.less
        image *= 1. - torch.logical_and(labels == 0, rand_flip).to(image.dtype)

    # Blur.
    if blur_std > 0:
        kernels = utils.gaussian_kernel(
            [blur_std] * num_dim, separate=True, random=blur_modulate,
            dtype=image.dtype, seed=seeds.get('blur'),
        )
        image = utils.separable_conv(image, kernels, batched=True)

    # Bias field.
    if bias_std > 0:
        bias_shape = (*out_shape, 1)
        bias_draw = lambda x: draw_perlin(
            bias_shape, scales=bias_res, seed=seeds.get('bias'),
            min_std=0 if bias_modulate else bias_std, 
            max_std=bias_std, device=device
        )
        bias_field = torch.stack([bias_draw(l) for l in labels])
        image *= torch.exp(bias_field)

    # Intensity manipulations.
    image = torch.clip(image, min=0, max=255)
    if normalize:
        image = torch.stack([minmax_norm(batch) for batch in image])  
    if gamma_std > 0:
        gamma = torch.empty(size=(batch_size, *[1] * num_dim, num_chan), device=device)
        gamma = gamma.normal_(std=gamma_std, generator=rng(seeds.get('gamma', default_seed())))
        image = torch.pow(image, torch.exp(gamma))
    if dc_offset > 0:
        offset = torch.rand(
            size=(batch_size, *[1] * num_dim, num_chan),
            generator=(rng(seeds.get('dc_offset'), default_seed())),
            device=device
        )
        offset *= dc_offset
        image += offset

    # Lookup table for converting the index labels back to the original values,
    # setting unwanted labels to background. If the output labels are provided
    # as a dictionary, it can be used e.g. to convert labels to GM, WM, CSF.
    if out_label_list is None:
        out_label_list = in_label_list
    if isinstance(out_label_list, (tuple, list, torch.Tensor)):
        out_label_list = {lab.item(): lab for lab in out_label_list}

    out_lut = torch.zeros((num_in_labels,), dtype=torch.int32)
    for i, lab in enumerate(in_label_list):
        if lab.item() in out_label_list:
            out_lut[i] = out_label_list[lab.item()]

    # For one-hot encoding, update the lookup table such that the M desired
    # output labels are rebased into the interval [0, M-1[. If the background
    # with value 0 is not part of the output labels, set it to -1 to remove it
    # from the one-hot maps.
    if one_hot:
        hot_label_list = torch.tensor(list(out_label_list.values())).unique() # Sorted.
        hot_lut = torch.full((hot_label_list[-1] + 1,), fill_value=-1, dtype=torch.int32, device=device)
        for i, lab in enumerate(hot_label_list):
            hot_lut[lab] = i
        out_lut = hot_lut[out_lut]

    # Convert indices to output labels only once.
    labels = out_lut[labels]
    if one_hot:
        labels = nnf.one_hot(labels.to(torch.int64), num_classes=len(hot_label_list))
        labels = labels.squeeze(1).permute(0, 3, 1, 2)

    outputs = {'image': image, 'label': labels}
    if return_vel:
        outputs['vel'] = vel_field
    if return_def:
        outputs['def'] =  def_field

    return outputs


class SynthMorphDataset(Dataset):
    def __init__(
        self,
        size: int,
        gen_arg: dict,
        input_size=(256,256),
        num_labels: int=26,
        **kwargs

    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.num_labels = num_labels
        self.size = size
        self.label_maps = [self._generate_label() for _ in tqdm(range(size))]
        self.gen_arg = gen_arg

    def _generate_label(self):
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

        fixed = labels_to_image(label, **self.gen_arg)
        moving = labels_to_image(label, **self.gen_arg)
        fixed_image, fixed_map = fixed['image'], fixed['label']
        moving_image, moving_map = moving['image'], moving['label']

        results = {
            "fixed": fixed_image.to(torch.float32),
            "moving": moving_image.to(torch.float32),
            "fixed_map": fixed_map.to(torch.int64),
            "moving_map": moving_map.to(torch.int64)
        }
        
        return results

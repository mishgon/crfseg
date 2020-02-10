import numpy as np
import torch
from itertools import product


def unfold(x, window_size, fill_value=np.nan):
    """
    Parameters
    ----------
    x : torch.tensor or np.ndarray
        Tensor/array of shape ``(..., *spatial)``.
    window_size : np.ndarray with ``len(spatial)`` ints
    fill_value : float
        Value to fill empty positions in the filters for border voxels. Default is Nan.

    Returns
    -------
    output : torch.tensor or np.ndarray
        Tensor/array of shape ``(..., *spatial, *filter_size)`` with nearest ``filter_size`` voxels for each voxel.
    """
    if isinstance(x, torch.Tensor):
        output = torch.full((*x.shape, *window_size), fill_value).to(x)
    elif isinstance(x, np.ndarray):
        output = np.full((*x.shape, *window_size), fill_value)
    else:
        raise TypeError('``x`` must be of type torch.Tensor or np.ndarray')

    def get_source_slice(shift):
        if shift > 0:
            return slice(0, -shift)
        elif shift < 0:
            return slice(-shift, None)
        else:
            return slice(0, None)

    def get_shifted_slice(shift):
        return get_source_slice(-shift)

    for shift in product(*[np.arange(-(fs // 2), fs // 2 + 1) for fs in window_size]):
        source_slice = tuple(map(get_source_slice, shift))
        shifted_slice = tuple(map(get_shifted_slice, shift))
        output[(...,) + source_slice + tuple(shift + window_size // 2)] = x[(...,) + shifted_slice]

    return output


def to_np(tensor):
    return tensor.data.cpu().numpy()

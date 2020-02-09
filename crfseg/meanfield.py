from itertools import product
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class MeanFieldCRF(nn.Module):
    """
    Class for learning and inference in conditional random field model using mean field approximation and convolutional approximation in pairwise potentials term.

    Parameters
    ----------
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing. If it is a sequence its length must be equal to the number of spatial dimensions of input tensors.
    n_iter : int
        Number of iterations in mean field approximation.
    n_classes_to_vectorize : int or None
        The number of classes which are processed in vectorized manner. Default is None, which means all classes. Large ``n_classes_to_vectorize`` leads to faster but more storage-consuming computations.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    return_log_proba : bool
        Whether to return log-probabilities (which is more computationally stable if then the nn.NLLLoss is used), or probabilities.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    appearance_weight : float
        Initial weight of appearance kernel.
    spatial_bandwidth : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel. If it is a sequence its length must be equal to the number of spatial dimensions of input tensors.
    bilateral_bandwidth : float or sequence of floats
        Initial bandwidths for each feature (spatial and another) in the gaussian bilateral kernel. If it is a sequence its length must be equal to 1 + the number of spatial dimensions of input tensors.
    compatibility_bandwidth : float
        Initial bandwidth for the gaussian compatibility function.
    """
    def __init__(self, filter_size=11, n_iter=5, n_classes_to_vectorize=None, requires_grad=True, return_log_proba=True,
                 smoothness_weight=1, appearance_weight=1, spatial_bandwidth=1, bilateral_bandwidth=1,
                 compatibility_bandwidth=1):
        super().__init__()
        self.n_iter = n_iter
        self.filter_size = filter_size
        self.n_classes_to_vectorize = n_classes_to_vectorize
        self.return_log_proba = return_log_proba

        def add_param(init_value):
            return Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=requires_grad))

        self.smoothness_weight = add_param(smoothness_weight)
        self.appearance_weight = add_param(appearance_weight)
        self.inverse_spatial_bandwidth = add_param(1 / np.asarray(spatial_bandwidth))
        self.inverse_bilateral_bandwidth = add_param(1 / np.asarray(bilateral_bandwidth))
        self.inverse_compatibility_bandwidth = add_param(1 / np.asarray(compatibility_bandwidth))

    def forward(self, x, spatial_spacings=None):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``. None is equivalent to all ones. Used for create adaptive spatial gaussian filters.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        filter_size = np.broadcast_to(self.filter_size, len(spatial))
        if spatial_spacings is None:
            spatial_spacings = torch.full((batch_size, len(spatial)), 1)

        for i in range(self.n_iter):
            # Normalizing
            output = F.softmax(x, dim=1)

            # Message Passing
            output = self._pad(output, filter_size)
            filter_ = self._create_gaussian_filter(filter_size, self.inverse_spatial_bandwidth, spatial_spacings).to(output)
            smoothness_filter_output = self._convolve_channelwisely(output, filter_)
            smoothness_filter_output = self._unpad(smoothness_filter_output, filter_size)

            # Weighting Filter Outputs
            output = self.smoothness_weight * smoothness_filter_output

            # Compatibility Transform
            output = self._compatibility_transform(output, self.inverse_compatibility_bandwidth)

            # Adding Unary Potentials
            output = x - output

        return F.log_softmax(output, dim=1) if self.return_log_proba else F.softmax(output, dim=1)

    # TODO: add support of different spatial spacings
    def compute_energy(self, state, unary):
        """
        Parameters
        ----------
        state : np.ndarray of ints
            Array of shape ``spatial`` with classes.
        unary : np.ndarray
            Array of shape ``(n_classes, *spatial)`` with negative unary potentials, e.g. logits.

        Returns
        -------
        energy : float
        """
        filter_size = np.broadcast_to(self.filter_size, len(state.shape))
        assert len(filter_size) <= 3
        ij = 'ijk'[:len(filter_size)]

        state = state[np.newaxis, np.newaxis]
        diff = state[(..., *len(filter_size) * [np.newaxis])] - self._unfold(state, filter_size, fill_value=np.nan)
        unfolded_compatibility = -np.exp(-diff ** 2 * self.inverse_compatibility_bandwidth.item() ** 2 / 2)
        unfolded_compatibility = np.nan_to_num(unfolded_compatibility)

        filter_ = (self.smoothness_weight * self._create_gaussian_filter(filter_size, self.inverse_spatial_bandwidth)
                   ).data.cpu().numpy()
        pairwise_term = np.sum(np.einsum(f'...{ij}, {ij}', unfolded_compatibility, filter_))

        return np.sum(np.take_along_axis(unary, state[0], 0)) + pairwise_term

    @staticmethod
    def _pad(x, filter_size):
        padding = []
        for fs in filter_size:
            padding += 2 * [fs // 2]

        return F.pad(x, list(reversed(padding)))  # F.pad pads from the end

    @staticmethod
    def _unpad(x, filter_size):
        return x[(..., *[slice(fs // 2, -(fs // 2)) for fs in filter_size])]

    @staticmethod
    def _create_gaussian_filter(filter_size, inverse_bandwidth, spatial_spacings):
        """
        Parameters
        ----------
        filter_size : sequence of ints
        inverse_bandwidth : torch.nn.parameter.Parameter containing 1 or ``len(filter_size)`` floats
        spatial_spacings : torch.tensor of shape ``(batch_size, len(filter_size))``.

        Returns
        -------
        filter_ : torch.tensor
            Gaussian filter of shape ``(batch_size, filter_size)``
        """
        distances = torch.stack(torch.meshgrid([torch.arange(-(fs // 2), fs // 2 + 1) for fs in filter_size]))
        distances = distances * spatial_spacings[(..., *len(filter_size) * [None])].to(distances)
        scaled_distances = distances.to(inverse_bandwidth) * inverse_bandwidth.view(-1, *len(filter_size) * [1])
        filter_ = torch.exp(-torch.sum(scaled_distances ** 2 / 2, dim=1))

        zero_center = torch.ones(tuple(filter_size)).to(filter_)
        zero_center[tuple([fs // 2 for fs in filter_size])] = 0
        filter_ = zero_center * filter_

        return filter_

    @staticmethod
    def _convolve_channelwisely(x, filter_):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)``.
        filter_ : torch.tensor
            Tensor of shape ``(batch_size, filter_size)``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)``.
        """
        conv = [F.conv1d, F.conv2d, F.conv3d][len(x.shape) - 3]
        padding = [fs // 2 for fs in filter_.shape[1:]]

        return torch.stack([conv(x_[0, :, None], f[None], padding=padding)
                            for x_, f in zip(torch.split(x, 1, 0), torch.split(filter_, 1, 0))]).squeeze(2)

    @staticmethod
    def _pass_message(x, filters, n_classes_to_vectorize=None):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        filters : torch.tensor
            Tensor of shape ``(batch_size, *spatial, *filter_size)``.
        n_classes_to_vectorize : int or None
            The number of classes which are processed in parallel. Default is None, which means all classes. Large ``n_classes_to_vectorize`` leads to faster but more storage-consuming computations.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        filter_size = np.array(filters.shape[len(x.shape) - 1:])
        assert len(filter_size) <= 3
        ij = 'ijk'[:len(filter_size)]

        if n_classes_to_vectorize is None:
            return torch.einsum(f'...{ij}, ...{ij}', *torch.broadcast_tensors(
                MeanFieldCRF._unfold(x, filter_size, fill_value=0), filters.unsqueeze(1)
            ))
        else:
            return torch.cat([
                torch.einsum(f'...{ij}, ...{ij}', *torch.broadcast_tensors(
                    MeanFieldCRF._unfold(x_, filter_size, fill_value=0), filters.unsqueeze(1)
                )) for x_ in torch.split(x, n_classes_to_vectorize, 1)
            ], dim=1)

    # TODO: add support of different spatial spacings
    @staticmethod
    def _create_gaussian_filters(feature, filter_size, inverse_bandwidth):
        """
        Parameters
        ----------
        feature : torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)``.
        filter_size : np.ndarray with ``len(spatial)`` ints
        inverse_bandwidth : torch.nn.parameter.Parameter containing 1 or ``n_channels`` floats

        Returns
        -------
        filters : torch.tensor
            Tensor of shape ``(batch_size, *spatial, *filter_size)``. Contains trash (not Nans) in the positions where filters' values are not defined.
        """
        diffs = feature[(..., *len(filter_size) * [None])] - MeanFieldCRF._unfold(feature, filter_size, fill_value=0)
        scaled_diffs = diffs * inverse_bandwidth.view(-1, *(len(feature.shape[2:]) + len(filter_size)) * [1])
        filters = torch.exp(-torch.sum(scaled_diffs ** 2 / 2, dim=1))

        # in-place operation ``filters[(..., *filter_size // 2)] = 0`` does not allow
        # to compute gradients w.r.t. ``inverse_bandwidth``
        zero_centers = torch.ones(filters.shape).to(filters)
        zero_centers[(..., *filter_size // 2)] = 0
        filters = zero_centers * filters

        return filters

    @staticmethod
    def _unfold(x, filter_size, fill_value=np.nan):
        """
        Parameters
        ----------
        x : torch.tensor or np.ndarray
            Tensor/array of shape ``(batch_size, n_channels, *spatial)``.
        filter_size : np.ndarray with ``len(spatial)`` ints
        fill_value : float
            Value to fill empty positions in the filters for border voxels. Default is Nan.

        Returns
        -------
        output : torch.tensor or np.ndarray
            Tensor/array of shape ``(batch_size, n_channels, *spatial, *filter_size)`` with nearest ``filter_size`` voxels for each voxel.
        """
        if isinstance(x, torch.Tensor):
            output = torch.full((*x.shape, *filter_size), fill_value).to(x)
        elif isinstance(x, np.ndarray):
            output = np.full((*x.shape, *filter_size), fill_value)
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

        for shift in product(*[np.arange(-(fs // 2), fs // 2 + 1) for fs in filter_size]):
            source_slice = tuple(map(get_source_slice, shift))
            shifted_slice = tuple(map(get_shifted_slice, shift))
            output[(...,) + source_slice + tuple(shift + filter_size // 2)] = x[(...,) + shifted_slice]

        return output

    @staticmethod
    def _compatibility_transform(x, inverse_bandwidth):
        """
        Parameters
        ----------
        x : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        inverse_bandwidth : torch.nn.parameter.Parameter

        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        diff = (torch.arange(x.shape[1]) - torch.arange(x.shape[1]).unsqueeze(1)).to(x)
        compatibility_matrix = -torch.exp(-diff ** 2 * inverse_bandwidth ** 2 / 2)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)


class CircularCRF(MeanFieldCRF):
    def __init__(self, axes, **kwargs):
        super().__init__(**kwargs)
        self.axes = np.asarray(axes)  # 0 is first spatial dimension

    def _pad(self, x, filter_size):
        circular_padding, replicate_padding = [], []
        for i, fs in enumerate(filter_size):
            crc, rpl = (2 * [fs // 2], 2 * [0]) if i in self.axes else (2 * [0], 2 * [fs // 2])
            circular_padding += crc
            replicate_padding += rpl

        x = F.pad(x, list(reversed(circular_padding)), mode='circular')
        x = F.pad(x, list(reversed(replicate_padding)), mode='replicate')
        return x




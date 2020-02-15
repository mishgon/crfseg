import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .utils import unfold, to_np


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
    smoothing_weight : float
        Initial weight of smoothness kernel.
    appearance_weight : float
        Initial weight of appearance kernel.
    smoothing_bandwidth : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel. If it is a sequence its length must be equal to the number of spatial dimensions of input tensors.
    appearance_bandwidth : float
        Initial bandwidth of adaptive gaussian kernel.
    """
    def __init__(self, filter_size=11, n_iter=5, n_classes_to_vectorize=None, requires_grad=True, return_log_proba=True,
                 smoothing_weight=1, appearance_weight=1, smoothing_bandwidth=1, appearance_bandwidth=1):
        super().__init__()
        self.n_iter = n_iter
        self.filter_size = filter_size
        self.n_classes_to_vectorize = n_classes_to_vectorize
        self.return_log_proba = return_log_proba
        self.requires_grad = requires_grad

        self._set_param('smoothing_weight', smoothing_weight)
        self._set_param('appearance_weight', appearance_weight)
        self._set_param('inverse_smoothing_bandwidth', 1 / np.asarray(smoothing_bandwidth))
        self._set_param('inverse_appearance_bandwidth', 1 / np.asarray(appearance_bandwidth))

    def _set_param(self, name, init_value):
        setattr(self, name, Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, features=None, spatial_spacings=None):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        features : torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)`` with features for creating a bilateral kernel.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``. None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        filter_size = np.broadcast_to(self.filter_size, len(spatial))
        if spatial_spacings is None:
            spatial_spacings = torch.full((batch_size, len(spatial)), 1)

        negative_unary = x.clone()
        if features is not None:
            features = self._pad(features, filter_size)

        for i in range(self.n_iter):
            # Normalizing
            x = F.softmax(x, dim=1)

            # Message Passing
            x = self._pad(x, filter_size)

            smoothing_filter = self._smoothing_filter(filter_size, self.inverse_smoothing_bandwidth, spatial_spacings).to(x)
            smoothing_filter_output = self._convolve_classwisely(x, smoothing_filter)
            smoothing_filter_output = self._unpad(smoothing_filter_output, filter_size)

            if features is not None:
                adaptive_filter = self._adaptive_filter(features, filter_size, self.inverse_appearance_bandwidth).to(x)
                adaptive_filter = self._make_broadcastable_to_adaptive(smoothing_filter) * adaptive_filter
                adaptive_filter_output = self._pass_message(x, adaptive_filter, self.n_classes_to_vectorize)
                adaptive_filter_output = self._unpad(adaptive_filter_output, filter_size)

            # Weighting Filter Outputs
            x = self.smoothing_weight * smoothing_filter_output
            if features is not None:
                x = x + self.appearance_weight * adaptive_filter_output

            # Compatibility Transform
            x = self._compatibility_transform(x)

            # Adding Unary Potentials
            x = negative_unary - x

        return F.log_softmax(x, dim=1) if self.return_log_proba else F.softmax(x, dim=1)

    def compute_energy(self, state, unary, features=None, spatial_spacing=None):
        """
        Parameters
        ----------
        state : np.ndarray of ints
            Array of shape ``spatial`` with classes.
        unary : np.ndarray
            Array of shape ``(n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        features : np.ndarray
            Array of shape ``(n_channels, *spatial)`` with with features for creating a bilateral kernel.
        spatial_spacing : None, or sequence of ints

        Returns
        -------
        energy : float
        """
        filter_size = np.broadcast_to(self.filter_size, len(state.shape))
        assert len(filter_size) <= 3
        ij = 'ijk'[:len(filter_size)]

        if spatial_spacing is None:
            spatial_spacing = len(state.shape) * [1]
        spatial_spacing = torch.tensor(spatial_spacing)[None]

        # Compute kernel
        smoothing_filter = to_np(self._smoothing_filter(filter_size, self.inverse_smoothing_bandwidth, spatial_spacing))
        kernel = to_np(self.smoothing_weight) * smoothing_filter
        if features is not None:
            adaptive_filter = to_np(self._adaptive_filter(features[None], filter_size, self.inverse_appearance_bandwidth))
            adaptive_filter = self._make_broadcastable_to_adaptive(smoothing_filter) * adaptive_filter
            kernel = self._make_broadcastable_to_adaptive(kernel) + to_np(self.appearance_weight) * adaptive_filter

        # Compute compatibilities
        compatibilities = np.nan_to_num(self._compatibility_function(state[(..., *len(filter_size) * [None])],
                                                                     unfold(state, filter_size, fill_value=np.nan)))

        pairwise_term = np.sum(np.einsum(f'...{ij}, ...{ij}', compatibilities, kernel))

        return np.sum(np.take_along_axis(unary, state[None], 0)) + pairwise_term

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
    def _smoothing_filter(filter_size, inverse_bandwidth, spatial_spacings):
        """
        Parameters
        ----------
        filter_size : sequence of ints
        inverse_bandwidth : torch.nn.parameter.Parameter containing 1 or ``len(filter_size)`` floats
        spatial_spacings : torch.tensor of shape ``(batch_size, len(filter_size))``.

        Returns
        -------
        filter_ : torch.tensor
            Gaussian filter of shape ``(batch_size, *filter_size)``
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
    def _adaptive_filter(features, filter_size, inverse_bandwidth):
        """
        Parameters
        ----------
        features : torch.tensor or np.ndarray
            Tensor of shape ``(batch_size, n_channels, *spatial)``.
        filter_size : np.ndarray with ``len(spatial)`` ints
        inverse_bandwidth : torch.nn.parameter.Parameter containing 1 or ``n_channels`` floats

        Returns
        -------
        filters : torch.tensor
            Tensor of shape ``(batch_size, *spatial, *filter_size)``. Contains trash (not Nans) in the positions where filters' values are not defined.
        """
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(inverse_bandwidth)

        diffs = features[(..., *len(filter_size) * [None])] - unfold(features, filter_size, fill_value=0)
        scaled_diffs = diffs * inverse_bandwidth.view(-1, *(len(features.shape[2:]) + len(filter_size)) * [1])
        filters = torch.exp(-torch.sum(scaled_diffs ** 2 / 2, dim=1))

        # in-place operation ``filters[(..., *filter_size // 2)] = 0`` does not allow
        # to compute gradients w.r.t. ``inverse_bandwidth``
        zero_centers = torch.ones(filters.shape).to(filters)
        zero_centers[(..., *filter_size // 2)] = 0
        filters = zero_centers * filters

        return filters

    @staticmethod
    def _make_broadcastable_to_adaptive(smoothing_filter):
        """
        Parameters
        ----------
        smoothing_filter : : torch.tensor or np.ndarray
            Tensor of shape ``(batch_size, *filter_size)``

        Returns
        -------
        smoothing_filter : torch.tensor or np.ndarray
            Tensor of shape ``(batch_size, *spatial, *filter_size)``.
        """
        n_spatial = len(smoothing_filter.shape) - 1  # len(filter_size) == len(spatial)
        return smoothing_filter[(slice(None), *n_spatial * [None])]

    @staticmethod
    def _convolve_classwisely(x, filter_):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        filter_ : torch.tensor
            Tensor of shape ``(batch_size, *filter_size)``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        conv = [F.conv1d, F.conv2d, F.conv3d][len(x.shape) - 3]
        padding = [fs // 2 for fs in filter_.shape[1:]]

        return torch.stack([conv(x_[0, :, None], f[None], padding=padding)
                            for x_, f in zip(torch.split(x, 1, 0), torch.split(filter_, 1, 0))]).squeeze(2)

    @staticmethod
    def _pass_message(x, adaptive_filter, n_classes_to_vectorize=None):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        adaptive_filter : torch.tensor
            Tensor of shape ``(batch_size, *spatial, *filter_size)``.
        n_classes_to_vectorize : int or None
            The number of classes which are processed in parallel. Default is None, which means all classes. Large ``n_classes_to_vectorize`` leads to faster but more storage-consuming computations.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        filter_size = np.array(adaptive_filter.shape[len(x.shape) - 1:])
        assert len(filter_size) <= 3
        ij = 'ijk'[:len(filter_size)]

        if n_classes_to_vectorize is None:
            return torch.einsum(f'...{ij}, ...{ij}', *torch.broadcast_tensors(
                unfold(x, filter_size, fill_value=0), adaptive_filter.unsqueeze(1)
            ))
        else:
            return torch.cat([
                torch.einsum(f'...{ij}, ...{ij}', *torch.broadcast_tensors(
                    unfold(x_, filter_size, fill_value=0), adaptive_filter.unsqueeze(1)
                )) for x_ in torch.split(x, n_classes_to_vectorize, 1)
            ], dim=1)

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.

        Returns
        -------
        output : torch.tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        labels = torch.arange(x.shape[1])
        compatibility_matrix = self._compatibility_function(labels, labels.unsqueeze(1)).to(x)
        return torch.einsum('ij..., jk -> ik...', x, compatibility_matrix)

    @staticmethod
    def _compatibility_function(label1, label2):
        """
        Inputs must be broadcastable.

        Parameters
        ----------
        label1 : torch.tensor or np.ndarray
        label2 : torch.tensor or np.ndarray

        Returns
        -------
        compatibility : float
        """
        if isinstance(label1, torch.Tensor) and isinstance(label2, torch.Tensor):
            return -(label1 == label2).float()
        elif isinstance(label1, np.ndarray) and isinstance(label2, np.ndarray):
            return -(label1 == label2).astype(float)
        else:
            raise TypeError('``label1`` and ``label2`` must both be torch tensors or both be numpy arrays.')


class RegressionCRF(MeanFieldCRF):
    """
    Subclass for real-valued labels prediction.

    Parameters
    ----------
    compatibility_bandwidth : float
        Initial bandwidth for the gaussian compatibility function.
    """
    def __init__(self, compatibility_bandwidth=1, **kwargs):
        super().__init__(**kwargs)
        self._set_param('inverse_compatibility_bandwidth', 1 / np.asarray(compatibility_bandwidth))

    def _compatibility_function(self, label1, label2):
        if isinstance(label1, torch.Tensor) and isinstance(label2, torch.Tensor):
            return -torch.exp(-(label1 - label2) ** 2 * self.inverse_compatibility_bandwidth ** 2 / 2)
        elif isinstance(label1, np.ndarray) and isinstance(label2, np.ndarray):
            return -np.exp(-(label1 - label2) ** 2 * self.inverse_compatibility_bandwidth.item() ** 2 / 2)
        else:
            raise TypeError('``label1`` and ``label2`` must both be torch tensors or both be numpy arrays.')


class CircularCRF(RegressionCRF):
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




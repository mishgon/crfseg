import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRF(nn.Module):
    """
    Class for learning and inference in conditional random field model using mean field approximation
    and convolutional approximation in pairwise potentials term.

    Parameters
    ----------
    filter_size : int or sequence of ints
        Size of the gaussian filters in message passing.
        If it is a sequence its length must be equal to the number of spatial dimensions of input tensors.
    n_iter : int
        Number of iterations in mean field approximation.
    n_classes_to_vectorize : int or None
        The number of classes which are processed in vectorized manner. Default is None, which means all classes.
        Large ``n_classes_to_vectorize`` leads to faster but more storage-consuming computations.
    requires_grad : bool
        Whether or not to train CRF's parameters.
    return_log_proba : bool
        Whether to return log-probabilities (which is more computationally stable if then the nn.NLLLoss is used),
        or probabilities.
    smoothness_weight : float
        Initial weight of smoothness kernel.
    smoothness_theta : float or sequence of floats
        Initial bandwidths for each spatial feature in the gaussian smoothness kernel.
        If it is a sequence its length must be equal to the number of spatial dimensions of input tensors.
    """

    def __init__(self, n_spatial_dims, filter_size=11, n_iter=5, n_classes_to_vectorize=None, requires_grad=True,
                 return_log_proba=True, smoothness_weight=1, smoothness_theta=1):
        super().__init__()
        self.n_spatial_dims = n_spatial_dims
        self.n_iter = n_iter
        self.filter_size = np.broadcast_to(filter_size, n_spatial_dims)
        self.n_classes_to_vectorize = n_classes_to_vectorize
        self.return_log_proba = return_log_proba
        self.requires_grad = requires_grad

        self._set_param('smoothness_weight', smoothness_weight)
        self._set_param('inv_smoothness_theta', 1 / np.broadcast_to(smoothness_theta, n_spatial_dims))

    def _set_param(self, name, init_value):
        setattr(self, name, nn.Parameter(torch.tensor(init_value, dtype=torch.float, requires_grad=self.requires_grad)))

    def forward(self, x, features=None, spatial_spacings=None):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        features : torch.tensor
            Tensor of shape ``(batch_size, n_channels, *spatial)`` with features for creating a bilateral kernel.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.
            None is equivalent to all ones. Used to adapt spatial gaussian filters to different inputs' resolutions.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with (log-)probabilities of assignment to each class.
        """
        batch_size, n_classes, *spatial = x.shape
        assert len(spatial) == self.n_spatial_dims
        if spatial_spacings is None:
            spatial_spacings = np.ones((batch_size, self.n_spatial_dims))

        negative_unary = x.clone()

        for i in range(self.n_iter):
            # normalizing
            x = F.softmax(x, dim=1)

            # message passing
            x = self.smoothness_weight * self._smoothing_filter(x, spatial_spacings)

            # compatibility transform
            x = self._compatibility_transform(x)

            # adding unary potentials
            x = negative_unary - x

        return F.log_softmax(x, dim=1) if self.return_log_proba else F.softmax(x, dim=1)

    def _smoothing_filter(self, x, spatial_spacings):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)`` with negative unary potentials, e.g. logits.
        spatial_spacings : torch.tensor or None
            Tensor of shape ``(batch_size, len(spatial))`` with spatial spacings of tensors in batch ``x``.

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(batch_size, n_classes, *spatial)``.
        """
        return torch.stack([self._gaussian_filter(x[i], self.inv_smoothness_theta, spatial_spacings[i], self.filter_size)
                            for i in range(x.shape[0])])

    @staticmethod
    def _gaussian_filter(x, inv_theta, spatial_spacing, filter_size):
        """
        Parameters
        ----------
        x : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        inv_theta : torch.tensor
            Tensor of shape ``(len(spatial),)``
        spatial_spacing : sequence of len(spatial) floats
        filter_size : sequence of len(spatial) ints

        Returns
        -------
        output : torch.tensor
            Tensor of shape ``(n, *spatial)``.
        """
        for i, dim in enumerate(range(1, x.ndim)):
            # reshape to (-1, 1, x.shape[dim])
            x = x.transpose(dim, -1)
            shape_before_flatten = x.shape[:-1]
            x = x.flatten(0, -2).unsqueeze(1)

            # 1d gaussian filter
            kernel = CRF.create_gaussian_kernel1d(inv_theta[i], spatial_spacing[i], filter_size[i]).view(1, 1, -1).to(x)
            x = F.conv1d(x, kernel, padding=(filter_size[i] // 2,))

            # reshape back to (n, *spatial)
            x = x.squeeze(1).view(*shape_before_flatten, x.shape[-1]).transpose(-1, dim)

        return x

    @staticmethod
    def create_gaussian_kernel1d(inverse_theta, spacing, filter_size):
        """
        Parameters
        ----------
        inverse_theta : torch.tensor
            Tensor of shape ``(,)``
        spacing : float
        filter_size : int

        Returns
        -------
        kernel : torch.tensor
            Tensor of shape ``(filter_size,)``.
        """
        distances = spacing * torch.arange(-(filter_size // 2), filter_size // 2 + 1).to(inverse_theta)
        kernel = torch.exp(-(distances * inverse_theta) ** 2 / 2)
        zero_center = torch.ones(filter_size).to(kernel)
        zero_center[filter_size // 2] = 0
        return kernel * zero_center

    def _compatibility_transform(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape ``(batch_size, n_classes, *spatial)``.

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
        Input tensors must be broadcastable.

        Parameters
        ----------
        label1 : torch.Tensor
        label2 : torch.Tensor

        Returns
        -------
        compatibility : torch.Tensor
        """
        return -(label1 == label2).float()

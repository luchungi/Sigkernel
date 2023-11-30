from __future__ import annotations

from abc import ABCMeta, abstractmethod

import warnings

import numpy as np
import torch

from . import utils
from . import static
from .projections import RandomProjection
from .static.kernels import Kernel
from .static.features import LowRankFeatures
from .algorithms import signature_kern, signature_kern_low_rank

from typing import Optional

from sklearn.base import clone

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureBase(Kernel, metaclass=ABCMeta):
    """Base class for signature kernels.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    @abstractmethod
    def __init__(self, n_levels : int = 5, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : int = 0,
                 n_features : Optional[int] = None) -> None:
        self.n_levels = utils.check_positive_value(n_levels, 'n_levels')
        self.order = self.n_levels if order <= 0 or order >= self.n_levels else order
        self.sigma = utils.check_positive_value(sigma, 'sigma')
        self.normalization = normalization
        self.difference = difference
        self.n_features = utils.check_positive_value(n_features, 'n_features') if n_features is not None else None

    def _validate_data(self, X : torch.Tensor, reset : Optional[bool] = False) -> torch.Tensor:

        n_features = self.n_features_ if hasattr(self, 'n_features_') and self.n_features_ is not None else self.n_features # default
        if X.ndim == 2:
            if n_features is None or reset:
                warnings.warn('The input array has ndim==2. Assuming inputs are univariate time series.',
                              'It is recommended to pass an n_features parameter during init when using flattened arrays of ndim==2.')
                n_features = 1
        elif X.ndim == 3:
            if n_features is None or reset:
                n_features = X.shape[-1]
            elif X.shape[-1] != n_features:
                raise ValueError('The last dimension of the 3-dim input array does not match the saved n_features parameter ',
                                 '(either during init or during the last time fit was called).')
        else:
            raise ValueError('Only input sequence arrays with ndim==2 or ndim==3 are supported.')
        # reshape data to ndim==3
        X = X.reshape([X.shape[0], -1, n_features])
        if reset:
            self.n_features_ = n_features

        return X

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class SignatureKernel(SignatureBase):
    """Class for full-rank signature kernel."""

    def __init__(self, n_levels : int = 4, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : int = 0,
                 n_features : Optional[int] = None, static_kernel : Optional[Kernel] = None, device_ids : Optional[list[int]] = None) -> None:
        '''
        Parameters
        ----------
        n_levels : int, default=4
            The number of levels of the signature kernel.
        order : int, default=1
            The order of the signature kernel. If n_levels > order, then the signature kernel is only accurate up to order.
        sigma : float, default=1.0
            The square root of the scalar factor multiplied to the kernel computation.
        difference : bool, default=True
            Whether to compute the difference kernel which is required.
            Refer to "Kernels for Sequentially Ordered Data" to see how to calculate signature in kernel feature space
        normalization : int, default=0
            The normalization scheme to use. Possible values are 0, 1, 2, 3.
            0 : No normalization.
            1 : Normalize each level of the signature kernel by its Frobenius norm.
            2 : Normalize the signature kernel by the Frobenius norm of the full signature kernel.
            3 : Normalize each level of the signature kernel the robust normalization described in "Signature Moments to Characterize Laws of Stochastic Processes".
        n_features : int, default=None
            The number of features/dimensions of the sequence data. Will be inferred from the data if not provided.
        static_kernel : Kernel, default=None
            The kernel to use for the static kernel. If None, the linear kernel is used.
            The signature is calculated in the feature space of the static kernel as described in "Kernels for Sequentially Ordered Data".
        '''

        super().__init__(n_levels=n_levels, order=order, sigma=sigma, difference=difference, normalization=normalization, n_features=n_features)
        self.static_kernel = static_kernel or static.kernels.LinearKernel()
        self.device_ids = device_ids

    def _compute_kernel(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None, diag : bool = False) -> torch.Tensor:

        # check compatible inputs
        X = self._validate_data(X)
        if Y is not None:
            Y = self._validate_data(Y)
            if X.shape[-1] != Y.shape[-1]:
                raise ValueError('The input arrays X and Y have different dimensionality along the last axis.')

        if diag:
            if Y is not None:
                raise ValueError('Diagonal mode does not support a 2nd input array.')
            M = self.static_kernel(X)
        else:
            M = self.static_kernel(X.reshape((-1, X.shape[-1]))).reshape((X.shape[0], X.shape[1], X.shape[0], X.shape[1])) if Y is None \
                else self.static_kernel(X.reshape((-1, X.shape[-1])), Y.reshape((-1, Y.shape[-1]))).reshape((X.shape[0], X.shape[1], Y.shape[0], Y.shape[1]))

        return_levels = True if self.normalization==1 or self.normalization==3 else False
        K = signature_kern(M, self.n_levels, order=self.order, difference=self.difference, return_levels=return_levels, device_ids=self.device_ids)

        if self.normalization == 1:
            if Y is None:
                K_X_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= K_X_sqrt[..., :, None] * K_X_sqrt[..., None, :]
            else:
                K_X_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K_Y_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference, return_levels=True))
                K /= K_X_sqrt[..., :, None] * K_Y_sqrt[..., None, :]
            K = torch.mean(K, dim=0)
        elif self.normalization == 2:
            if Y is None:
                K_X_sqrt = utils.robust_sqrt(utils.matrix_diag(K))
                K /= K_X_sqrt[:, None] * K_X_sqrt[None, :]
            else:
                K_X_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(X), self.n_levels, order=self.order, difference=self.difference))
                K_Y_sqrt = utils.robust_sqrt(signature_kern(self.static_kernel(Y), self.n_levels, order=self.order, difference=self.difference))
                K /= K_X_sqrt[:, None] * K_Y_sqrt[None, :]

        return self.sigma**2 * K

    def _Kdiag(self, X : torch.Tensor) -> torch.Tensor:
        if self.normalization != 0:
            return torch.full((X.shape[0],), self.sidim*2)
        else:
            return self._compute_kernel(X, diag=True)

    def _K(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
        return self._compute_kernel(X, Y)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

class LowRankSignatureKernel(SignatureBase, LowRankFeatures):
    """Class for low-rank signature kernel."""
    def __init__(self, n_levels : int = 4, order : int = 1, sigma : float = 1.0, difference : bool = True, normalization : bool = False,
                 n_features : Optional[int] = None, static_features : Optional[LowRankFeatures] = None, projection : Optional[RandomProjection] = None) -> None:

        super().__init__(n_levels=n_levels, order=order, sigma=sigma, difference=difference, normalization=normalization, n_features=n_features)
        self.static_features = static_features
        self.projection = projection

    def _make_feature_components(self, X : torch.Tensor) -> None:
        if self.static_features is not None:
            self.static_features_ = self.static_features.fit(X)
            U = self.static_features_.transform(X)
        else:
            U = X
        if self.projection is not None:
            self.projections_ = [clone(self.projection).fit(U)]
            V = self.projections_[0](U)
            self.projections_ += [clone(self.projection).fit(V, Z=U) for i in range(1, self.n_levels)]
        else:
            self.projections_ = None

    def _compute_features(self, X : torch.Tensor) -> torch.Tensor:
        U = self.static_features_.transform(X, return_on_gpu=True) if self.static_features is not None else X
        P = signature_kern_low_rank(U, self.n_levels, order=self.order, difference=self.difference, return_levels=self.normalization==1, projections=self.projections_)
        if self.normalization == 1:
            P_norms = [utils.robust_sqrt(utils.squared_norm(p, axis=-1)) for p in P]
            P = torch.cat([p / P_norms[i][..., None] for i, p in enumerate(P)], dim=-1) / torch.sqrt(self.n_levels+1)
        elif self.normalization == 2:
            P_norms = utils.robust_sqrt(utils.squared_norm(p, axis=-1))
            P /= P_norms

        return self.sigma * P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import torch
from typing import Optional, Union, List

from .dgpu import SplitTensor, split_multiply

from .projections import RandomProjection, TensorizedRandomProjection

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def multi_cumsum(M : torch.Tensor, exclusive : bool = False, axis : int = -1) -> torch.Tensor:
    """Computes the exclusive cumulative sum along a given set of axes.

    Args:
        K (torch.Tensor): A matrix over which to compute the cumulative sum
        axis (int or iterable, optional): An axis or a collection of them. Defaults to -1 (the last axis).
    """

    ndim = M.ndim
    axis = [axis] if np.isscalar(axis) else axis
    axis = [ndim+ax if ax < 0 else ax for ax in axis]

    # create slice for exclusive cumsum (slice off last element along given axis then pre-pad with zeros)
    if exclusive:
        slices = tuple(slice(-1) if ax in axis else slice(None) for ax in range(ndim))
        M = M[slices]

    # compute actual cumsums
    for ax in axis:
        M = torch.cumsum(M, dim=ax)

    # pre-pad with zeros along the given axis if exclusive cumsum
    if exclusive:
        pads = tuple(x for ax in reversed(range(ndim)) for x in ((1, 0) if ax in axis else (0, 0)))
        M = torch.nn.functional.pad(M, pads)

    return M

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Signature algs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern(M : torch.Tensor, n_levels : int, order : int = -1, difference : bool = True,
                   return_levels : bool = False, device_ids : Optional[list[int]] = None) -> torch.Tensor:
    """Wrapper for signature kernel algorithms. If order==1 then it uses a simplified, more efficient implementation."""
    order = n_levels if order <= 0 or order >= n_levels else order
    if order==1:
        return signature_kern_first_order(M, n_levels, difference=difference, return_levels=return_levels)
    else:
        return signature_kern_higher_order(M, n_levels, order=order, difference=difference, return_levels=return_levels, device_ids=device_ids)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_first_order(M : torch.Tensor, n_levels : int, difference : bool = True, return_levels : bool = False) -> torch.Tensor:
    """
    Computes the signature kernel matrix with first-order embedding into the tensor algebra.
    """

    if difference:
        M = torch.diff(torch.diff(M, dim=1), dim=-1) # computes d(i,j)(k(x,y)) = k(x(i+1),y(j+1)) + k(x(i),y(j)) - k(x(i+1),y(j)) - k(x(i),y(j+1))

    if M.ndim == 4:
        n_X, n_Y = M.shape[0], M.shape[2]
        K = torch.ones((n_X, n_Y), dtype=M.dtype)
    else:
        n_X = M.shape[0]
        K = torch.ones((n_X,), dtype=M.dtype)

    if return_levels:
        K = [K, torch.sum(M, dim=(1, -1))]
    else:
        K += torch.sum(M, dim=(1, -1))

    R = torch.clone(M)
    for i in range(1, n_levels):
        R = M * multi_cumsum(R, exclusive=True, axis=(1, -1))
        if return_levels:
            K.append(torch.sum(R, dim=(1, -1)))
        else:
            K += torch.sum(R, dim=(1, -1))

    return torch.stack(K, dim=0) if return_levels else K

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_higher_order(M : torch.Tensor, n_levels : int, order : int, difference : bool = True,
                                return_levels : bool = False, device_ids : Optional[list[int]] = None) -> torch.Tensor:
    '''
    Computes the higher-order full rank signature kernel using a kernel trick.
    Args:
        M: Kernel evaluations of shape `[n_X, n_Y, l_X, l_Y]` or `[n, l_X, l_Y]`.
        n_levels: Number of signature levels.
        order: Signature embedding order.
        difference: Whether to take increments of lifted sequences in the RKHS.
        return_levels: Whether to return the kernel for each level separately.
    Returns:
        The signature kernel matrix of shape `[..., n_X, n_Y]` or `[..., n]`,
        depending on `M` above, and `...` is `n_levels` when `return_levels`.
    '''

    device = M.device
    if device_ids:
        if difference:
            M = torch.diff(torch.diff(M, dim=1), dim=-1) # computes d(i,j)(k(x,y)) = k(x(i+1),y(j+1)) + k(x(i),y(j)) - k(x(i+1),y(j)) - k(x(i),y(j+1))
        else:
            M = M
        ndim = M.ndim
        M_shape = M.shape
        M_dtype = M.dtype

        if ndim == 4:
            n_X, n_Y = M_shape[0], M_shape[2]
            K = torch.ones((n_X, n_Y), dtype=M_dtype, device=device)
        else:
            n_X = M_shape[0]
            K = torch.ones((n_X,), dtype=M_dtype, device=device)

        if return_levels:
            K = [K, torch.sum(M, dim=(1, -1))]
        else:
            K += torch.sum(M, dim=(1, -1))

        R = SplitTensor(torch.clone(M).unsqueeze(0).unsqueeze(0), device_ids=device_ids, dim=2)
        M = SplitTensor(M, device_ids=device_ids, dim=0)
        for i in range(1, n_levels):
            d = min(i+1, order)
            R_next = torch.empty((d, d) + M_shape, dtype=M_dtype)
            R_next = SplitTensor(R_next, device_ids=device_ids, dim=2)
            # R_next[0, 0] = M * multi_cumsum(torch.sum(R, dim=(0, 1)), exclusive=True, axis=(1, -1))
            R_next.assign([0,0] + [slice(None)]*(ndim-2), split_multiply(M, R.sum(dim=(0,1)).multi_cumsum(exclusive=True, dim=(1, -1))))
            for r in range(1, d):
                # R_next[0, r] = 1./(r+1) * M * multi_cumsum(torch.sum(R[:, r-1], dim=0), exclusive=True, axis=1)
                idx = [slice(None), r-1] + [slice(None)]*(ndim-2)
                value = split_multiply(M, R.slice(idx).sum(dim=0).multi_cumsum(exclusive=True, dim=1))
                value.div(r+1)
                R_next.assign([0,r] + [slice(None)]*(ndim-2), value)
                # R_next[r, 0] = 1./(r+1) * M * multi_cumsum(torch.sum(R[r-1, :], dim=0), exclusive=True, axis=-1)
                idx = [r-1, slice(None)] + [slice(None)]*(ndim-2)
                value = split_multiply(M, R.slice(idx).sum(dim=0).multi_cumsum(exclusive=True, dim=-1))
                value.div(r+1)
                R_next.assign([r,0] + [slice(None)]*(ndim-2), value)
                for s in range(1, d):
                    # R_next[r, s] = 1./((r+1)*(s+1)) * M * R[r-1, s-1]
                    idx = [r-1, s-1] + [slice(None)]*(ndim-2)
                    value = split_multiply(M, R.slice(idx))
                    value.div((r+1)*(s+1))
                    R_next.assign([r,s], value)
            R = R_next
            if return_levels:
                # K.append(torch.sum(R, dim=(0, 1, 3, -1)))
                K.append(R.sum(dim=(0, 1, 3, -1)).recombine(out_device=device))
            else:
                # K += torch.sum(R, dim=(0, 1, 3, -1))
                K += R.sum(dim=(0, 1, 3, -1)).recombine(out_device=device)
    else:
        if difference:
            M = torch.diff(torch.diff(M, dim=1), dim=-1) # computes d(i,j)(k(x,y)) = k(x(i+1),y(j+1)) + k(x(i),y(j)) - k(x(i+1),y(j)) - k(x(i),y(j+1))

        if M.ndim == 4:
            n_X, n_Y = M.shape[0], M.shape[2]
            K = torch.ones((n_X, n_Y), dtype=M.dtype, device=device)
        else:
            n_X = M.shape[0]
            K = torch.ones((n_X,), dtype=M.dtype, device=device)

        if return_levels:
            K = [K, torch.sum(M, dim=(1, -1))]
        else:
            K += torch.sum(M, dim=(1, -1))

        R = torch.clone(M).unsqueeze(0).unsqueeze(0)
        for i in range(1, n_levels):
            d = min(i+1, order)
            R_next = torch.empty((d, d) + M.shape, dtype=M.dtype, device=device)
            R_next[0, 0] = M * multi_cumsum(torch.sum(R, dim=(0, 1)), exclusive=True, axis=(1, -1))
            for r in range(1, d):
                R_next[0, r] = 1./(r+1) * M * multi_cumsum(torch.sum(R[:, r-1], dim=0), exclusive=True, axis=1)
                R_next[r, 0] = 1./(r+1) * M * multi_cumsum(torch.sum(R[r-1, :], dim=0), exclusive=True, axis=-1)
                for s in range(1, d):
                    R_next[r, s] = 1./((r+1)*(s+1)) * M * R[r-1, s-1]
            R = R_next
            if return_levels:
                K.append(torch.sum(R, dim=(0, 1, 3, -1)))
            else:
                K += torch.sum(R, dim=(0, 1, 3, -1))

    return torch.stack(K, dim=0) if return_levels else K

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Low-Rank algs
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_low_rank(U : torch.Tensor, n_levels : int, order : int = -1, difference : bool = True, return_levels : bool = False,
                            projections : Optional[RandomProjection] = None) -> Union[List[torch.Tensor], torch.Tensor]:
    """Wrapper for low-rank signature kernel algs. If order==1 then it uses a simplified, more efficient implementation."""
    order = n_levels if order <= 0 or order >= n_levels else order
    if order==1:
        return signature_kern_first_order_low_rank(U, n_levels, difference=difference, return_levels=return_levels, projections=projections)
    else:
        return signature_kern_higher_order_low_rank(U, n_levels, order=order, difference=difference, return_levels=return_levels, projections=projections)

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_first_order_low_rank(U : torch.Tensor, n_levels : int, difference : bool = True, return_levels : bool = False,
                                        projections : Optional[RandomProjection] = None) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Computes a low-rank feature approximation corresponding to the signature kernel with first-order embedding into the tensor algebra.
    """

    if difference:
        U = torch.diff(U, dim=1)

    n_X, l_X, n_d = U.shape
    P = torch.ones((n_X, 1), dtype=U.dtype)

    R = projections[0](U, return_on_gpu=True) if projections is not None else U.clone()
    R_sum = torch.sum(R.reshape([n_X, l_X, projections[0].n_components, projections[0].rank]), dim=(1, -1)) if projections is not None \
            and isinstance(projections[0], TensorizedRandomProjection) else torch.sum(R, dim=1)
    if return_levels:
        P = [P, R_sum]
    else:
        P = torch.cat((P, R_sum), dim=-1)

    for i in range(1, n_levels):
        R = multi_cumsum(R, axis=1, exclusive=True)
        if projections is None:
            R = torch.reshape(R[..., :, None] * U[..., None, :], (n_X, l_X, -1))
        else:
            R = projections[i](R, U, return_on_gpu=True)
        R_sum = torch.sum(R.reshape([n_X, l_X, projections[i].n_components, projections[i].rank]), dim=(1, -1)) if projections is not None \
                and isinstance(projections[i], TensorizedRandomProjection) else torch.sum(R, dim=1)
        if return_levels:
            P.append(R_sum)
        else:
            P = torch.cat((P, torch.sum(R, dim=1)), dim=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def signature_kern_higher_order_low_rank(U : torch.Tensor, n_levels : int, order : int = -1, difference : bool = True, return_levels : bool = False,
                                         projections : Optional[RandomProjection] = None) -> Union[List[torch.Tensor], torch.Tensor]:
    """
    Computes a low-rank feature approximation corresponding to the signature kernel with higher-order embedding into the tensor algebra.
    """

    if difference:
        U = torch.diff(U, dim=1)

    n_X, l_X, n_d = U.shape
    P = torch.ones((n_X, 1), dtype=U.dtype)

    R = projections[0](U, return_on_gpu=True) if projections is not None else U.clone()
    R_sum = torch.sum(R.reshape([n_X, l_X, projections[0].n_components, projections[0].rank]), dim=(1, -1)) if projections is not None \
            and isinstance(projections[0], TensorizedRandomProjection) else torch.sum(R, dim=1)
    if return_levels:
        P = [P, R_sum]
    else:
        P = torch.cat((P, R_sum), dim=-1)

    R = R[None]
    for i in range(1, n_levels):
        d = min(i+1, order)
        n_components = projections[i].n_components_ if projections is not None else n_d**(i+1)
        R_next = torch.empty((d, n_X, l_X, n_components))

        Q = multi_cumsum(torch.sum(R, dim=0), axis=1, exclusive=True)
        if projections is None:
            R_next[0] = torch.reshape(Q[..., :, None] * U[..., None, :], (n_X, l_X, -1))
        else:
            R_next[0] = projections[i](Q, U, return_on_gpu=True)
        for r in range(1, d):
            if projections is None:
                R_next[r] = 1. / (r+1) * torch.reshape(R[r-1, ..., :, None] * U[..., None, :], (n_X, l_X, n_components))
            else:
                R_next[r] = 1. / (r+1) * projections[i](R[r-1], U, return_on_gpu=True)
        R = R_next
        R_sum = torch.sum(R.reshape([d, n_X, l_X, projections[i].n_components, projections[i].rank]), dim=(0, 2, -1)) if projections is not None \
                and isinstance(projections[i], TensorizedRandomProjection) else torch.sum(R, dim=(0, 2))
        if return_levels:
            P.append(R_sum)
        else:
            P = torch.cat((P, R_sum), dim=-1)
    return P

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
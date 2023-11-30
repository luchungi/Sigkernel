import numpy as np
import torch

from numbers import Number, Integral

from typing import Optional, Union, List, Tuple

RandomStateOrSeed = Union[Integral, torch.Generator]

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (Type) checkers
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def check_positive_value(scalar : Number, name : str) -> Number:
    if scalar <= 0:
        raise ValueError(f'The parameter \'{name}\' should have a positive value.')
    return scalar

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# (Batched) computations
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def matrix_diag(A : torch.Tensor) -> torch.Tensor:
    """Takes as input an array of shape (..., d, d) and returns the diagonals along the last two axes with output shape (..., d)."""
    return torch.einsum('...ii->...i', A)

def matrix_mult(X : torch.Tensor, Y : Optional[torch.Tensor] = None, transpose_X : bool = False, transpose_Y : bool = False) -> torch.Tensor:
    subscript_X = '...ji' if transpose_X else '...ij'
    subscript_Y = '...kj' if transpose_Y else '...jk'
    return torch.einsum(f'{subscript_X},{subscript_Y}->...ik', X, Y if Y is not None else X)

def squared_norm(X : torch.Tensor, dim : int = -1) -> torch.Tensor:
    return torch.sum(torch.square(X), dim=dim)

def squared_euclid_dist(X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
    X_n2 = squared_norm(X)
    if Y is None:
        D2 = (X_n2[..., :, None] + X_n2[..., None, :]) - 2 * matrix_mult(X, X, transpose_Y=True)
    else:
        Y_n2 = squared_norm(Y, dim=-1)
        D2 = (X_n2[..., :, None] + Y_n2[..., None, :]) - 2 * matrix_mult(X, Y, transpose_Y=True)
    return D2

def outer_prod(X : torch.Tensor, Y : torch.Tensor) -> torch.Tensor:
    return torch.reshape(X[..., :, None] * Y[..., None, :], X.shape[:-1] + (-1,))

def robust_sqrt(X : torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.maximum(X, 1e-20))

def euclid_dist(self, X : torch.Tensor, Y : Optional[torch.Tensor] = None) -> torch.Tensor:
    return robust_sqrt(squared_euclid_dist(X, Y))

def robust_nonzero(X : torch.Tensor) -> torch.Tensor:
    return torch.abs(X) > 1e-10


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Probability stuff
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def check_random_state(seed : Optional[RandomStateOrSeed] = None) -> RandomStateOrSeed:
    if seed is None:
        return torch.Generator()
    elif isinstance(seed, Integral):
        gen = torch.Generator()
        gen.manual_seed(seed)
        return gen
    elif isinstance(seed, torch.Generator):
        return torch.Generator()
    raise ValueError(f'{seed} cannot be used to seed a cupy.random.RandomState instance')

def draw_rademacher_matrix(shape : Union[List[int], Tuple[int]], random_state : Optional[RandomStateOrSeed] = None) -> torch.Tensor:
    random_state = check_random_state(random_state)
    return torch.where(random_state.uniform(size=shape) < 0.5, torch.ones(shape), -torch.ones(shape))

def draw_bernoulli_matrix(shape : Union[List[int], Tuple[int]], prob : float, random_state : Optional[RandomStateOrSeed] = None) -> torch.Tensor:
    random_state = check_random_state(random_state)
    return torch.where(random_state.uniform(size=shape) < prob, torch.ones(shape), torch.zeros(shape))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Projection stuff
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

def subsample_outer_prod_comps(X : torch.Tensor, Z : torch.Tensor, sampled_idx : Union[torch.Tensor, List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    idx_X = torch.arange(X.shape[-1]).reshape([-1, 1, 1])
    idx_Z = torch.arange(Z.shape[-1]).reshape([1, -1, 1])
    idx_pairs = torch.reshape(torch.concatenate((idx_X + torch.zeros_like(idx_Z), idx_Z + torch.zeros_like(idx_X)), dim=-1), (-1, 2))
    sampled_idx_pairs = torch.squeeze(torch.take(idx_pairs, sampled_idx, dim=0))
    X_proj = torch.take(X, sampled_idx_pairs[:, 0], dim=-1)
    Z_proj = torch.take(Z, sampled_idx_pairs[:, 1], dim=-1)
    return X_proj, Z_proj

def compute_count_sketch(X : torch.Tensor, hash_index : torch.Tensor, hash_bit : torch.Tensor, n_components : Optional[int] = None) -> torch.Tensor:
    # if n_components is None, get it from the hash_index array
    n_components = n_components if n_components is not None else torch.max(hash_index)
    hash_mask = torch.asarray(hash_index[:, None] == torch.arange(n_components)[None, :], dtype=X.dtype)
    X_count_sketch = torch.einsum('...i,ij,i->...j', X, hash_mask, hash_bit)
    return X_count_sketch

def convolve_count_sketches(X_count_sketch : torch.Tensor, Z_count_sketch : torch.Tensor) -> torch.Tensor:
    X_count_sketch_fft = torch.fft.fft(X_count_sketch, dim=-1)
    Z_count_sketch_fft = torch.fft.fft(Z_count_sketch, dim=-1)
    XZ_count_sketch = torch.real(torch.fft.ifft(X_count_sketch_fft * Z_count_sketch_fft, dim=-1))
    return XZ_count_sketch

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
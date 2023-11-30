import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import QuantLib as ql

from .kernels import SignatureKernel
from .utils import matrix_diag

def psi(x, a=1, C=4):
    # helper function to calculate the normalisation constant for the characteristic signature kernel
    x = C+C**(1+a)*(C**-a - x**-a)/a if x>4 else x
    return x

def norm_func(λ: float, norms: np.ndarray, a: float, c: float):
    '''
    Function to solve for root which are the normalisation constants for the characteristic signature kernel
    λ: float, normalisation constant
    norms: torch.tensor of shape (n_levels,) where n_levels is the number of signature levels
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    norm_sum = norms.sum()
    m = len(norms)
    λ = np.ones(m) * λ
    powers = np.arange(m) * 2
    return np.sum(norms * np.power(λ, powers)) - psi(norm_sum, a, c)

def np_matrix_diag(A : torch.Tensor) -> torch.Tensor:
    """Takes as input an array of shape (..., d, d) and returns the diagonals along the last two axes with output shape (..., d)."""
    return np.einsum('...ii->...i', A)

def get_normalisation_constants(gram_matrix: torch.tensor, a: float, C: float) -> np.ndarray:
    '''
    Calculate normalisation constants for each path
    normsq_levels: torch.tensor of shape (n_samples, n_levels) where n_levels is the number of signature levels
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    gram_matrix = gram_matrix.clone().detach().cpu().numpy()
    normsq_levels = np_matrix_diag(gram_matrix).T # shape (n_samples, n_levels) each row is the norm squared of the signature at each level for a sample
    n_samples = normsq_levels.shape[0]
    normsq = np.sum(normsq_levels, axis=1) # shape (n_samples,) each entry is the norm squared of the signature for a sample
    norm_condition = normsq > C # check which samples need normalisation
    λ = np.ones(n_samples)
    for i in range(n_samples):
        if norm_condition[i]:
            λ[i] = brentq(norm_func, 0, 1, args=(normsq_levels[i], a, C)) # find normalisation constant for each sample
    return λ

def sum_normalise_gram_matrix(K: torch.tensor, λ_X: np.ndarray, λ_Y: np.ndarray, n_levels: float) -> torch.tensor:
    '''
    Normalise the gram matrix of a signature kernel at each signature level then sum
    K: Gram matrix of the signature kernel torch.tensor of shape (n_samples X, n_samples Y)
    λ_X: normalisation constants for X torch.tensor of shape (n_samples X,)
    λ_Y: normalisation constants for Y torch.tensor of shape (n_samples Y,)
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    '''
    m_λ = torch.tensor((λ_X[:,np.newaxis] @ λ_Y[np.newaxis,:]) ** np.arange(n_levels+1)[:, np.newaxis, np.newaxis]).to(K.device)
    K = torch.sum(m_λ * K, dim=0)
    return K

def get_robust_gram_matrices(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel, a:float, C:float) -> (torch.tensor, torch.tensor, torch.tensor):
    '''
    Calculate the normalised gram matrices for X, Y and X,Y
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    kernel: the signature kernel to be used
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    n_levels = kernel.n_levels
    K_XX = kernel(X,X)
    λ_X = get_normalisation_constants(K_XX, a, C)
    K_XX = sum_normalise_gram_matrix(K_XX, λ_X, λ_X, n_levels)

    K_YY = kernel(Y,Y)
    λ_Y = get_normalisation_constants(K_YY, a, C)
    K_YY = sum_normalise_gram_matrix(K_YY, λ_Y, λ_Y, n_levels)

    K_XY = kernel(X,Y)
    K_XY = sum_normalise_gram_matrix(K_XY, λ_X, λ_Y, n_levels)

    # zero the diagonals as the sums will include the diagonals and we need kernel(x,x) excluded i.e. kernel of the same sample
    K_XX.fill_diagonal_(0)
    K_YY.fill_diagonal_(0)

    return K_XX, K_YY, K_XY

def get_permutation_indices(n: int, m: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    '''
    get indices for permutation test
    n: int, number of samples in first sample (X)
    m: int, number of samples in second sample (Y)
    '''
    p = np.random.permutation(n+m)
    sample_1 = p[:n]
    X_inds_sample_1 = sample_1[sample_1 < n] # numbers < n are indices for samples in first sample (X)
    Y_inds_sample_1 = sample_1[sample_1 >= n] - n # numbers >= n are indices for samples in second sample (Y) but need to subtract n to start from 0
    sample_2 = p[n:]
    X_inds_sample_2 = sample_2[sample_2 < n]
    Y_inds_sample_2 = sample_2[sample_2 >= n] - n

    return X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2

def get_permuted_kernel_sum(K_XX: torch.tensor, K_YY: torch.tensor, K_XY: torch.tensor, inds_X: np.ndarray, inds_Y: np.ndarray) -> torch.tensor:
    '''
    Calculate the sum of the permuted gram matrix
    gram_X: torch.tensor of shape (n, n)
    gram_Y: torch.tensor of shape (m, m)
    inds_X: 1D torch.tensor up to size n and containing integers in [0,n]
    inds_Y: 1D torch.tensor up to size m and containing integers in [0,m]
    '''
    if len(inds_X) == 0:
        return torch.sum(K_YY[np.ix_(inds_Y, inds_Y)]) # a[np.ix_([1,3],[2,5])] returns the array [[a[1,2] a[1,5]], [a[3,2] a[3,5]]]
    elif len(inds_Y) == 0:
        return torch.sum(K_XX[np.ix_(inds_X, inds_X)])
    else:
        return torch.sum(K_XX[np.ix_(inds_X, inds_X)]) + torch.sum(K_YY[np.ix_(inds_Y, inds_Y)]) + 2 * torch.sum(K_XY[np.ix_(inds_X, inds_Y)])

def get_permuted_cross_kernel_sum(K_XX: torch.tensor, K_YY: torch.tensor, K_XY: torch.tensor,
                                  inds_X_1: np.ndarray, inds_Y_1: np.ndarray, inds_X_2: np.ndarray, inds_Y_2: np.ndarray) -> torch.tensor:
    '''
    Calculate the sum of the permuted cross gram matrix
    K_XX: torch.tensor of shape (n, n)
    K_YY: torch.tensor of shape (m, m)
    K_XY: torch.tensor of shape (n, m)
    inds_X_1: 1D torch.tensor up to size n and containing integers in [0,n]
    inds_Y_1: 1D torch.tensor up to size m and containing integers in [0,m]
    inds_X_2: 1D torch.tensor up to size n and containing integers in [0,n]
    inds_Y_2: 1D torch.tensor up to size m and containing integers in [0,m]
    '''
    if K_XY.shape[0] == K_XY.shape[1]: # if X and Y have the same number of samples
        if len(inds_X_1) == 0:
            assert len(inds_Y_2) == 0, 'if inds_X_1 is empty then inds_Y_2 must also be empty'
            return torch.sum(K_XY)
        elif len(inds_Y_1) == 0:
            assert len(inds_X_2) == 0, 'if inds_Y_1 is empty then inds_X_2 must also be empty'
            return torch.sum(K_XY)
    else:
        if len(inds_X_1) == 0: # first permuted sample is all from Y
            return torch.sum(K_YY[np.ix_(inds_Y_1, inds_Y_2)]) + torch.sum(K_XY[np.ix_(inds_X_2, inds_Y_1)])
        elif len(inds_Y_1) == 0: # first permuted sample is all from X
            return torch.sum(K_XX[np.ix_(inds_X_1, inds_X_2)]) + torch.sum(K_XY[np.ix_(inds_X_1, inds_Y_2)])
        elif len(inds_X_2) == 0: # second permuted sample is all from Y
            return torch.sum(K_YY[np.ix_(inds_Y_1, inds_Y_2)]) + torch.sum(K_XY[np.ix_(inds_X_1, inds_Y_2)])
        elif len(inds_Y_2) == 0: # second permuted sample is all from X
            return torch.sum(K_XX[np.ix_(inds_X_1, inds_X_2)]) + torch.sum(K_XY[np.ix_(inds_X_1, inds_Y_1)])

    # if we get here then both permuted samples are a mix of X and Y
    # a[np.ix_([1,3],[2,5])] returns the array [[a[1,2] a[1,5]], [a[3,2] a[3,5]]]
    return (torch.sum(K_XX[np.ix_(inds_X_1, inds_X_2)]) +
            torch.sum(K_YY[np.ix_(inds_Y_1, inds_Y_2)]) +
            torch.sum(K_XY[np.ix_(inds_X_1, inds_Y_2)]) +
            torch.sum(K_XY[np.ix_(inds_X_2, inds_Y_1)]))

def sig_kernel_test(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel, num_permutations: float=1000,
                    robust: bool=True, a: float=1., C: float=4.,
                    stats_plot: bool=True, percentile: float=0.9) -> (float, np.ndarray):
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    static_kernel: static kernel to be lifted to sequence kernel e.g. LinearKernel or RBFKernel
    percentile: float, percentile to be used for the test
    num_permutations: int, number of permutations to be used to generate the null distribution
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    with torch.no_grad():
        if robust:
            K_XX, K_YY, K_XY = get_robust_gram_matrices(X, Y, kernel, a, C)
        else:
            K_XX = kernel(X,X)
            K_YY = kernel(Y,Y)
            K_XY = kernel(X,Y)
            K_XX.fill_diagonal_(0)
            K_YY.fill_diagonal_(0)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)
    mmd = (torch.sum(K_XX) / (n*(n-1))  + torch.sum(K_YY) / (m*(m-1))  - 2*torch.sum(K_XY)/(n*m)).item()

    null_mmds = np.zeros(num_permutations)
    for i in range(num_permutations):
        X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2 = get_permutation_indices(n, m)
        perm_K_XX_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1)
        perm_K_YY_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_2, Y_inds_sample_2)
        perm_K_XY_sum = get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2)
        null_mmds[i] = (perm_K_XX_sum / (n*(n-1))  + perm_K_YY_sum / (m*(m-1))  - 2*perm_K_XY_sum/(n*m)).item()

    if stats_plot:
        plot_permutation_samples(null_mmds, statistic=mmd, percentile=percentile)

    return mmd, null_mmds

def mmd_permutation_ratio_plot(X, Y, n_levels, static_kernel, n_steps=10, a=1, C=4):
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    static_kernel: static kernel to be lifted to sequence kernel e.g. LinearKernel or RBFKernel
    quantile: float, quantile to be used for the test
    num_permutations: int, number of permutations to be used to generate the null distribution
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    kernel = SignatureKernel(n_levels=n_levels, order=n_levels, normalization=3, static_kernel=static_kernel)

    # calculate Gram matrices
    K_XX, K_YY, K_XY = get_robust_gram_matrices(X, Y, kernel, n_levels, a, C)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)
    mmd = np.sum(K_XX) / (n*(n-1)) + np.sum(K_YY) / (m*(m-1)) - 2*np.sum(K_XY)/(n*m)

    plot_mmd_splits(K_XX, K_YY, K_XY, n, m, mmd, n_steps)

def plot_mmd_splits(K_XX, K_YY, K_XY, n, m, mmd, n_steps=10):
    mmd_splits = np.empty((2, n_steps+1))
    for i in range(n_steps+1):
        split = i / n_steps
        split_x = int(split * n)
        split_y = int(split * m)
        X_inds_sample_1 = np.arange(split_x)
        Y_inds_sample_1 = np.arange(split_y, m)
        X_inds_sample_2 = np.arange(split_x, n)
        Y_inds_sample_2 = np.arange(split_y)

        perm_K_XX_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1)
        perm_K_YY_sum = get_permuted_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_2, Y_inds_sample_2)
        perm_K_XY_sum = get_permuted_cross_kernel_sum(K_XX, K_YY, K_XY, X_inds_sample_1, Y_inds_sample_1, X_inds_sample_2, Y_inds_sample_2)

        mmd_splits[0, i] = split
        mmd_splits[1, i] = perm_K_XX_sum / (n*(n-1))  + perm_K_YY_sum / (m*(m-1))  - 2*perm_K_XY_sum/(n*m)

    plt.plot(mmd_splits[0], mmd_splits[1])
    plt.axhline(y=mmd, c='r')
    legend = ['MMD at different split ratios', 'Actual test statistic']
    plt.legend(legend)

def plot_permutation_samples(null_samples, statistic=None, percentile=0.9, two_tailed=False):
    plt.hist(null_samples, bins=100)

    if two_tailed:
        plt.axvline(x=np.percentile(null_samples, 50 * (1 + percentile)), c='b')
    else:
        plt.axvline(x=np.percentile(null_samples, 100*percentile), c='b')
    legend = [f'{int(100*percentile)} percentile']

    if statistic is not None:
        percentile = (null_samples < statistic).sum() / len(null_samples)
        plt.axvline(x=statistic, c='r')
        legend += [f'Test statistic at {int(percentile*100)} percentile']

    plt.legend(legend)
    plt.xlabel('Test statistic value')
    plt.ylabel('Counts')

def gen_GBM_path(initial_value, mu, sigma, dt, n_paths, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = seq_len - 1
    path = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal((n_paths, n_steps)))
    path = np.cumprod(path, axis=1)
    path = np.concatenate([np.ones((n_paths, 1)), path], axis=1)
    return path[..., np.newaxis] * initial_value # shape (n_paths, seq_len, 1)

def gen_OU_path(initial_value, kappa, theta, sigma, dt, n_paths, seq_len, seed=None):
    rng = np.random.default_rng(seed)
    n_steps = seq_len - 1
    paths = np.ones((n_paths, seq_len)) * initial_value
    non_path_dependent_part = theta * (1 - np.exp(-kappa * dt)) + sigma / np.sqrt(2 * kappa) * np.sqrt(1 - np.exp(-2 * kappa * dt)) * rng.standard_normal((n_paths, n_steps))
    for t in range(n_steps):
        paths[:, t+1] = paths[:, t] * np.exp(-kappa * dt) + non_path_dependent_part[:, t]
    return paths[..., np.newaxis] # shape (n_paths, seq_len, 1)

def gen_quantlib_paths(process, dt, n_paths, seq_len, seed, return_all_paths):

    times = ql.TimeGrid((seq_len-1)*dt, seq_len-1) # creates list of times starting from 0 to (seq_len-1)*dt with step size dt
    dimension = process.factors() # 2 factors for Heston model i.e. spot and vol

    randomGenerator = ql.UniformRandomGenerator() if seed is None else ql.UniformRandomGenerator(seed=seed) # seed of 0 seems to not set a seed
    rng = ql.UniformRandomSequenceGenerator(dimension * (seq_len-1), randomGenerator)
    sequenceGenerator = ql.GaussianRandomSequenceGenerator(rng)
    pathGenerator = ql.GaussianMultiPathGenerator(process, list(times), sequenceGenerator, False)

    paths = [[] for i in range(dimension)]
    for _ in range(n_paths):
        samplePath = pathGenerator.next()
        values = samplePath.value()

        for j in range(dimension):
            paths[j].append([x for x in values[j]])

    if return_all_paths:
        return np.array(paths).transpose([1,2,0])
    else:
        return np.array(paths[0])[..., np.newaxis]

def gen_Heston_path(initial_value, mu, v0, kappa, theta, rho, sigma, dt, n_paths, seq_len, seed=None, return_vols=False):
    today = ql.Date().todaysDate()
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, mu, ql.Actual365Fixed()))
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.00, ql.Actual365Fixed()))
    initialValue = ql.QuoteHandle(ql.SimpleQuote(initial_value))

    hestonProcess = ql.HestonProcess(riskFreeTS, dividendTS, initialValue, v0, kappa, theta, sigma, rho)

    return gen_quantlib_paths(hestonProcess, dt, n_paths, seq_len, seed=seed, return_all_paths=return_vols)

@torch.compile
def mmd_loss(X: torch.tensor, Y: torch.tensor, kernel: SignatureKernel, robust: bool=True, a: float=1., C: float=4.) -> torch.tensor:
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    kernel: kernel to be used e.g. SignatureKernel
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    # if torch.cuda.device_count() > 3:
    #     device = torch.device('cuda:3')

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    if robust:
        K_XX, K_YY, K_XY = get_robust_gram_matrices(X, Y, kernel, a, C)
    else:
        K_XX = kernel(X,X)#.to(device)
        K_YY = kernel(Y,Y)#.to(device)
        K_XY = kernel(X,Y)#.to(device)
        K_XX.fill_diagonal_(0)
        K_YY.fill_diagonal_(0)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)

    mmd = torch.sum(K_XX) / (n*(n-1))  + torch.sum(K_YY) / (m*(m-1)) - 2*torch.sum(K_XY)/(n*m)

    return mmd#.to(X.device)

def mmd_loss_no_compile(X, Y, kernel, robust, a, C) -> torch.tensor:
    '''
    X: torch.tensor of shape (n_samples, n_features)
    Y: torch.tensor of shape (n_samples, n_features)
    kernel: kernel to be used e.g. SignatureKernel
    n_levels: int, number of levels of the signature kernel where level 0 is not included e.g. 3 means signature (t0, t1, t2, t3)
    a: float, parameter for psi function used in the tensor normalisation to get the characteristic signature kernel
    C: float, as above
    '''
    assert X.ndim == Y.ndim

    # calculate Gram matrices with normalisation and diagonal of XX/YY zeroed
    if robust:
        K_XX, K_YY, K_XY = get_robust_gram_matrices(X, Y, kernel, a, C)
    else:
        K_XX = kernel(X,X)
        K_YY = kernel(Y,Y)
        K_XY = kernel(X,Y)
        K_XX.fill_diagonal_(0)
        K_YY.fill_diagonal_(0)

    # unbiased MMD statistic (could also use biased, doesn't matter if we use permutation tests)
    n = len(K_XX)
    m = len(K_YY)

    mmd = torch.sum(K_XX) / (n*(n-1))  + torch.sum(K_YY) / (m*(m-1)) - 2*torch.sum(K_XY)/(n*m)

    return mmd

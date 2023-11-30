from typing import Optional, Union, List, Tuple
import numpy as np
import torch

def split_tensor_gpu(X: torch.Tensor, device_ids: List[int], dim:int = 0) -> List[torch.Tensor]:
    '''
    Splits a tensor into chunks and moves them to different GPUs.
    Remaining size n < num_devices are assigned to equally to the first n GPUs in the device_ids list.
    '''
    chunks = np.zeros(len(device_ids), dtype=int)
    chunks[:X.shape[dim] % len(device_ids)] = 1
    chunks += np.array([X.shape[dim] // len(device_ids)] * len(device_ids))
    chunks = list(chunks)

    split_tensors = []
    for i, chunk in enumerate(X.split(chunks, dim=dim)):
        if chunk.device.index == device_ids[i]:
            split_tensors.append(chunk)
        else:
            split_tensors.append(chunk.to(f"cuda:{device_ids[i]}"))
    return split_tensors

class SplitTensor:
    def __init__(self, X: torch.Tensor|List[torch.Tensor], device_ids: List[int], dim: int):
        if type(X) == torch.Tensor:
            self.device_ids = device_ids
            self.split_tensors = split_tensor_gpu(X, device_ids, dim)
            self.dim = dim
            self.device_ids = device_ids
            self.shape = list(X.shape)
            self.dim_size = X.shape[dim]
        else:
            self.split_tensors = X
            self.dim = dim
            size = 0
            self.shape = list(X[0].shape)
            if len(X) > 1:
                self.device_ids = []
                for tensor in X:
                    self.device_ids.append(tensor.device.index)
                    size += tensor.shape[dim]
                    self.shape[dim] = size
                    self.dim_size = size
            else:
                self.device_ids = [X[0].device.index]
                self.dim_size = 0

    def recombine(self, out_device=None):
        '''
        Combines split tensors from different GPUs onto a single GPU.
        Either the GPU of the first tensor in the list or the one specified by out_device.
        '''
        if out_device is None:
            if self.dim is None:
                return self.split_tensors[0]
            else:
                return torch.cat([tensor.to(self.split_tensors[0].device) if tensor.device.index != self.split_tensors[0].device.index else tensor
                                  for tensor in self.split_tensors], dim=self.dim)
        else:
            if self.dim is None:
                if self.split_tensors[0].device.index == out_device:
                    return self.split_tensors[0]
                else:
                    return self.split_tensors[0].to(out_device)
            else:
                return torch.cat([tensor.to(out_device) if tensor.device.index != out_device else tensor
                                  for tensor in self.split_tensors], dim=self.dim)

    def resplit_to_gpu(self, device_ids=None, dim=None):
        '''
        Resplits unbalanced split tensors to given devices
        '''
        if not (device_ids is None):
            self.device_ids = device_ids

        self.dim = dim if dim else self.dim

        size = 0
        for tensor in self.split_tensors:
            size += tensor.shape[self.dim]
        chunks = np.zeros(len(self.device_ids), dtype=int)
        chunks[:size % len(self.device_ids)] = 1
        chunks += np.array([size // len(self.device_ids)] * len(self.device_ids))
        chunks = list(chunks)
        resplit_tensors = []
        excess = []
        for i, chunk in enumerate(chunks):
            tensor_size = self.split_tensors[i].shape[self.dim]
            if len(excess) > 0: tensor_size += excess[0].shape[self.dim]
            if chunk == tensor_size:
                if len(excess) > 0:
                    if excess[0].device.index != device_ids[i]:
                        excess[0] = excess[0].to(f"cuda:{self.device_ids[i]}")
                    if self.split_tensors[i].device.index != device_ids[i]:
                        self.split_tensors[i] = self.split_tensors[i].to(f"cuda:{self.device_ids[i]}")
                    resplit_tensors.append(torch.cat([excess[0], self.split_tensors[i]], dim=self.dim))
                    excess = []
                else:
                    resplit_tensors.append(self.split_tensors[i])
            elif chunk < tensor_size:
                remainder = tensor_size - chunk
                if len(excess) > 0:
                    excess_size = excess[0].shape[self.dim]
                    if chunk > excess_size:
                        if excess[0].device.index != device_ids[i]:
                            excess[0] = excess[0].to(f"cuda:{self.device_ids[i]}")
                        if self.split_tensors[i][:chunk-excess_size].device.index != device_ids[i]:
                            self.split_tensors[i][:chunk-excess_size] = self.split_tensors[i][:chunk-excess_size].to(f"cuda:{self.device_ids[i]}")
                        resplit_tensors.append(torch.cat([excess[0], self.split_tensors[i][:chunk-excess_size]], dim=self.dim))
                        excess = []
                        excess.append(self.split_tensors[i][chunk-excess_size:].to(f"cuda:{self.device_ids[i+1]}"))
                    else:
                        raise NotImplementedError
                else:
                    for j, tensor in enumerate(self.split_tensors[i].split([chunk, remainder], dim=self.dim)):
                        if j == 0:
                            tensor = tensor.to(f"cuda:{self.device_ids[i]}") if tensor.device.index != device_ids[i] else tensor
                            resplit_tensors.append(tensor)
                        else:
                            tensor = tensor.to(f"cuda:{self.device_ids[i+1]}") if tensor.device.index != device_ids[i+1] else tensor
                            excess.append(tensor)
            else:
                # not implemented
                raise NotImplementedError
        self.split_tensors = resplit_tensors

    def clone(self):
        '''
        Clones split tensors
        '''
        cloned_tensors = []
        for tensor in self.split_tensors:
            cloned_tensors.append(tensor.clone())
        return SplitTensor(cloned_tensors, self.device_ids, dim=self.dim)

    def unsqueeze(self, dim):
        '''
        Unsqueezes split tensors along a given dimension
        '''
        for tensor in self.split_tensors:
            tensor = tensor.unsqueeze(dim=dim)
        if dim <= self.dim:
            self.dim += 1
        self.shape = list(self.split_tensors[0].shape)
        self.shape[self.dim] = self.dim_size

    def slice(self, idx):
        '''
        Slices split tensors
        '''
        sliced_tensors = []
        # self.dim needs to be reduced by 1 for each scalar index before self.dim
        dim = self.dim
        for i, ax in enumerate(idx):
                if np.isscalar(ax):
                    dim -= 1
                if i == self.dim:
                    if np.isscalar(ax): raise NotImplementedError
                    break
        for tensor in self.split_tensors:
            sliced_tensors.append(tensor[idx])
        return SplitTensor(sliced_tensors, self.device_ids, dim=dim)

    def assign(self, idx, value):
        '''
        Assigns a value to a given index in a split tensor
        '''
        # only implemented for assigning to indices before self.dim or all of self.dim is selected
        if len(idx) <= self.dim or idx[self.dim] == slice(None):
            for i, tensor in enumerate(self.split_tensors):
                tensor[idx] = value.split_tensors[i]#.to(tensor.device)
        else:
            raise NotImplementedError
            # get the arg in self.split_tensors that contains the index
            # sizes = []
            # for tensor in self.split_tensors:
            #     sizes.append(tensor.shape[self.dim])
            # arg = np.argmax(np.array(sizes) > idx[self.dim])
            # cum_sizes = np.cumsum(sizes)
            # idx[0] = idx[0] - cum_sizes[arg-1] if arg > 0 else idx[0]
            # self.split_tensors[arg][idx] = value.split_tensor.to(self.split_tensors[arg].device)

    def div(self, scalar):
        '''
        Divides split tensors by a scalar
        '''
        for tensor in self.split_tensors:
            tensor /= scalar

    def cumsum(self, dim=0, inplace=False):
        '''
        Computes cumulative sum of split tensors along a given dimension.
        '''
        sum_tensors = []
        if dim == self.dim:
            sum_tensors.append(self.split_tensors[0].cumsum(dim=dim))
            for tensor in self.split_tensors[1:]:
                # get the slice with last element of the selected dimension and : everywhere else
                slc = [slice(None)] * tensor.ndim
                slc[dim] = slice(-1, None)
                prev_sum = sum_tensors[-1][slc]
                # add the last element of the previous cumsum to the cumsum of the current tensor
                if prev_sum.device.index != tensor.device.index:
                    prev_sum = prev_sum.to(tensor.device)
                sum_tensors.append(prev_sum + tensor.cumsum(dim=dim))
        else:
            for tensor in self.split_tensors:
                sum_tensors.append(torch.cumsum(tensor, dim=dim))
        if inplace:
            self.split_tensors = sum_tensors
        else:
            return SplitTensor(sum_tensors, self.device_ids, dim=self.dim)

    def _sum(self, dim=0, inplace=False):
        '''
        Computes sum of split tensors along a given dimension.
        '''
        sum_tensors = []
        # sum tensors along the given dimension
        for tensor in self.split_tensors:
            sum_tensors.append(tensor.sum(dim=dim))

        if dim == self.dim:
            # if the sum is on the split dimension, sum the tensors for the final result
            device = sum_tensors[0].device
            sum_tensor = sum_tensors[0]
            for tensor in sum_tensors[1:]:
                sum_tensor += tensor.to(device) if tensor.device.index != device.index else tensor
            # update self.dim to None as all tensors are on the same device hence no longer split
            if inplace:
                self.shape = list(sum_tensor.shape)
                self.dim_size = 0
                self.dim = None
                self.split_tensors = [sum_tensor]
            else:
                return SplitTensor([sum_tensor], [device], dim=None)
        else:
            if inplace:
                if self.dim:
                    if dim < self.dim:
                        self.dim -= 1
                    self.shape = list(sum_tensors[0].shape)
                    self.shape[self.dim] = self.dim_size
                self.split_tensors = sum_tensors
            else:
                return SplitTensor(sum_tensors, self.device_ids, dim=(self.dim if self.dim is None or dim > self.dim else self.dim-1))

    def sum(self, dim=0, inplace=False):
        '''
        Computes sum of split tensors along a given dimension(s).
        '''
        if np.isscalar(dim):
            if inplace:
                self._sum(dim=dim, inplace=inplace)
            else:
                return self._sum(dim=dim, inplace=inplace)
        else:
            ndim = self.split_tensors[0].ndim
            dim = [ndim+ax if ax < 0 else ax for ax in dim]
            temp_tensors = self.clone()

            for d in reversed(dim):
                if inplace:
                    self._sum(dim=d, inplace=inplace)
                else:
                    temp_tensors = temp_tensors._sum(dim=d, inplace=inplace)

            if not inplace:
                return temp_tensors

    def exclude_last(self, dim, inplace=False):
        ndim = self.split_tensors[0].ndim
        exclude_tensors = []

        # create slice to exclude last element along the given dim(s) except for the split dim
        slices = tuple(slice(-1) if (ax != self.dim) and (ax in dim) else slice(None) for ax in range(ndim))
        for tensor in self.split_tensors:
            exclude_tensors.append(tensor[slices])

        # for the split dim, slice off the last element
        if self.dim in dim:
            slc = [slice(None)] * ndim
            slc[self.dim] = slice(-1, None)
            exclude_tensors[-1] = exclude_tensors[-1][slc]

        if inplace:
            self.split_tensors = exclude_tensors
            self.resplit_to_gpu()
        else:
            exclude_tensors = SplitTensor(exclude_tensors, self.device_ids, dim=self.dim)
            exclude_tensors.resplit_to_gpu()
            return exclude_tensors

    def pad_first(self, dim, inplace=False):
        ndim = self.split_tensors[0].ndim
        padded_tensors = []

        # create slice to pad first element along the given dim(s) except for the split dim
        pads = tuple(x for ax in reversed(range(ndim)) for x in ((1, 0) if (ax != self.dim) and (ax in dim) else (0, 0)))
        for tensor in self.split_tensors:
            padded_tensors.append(torch.nn.functional.pad(tensor, pads))

        # for the split dim, pad first element
        if self.dim in dim:
            pads = tuple(x for ax in reversed(range(ndim)) for x in((0,0) if (ax != self.dim) else (1,0)))
            padded_tensors[self.dim] = torch.nn.functional.pad(padded_tensors[self.dim], pads)

        if inplace:
            self.split_tensors = padded_tensors
            self.resplit_to_gpu()
        else:
            padded_tensors = SplitTensor(padded_tensors, self.device_ids, dim=self.dim)
            padded_tensors.resplit_to_gpu()
            return padded_tensors

    def multi_cumsum(self, exclusive=False, dim=-1, inplace=False):
        ndim = self.split_tensors[0].ndim
        dim = [dim] if np.isscalar(dim) else dim
        dim = [ndim+ax if ax < 0 else ax for ax in dim]

        if inplace:
            # create slice for exclusive cumsum (slice off last element along given dim then pre-pad with zeros)
            if exclusive:
                self.exclude_last(dim=dim, inplace=True)

            # compute actual cumsums
            for ax in dim:
                self.cumsum(dim=ax, inplace=True)

            # pre-pad with zeros along the given dim if exclusive cumsum
            if exclusive:
                self.pad_last(dim=dim, inplace=True)
        else:
            if exclusive:
                temp_tensors = self.exclude_last(dim=dim, inplace=False)

            for ax in dim:
                temp_tensors = temp_tensors.cumsum(dim=ax, inplace=False)

            if exclusive:
                temp_tensors = temp_tensors.pad_first(dim=dim, inplace=False)

            return temp_tensors

def split_multiply(A: SplitTensor, B: SplitTensor) -> SplitTensor:
    '''
    Multiplies split tensors element-wise.
    Assumes A and B are split tensors and have the same number of chunks and sizes on the same devices
    '''
    if A.shape != B.shape:
        raise ValueError(f'Split tensors must have the same shape. A.shape = {A.shape}, B.shape = {B.shape}')
    C = []
    for a, b in zip(A.split_tensors, B.split_tensors):
        C.append(a * b)
    return SplitTensor(C, A.device_ids, dim=A.dim)
from abc import ABC, abstractmethod

import gpytorch
import torch
import torch.nn as nn


class KernelModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        pass


class ScalarProductKernel(KernelModule):
    def __init__(self, diag: bool = False):
        super().__init__()
        self.diag = diag

    def forward(self, x1, x2):
        out = torch.matmul(x1, x2.T)
        if self.diag:
            return torch.diag(out)
        else:
            return out


class CosineDistanceKernel(KernelModule):
    def forward(self, x1, x2):
        normalized_input_a = torch.nn.functional.normalize(x1)
        normalized_input_b = torch.nn.functional.normalize(x2)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        res *= -1  # 1-res without copy
        res += 1
        return res


class NegCosineDistanceKernel(CosineDistanceKernel):
    def forward(self, x1, x2):
        res = super().forward(x1, x2)
        return -res


class NNKernel(KernelModule):
    def __init__(
            self, input_dim: int, output_dim: int, num_layers: int, hidden_dim: int,
            kernel_op: KernelModule = ScalarProductKernel(),
            flatten: bool = False,
    ):
        super().__init__()
        assert num_layers >= 0, "Number of hidden layers must be at least 0"
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.flatten = flatten
        self.model = self.create_model()
        self.kernel_op = kernel_op

    def create_model(self):

        if self.num_layers == 0:
            assert (
                self.input_dim == self.output_dim
            ), f"If number of layers is 0, {self.input_dim=} and {self.output_dim=} should be equal!"

            modules = []
        elif self.num_layers == 1:
            modules = [nn.Linear(self.input_dim, self.output_dim)]
        elif self.num_layers >= 2:
            modules = [nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU()]
            if self.flatten:
                modules = [nn.Flatten()] + modules
            for i in range(self.num_layers - 1):
                modules.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                modules.append(nn.ReLU())
            modules.append(nn.Linear(self.hidden_dim, self.output_dim))
        else:
            raise TypeError(self.num_layers)
        model = nn.Sequential(*modules)
        return model

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, full_covar=True, **params):
        r"""
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            :attr:`last_dim_is_batch` (tuple, optional):
                If this is true, it treats the last dimension of the data as another batch dimension.
                (Useful for additive structure over the dimensions). Default: False

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
        """
        if last_dim_is_batch:
            raise NotImplementedError()
        else:

            z1 = self.model(x1)
            z2 = self.model(x2)
            return self.kernel_op(z1, z2)


class PositiveLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input):
        w = nn.functional.softplus(self.weight)
        return nn.functional.linear(input, w)


class NNKernelNoInner(gpytorch.kernels.Kernel):
    def __init__(self, input_dim, num_layers, hidden_dim, flatten=False, **kwargs):
        super(NNKernelNoInner, self).__init__(**kwargs)

        self.input_dim = input_dim * 2
        self.output_dim = 1
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.flatten = flatten
        self.model = self.create_model()

    def create_model(self):

        assert self.num_layers >= 1, "Number of hidden layers must be at least 1"
        modules = [PositiveLinear(self.input_dim, self.hidden_dim), nn.Sigmoid()]
        if self.flatten:
            modules = [nn.Flatten()] + modules
        for i in range(self.num_layers - 1):
            modules.append(PositiveLinear(self.hidden_dim, self.hidden_dim))
            modules.append(nn.Sigmoid())
        modules.append(PositiveLinear(self.hidden_dim, self.output_dim))

        model = nn.Sequential(*modules)
        return model

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, full_covar=True, **params):
        r"""
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            :attr:`last_dim_is_batch` (tuple, optional):
                If this is true, it treats the last dimension of the data as another batch dimension.
                (Useful for additive structure over the dimensions). Default: False

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
        """
        if last_dim_is_batch:
            raise NotImplementedError()
        else:
            n = x1.shape[0]
            m = x2.shape[0]
            out = torch.zeros((n, m), device=x1.get_device())

            for i in range(n):
                for j in range(i + 1):
                    out[i, j] = self.model(torch.cat((x1[i], x2[j]))).view(-1)
                    if i != j:
                        out[j, i] = out[i, j]

            # npout = out.cpu().detach().numpy()
            # print(np.linalg.eigvals(npout))
            # assert np.all(np.linalg.eigvals(npout) +1e-2 >= 0), "not positive"
            if diag:
                return torch.diag(out)
            else:
                return out


class MultiNNKernel(gpytorch.kernels.Kernel):
    def __init__(self, num_tasks, kernels, **kwargs):
        super(MultiNNKernel, self).__init__(**kwargs)
        assert isinstance(kernels, list), "kernels must be a list of kernels"
        self.num_tasks = num_tasks
        self.kernels = nn.ModuleList(kernels)

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, full_covar=True, **params):
        r"""
        Computes the covariance between x1 and x2.
        This method should be imlemented by all Kernel subclasses.

        Args:
            :attr:`x1` (Tensor `n x d` or `b x n x d`):
                First set of data
            :attr:`x2` (Tensor `m x d` or `b x m x d`):
                Second set of data
            :attr:`diag` (bool):
                Should the Kernel compute the whole kernel, or just the diag?
            :attr:`last_dim_is_batch` (tuple, optional):
                If this is true, it treats the last dimension of the data as another batch dimension.
                (Useful for additive structure over the dimensions). Default: False

        Returns:
            :class:`Tensor` or :class:`gpytorch.lazy.LazyTensor`.
                The exact size depends on the kernel's evaluation mode:

                * `full_covar`: `n x m` or `b x n x m`
                * `full_covar` with `last_dim_is_batch=True`: `k x n x m` or `b x k x n x m`
                * `diag`: `n` or `b x n`
                * `diag` with `last_dim_is_batch=True`: `k x n` or `b x k x n`
        """
        if last_dim_is_batch:
            raise NotImplementedError()
        else:
            n = x1.shape[0]
            m = x2.shape[0]
            out = torch.zeros((n * self.num_tasks, m * self.num_tasks), device=x1.get_device())
            for i in range(self.num_tasks):
                for j in range(self.num_tasks):
                    z1 = self.kernels[i].model(x1)
                    z2 = self.kernels[j].model(x2)

                    out[i:n * self.num_tasks:self.num_tasks, j:m * self.num_tasks:self.num_tasks] = torch.matmul(z1,
                                                                                                                 z2.T)
            if diag:
                return torch.diag(out)
            else:
                return out

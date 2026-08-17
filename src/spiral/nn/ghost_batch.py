import torch
from torch import nn


class GhostBatchNorm1D(nn.Module):
    def __init__(self,
            n_features: int,
            batch_size: int,
            mini_batch_size: int,
            epsilon: float =1e-5,
            momentum: float=0.1):
        """
        Implementation of Ghost Batch Normalization for 1d input, paper here: https://arxiv.org/abs/1705.08741

        Args:
        n_features: Number of features
        batch_size (b_l): Size of the Large Batch
        mini_batch_size (b_s): Size of the ghost batch
        eps: Constant added to the variance
        momentum: Used to estimate the running mean and std
        """
        super().__init__()
        self.register_buffer("running_mean", torch.zeros([1, n_features]))
        self.register_buffer("running_std", torch.ones([1,n_features]))
        self.shift = nn.Parameter(torch.zeros(n_features))
        self.scale = nn.Parameter(torch.ones(n_features))
        self.momentum = momentum
        self.epsilon = epsilon
        self.b_l = batch_size
        self.b_s = mini_batch_size
        self.n_chunks = self.get_chunks()
        self.n_features = n_features

    def get_chunks(self):
        "Returns the number of chunks"
        if self.b_l % self.b_s != 0:
            raise ValueError(f"Large Batch must be divisible by small batch size, {self.b_l} % {self.b_s} > 0 ")
        return self.b_l // self.b_s

    def compute_stats(self, X: torch.Tensor):
        """Returns the mean and std of the batched input"""
        mean = X.mean(dim=1)
        var = X.var(dim=1)
        std = torch.sqrt(var + self.epsilon)
        return mean, std

    # def update_running(self, mean: torch.Tensor, std: torch.Tensor):
    #     """Updates the running mean and average based on the paper"""
    #     with torch.no_grad():
    #         device = self.running_mean.device
    #         inner_momentum = torch.full([self.n_chunks], (1 - self.momentum), device=device)
    #         exponents = torch.arange(1, self.n_chunks + 1, device=device)
    #         inner_momentum = (torch.pow(inner_momentum, exponents) * self.momentum).unsqueeze(dim=1)
    #         self.running_mean = ((1 - self.momentum)**self.n_chunks * self.running_mean) + (mean * inner_momentum).sum(dim=0)
    #         self.running_std = ((1 - self.momentum)**self.n_chunks * self.running_std) + (std * inner_momentum).sum(dim=0)

    def update_running(self, mean: torch.Tensor, std: torch.Tensor):
        """Updates the running mean and std based on the paper"""
        with torch.no_grad():
            device = self.running_mean.device
            exponents = torch.arange(0, self.n_chunks, device=device)  # 0 ... n-1, not 1 ... n
            weights = (self.momentum * (1 - self.momentum) ** exponents).unsqueeze(dim=1)
            decay = (1 - self.momentum) ** self.n_chunks
            self.running_mean = decay * self.running_mean + (mean * weights).sum(dim=0)
            self.running_std = decay * self.running_std + (std * weights).sum(dim=0)


    def normalize(self, X, mean, std):
        """Normalizes the input based on the mean and std"""
        return (X - mean) / std


    def forward(self, X: torch.Tensor):
        mean, std = self.running_mean, self.running_std
        if self.training:
            X = X.reshape(self.n_chunks, self.b_s, self.n_features)
            mean, std = self.compute_stats(X)
            self.update_running(mean, std)
            mean, std = mean.unsqueeze(dim=1), std.unsqueeze(dim=1)

        norm_X = self.normalize(X, mean, std)
        if self.training:
            norm_X = norm_X.reshape(self.b_l, self.n_features)
        return (self.scale * norm_X) + self.shift
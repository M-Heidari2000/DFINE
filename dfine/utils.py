import torch


def pearson_corr(
    true: torch.Tensor,
    pred: torch.Tensor
):
    T, B, D = true.shape

    # mean and std along time dimension
    true_mean = true.mean(dim=0, keepdim=True)  # (1, B, D)
    pred_mean = pred.mean(dim=0, keepdim=True)
    true_std = true.std(dim=0, unbiased=False, keepdim=True)
    pred_std = pred.std(dim=0, unbiased=False, keepdim=True)

    # covariance across time
    cov = ((true - true_mean) * (pred - pred_mean)).mean(dim=0)  # (B, D)

    corr = cov / (true_std.squeeze(0) * pred_std.squeeze(0) + 1e-8)  # (B, D)
    return corr.mean()

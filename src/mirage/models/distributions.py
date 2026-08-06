from __future__ import annotations

import torch


def student_t_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    degrees_of_freedom: float = 5.0,
) -> torch.Tensor:
    distribution = torch.distributions.StudentT(
        df=torch.as_tensor(degrees_of_freedom, device=target.device),
        loc=location,
        scale=scale,
    )
    return -distribution.log_prob(target)


def gaussian_nll(
    target: torch.Tensor, location: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    distribution = torch.distributions.Normal(location, scale)
    return -distribution.log_prob(target)


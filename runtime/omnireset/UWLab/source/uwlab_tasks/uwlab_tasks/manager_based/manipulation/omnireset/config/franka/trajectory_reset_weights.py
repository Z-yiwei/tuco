"""Explicit pregrasp20% weights, preserving every original/new row."""
import torch


def reset_weights(count, path_index, device='cpu'):
    assert 0 <= path_index < 4
    old_count = 2500 if path_index == 0 else 5000
    assert count >= old_count
    if path_index == 0:
        return torch.full((count,), 1./count, dtype=torch.float64, device=device)
    ordinary_count = count - 2500
    result = torch.full((count,), .8/ordinary_count, dtype=torch.float64, device=device)
    result[2500:5000] = .2/2500
    assert abs(float(result.sum())-1) < 1e-10
    return result

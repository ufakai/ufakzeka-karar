"""The seed soup: several trained heads averaged into one network."""

from __future__ import annotations


def average_states(states: list[dict]) -> dict:
    """The element-wise mean of several state dicts (a seed soup)."""
    import torch

    out = {}
    for key, first in states[0].items():
        if torch.is_floating_point(first):
            out[key] = sum(s[key].float() for s in states) / len(states)
        else:
            out[key] = first
    return out

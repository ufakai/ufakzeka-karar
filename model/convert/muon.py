"""Muon, as ufakzeka-1-base was pretrained with it.

Taken from the lab's ufakzeka repository (Apache-2.0). The conversion uses it
because adapting a model with a different optimizer from the one that shaped it
is a measured cost, not a neutral choice (arXiv 2605.10468): the weights
sit where Muon's orthogonalised updates left them, and AdamW's per-coordinate
steps move them off that geometry before they move them anywhere useful.

Nesterov momentum, Newton-Schulz orthogonalisation in bfloat16, update scaling
by sqrt(max(1, rows/cols)), and the fixed-norm form the backbone's continued
pretraining used: each matrix is held at the Frobenius norm it started with,
which replaced weight decay there and beat it (arXiv 2606.16899, minimal form).
For the 2D weights inside the blocks only. Embeddings and norms use AdamW.

newton_schulz and the update scaling are adapted from Keller Jordan's Muon
(https://github.com/KellerJordan/Muon, zeropower_via_newtonschulz5 and
muon_update: the quintic coefficients, the transpose of a tall matrix, the
normalisation and the iteration), which is under the MIT License:

    MIT License

    Copyright (c) 2024 Keller Jordan

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

from __future__ import annotations

import torch


def newton_schulz(gradient: torch.Tensor, steps: int = 5) -> torch.Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = gradient.bfloat16()
    tall = gradient.size(0) > gradient.size(1)
    if tall:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.mT if tall else x


_orthogonalise = torch.compile(newton_schulz) if torch.cuda.is_available() else newton_schulz


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.95,
        fixed_norm: bool = True,
        ns_steps: int = 5,
    ):
        super().__init__(
            params, dict(lr=lr, momentum=momentum, fixed_norm=fixed_norm, ns_steps=ns_steps)
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p.grad)
                    # The norm to hold, taken once, from the weights as they
                    # arrive. It travels in the state, so a resume holds the
                    # same norm and not the one it happens to wake up with.
                    state["target_norm"] = p.norm().clone()
                buf = state["buf"]
                buf.mul_(group["momentum"]).add_(p.grad)
                update = _orthogonalise(p.grad.add(buf, alpha=group["momentum"]), group["ns_steps"])
                update = update * max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update.type_as(p), alpha=-group["lr"])
                if group["fixed_norm"]:
                    p.mul_(state["target_norm"] / (p.norm() + 1e-8))

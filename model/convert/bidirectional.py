"""Turning the causal backbone bidirectional, without forking the model code.

A decoder hides the future from every position. An encoder must not. The
backbone is a Qwen3-architecture causal model, and the obvious ways to change
that are all bad: forking `modeling_qwen3`, monkeypatching the module's
`create_causal_mask`, or subclassing to override the forward. Each of those
pins us to one version of the library and breaks quietly when it moves.

There is a supported path instead. `Qwen3Model.forward` checks whether the
`attention_mask` it was handed is already a mapping of layer type to prepared
mask, and uses it directly when it is; the library does this itself for
generation. So the conversion builds the mask with the library's own
`create_bidirectional_mask` and passes it in that form. No library code is
copied, patched or subclassed, and the attention implementation stays whatever
the config asks for (transformers 5.17.0, read 2026-09-21).

Whether documents packed into one window may attend to each other is a
separate question, answered by `packed` below.
"""

from __future__ import annotations

import torch
from transformers.masking_utils import create_bidirectional_mask, packed_sequence_mask_function

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


def bidirectional_masks(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    document_ids: torch.Tensor | None = None,
) -> dict[str, object]:
    """The mask mapping to hand to the backbone so it reads in both directions.

    `attention_mask` is the ordinary 2D padding mask. `document_ids` gives each
    position the index of the document it came from; supply it when several
    documents share a window and attention must not cross between them.

    The result is passed as the model's `attention_mask`, which the forward
    accepts as a prepared mapping.
    """
    config = model.config
    and_mask = packed_sequence_mask_function(document_ids) if document_ids is not None else None
    kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "and_mask_function": and_mask,
        # Never skipped. The skip returns None when every position may see
        # every other, and this model reads a None mask as "use my default",
        # which is causal. The one case that took the skip, no padding and no
        # document mask, therefore ran causally under a masked-language loss:
        # it trained, the loss fell, and nothing was bidirectional. That case
        # was also the fastest configuration the throughput probe found, which
        # is part of why it was fast. tests/test_native.py holds this to the
        # hand-written network, which has no such default.
        "allow_is_bidirectional_skip": False,
    }
    masks: dict[str, object] = {FULL_ATTENTION: create_bidirectional_mask(**kwargs)}
    # The flag lives on the inner model, not on the language-model wrapper, so
    # a wrapper passed here would always read False and lose the sliding mask.
    if getattr(getattr(model, "model", model), "has_sliding_layers", False):
        from transformers.masking_utils import create_bidirectional_sliding_window_mask

        masks[SLIDING_ATTENTION] = create_bidirectional_sliding_window_mask(**kwargs)
    return masks


def document_ids_for(input_ids: torch.Tensor, separator_id: int) -> torch.Tensor:
    """Number each packed document in a window, counting from the separators.

    A separator belongs to the document it closes, so a boundary token is never
    stranded in a document of its own.
    """
    is_separator = input_ids == separator_id
    # Shift so the separator keeps the index of the text before it.
    starts = torch.zeros_like(input_ids)
    starts[:, 1:] = is_separator[:, :-1].to(starts.dtype)
    return starts.cumsum(dim=1)

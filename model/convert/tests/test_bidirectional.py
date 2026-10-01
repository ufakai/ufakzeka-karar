"""Proof that the conversion actually reads both directions, on a tiny real model.

The test that matters is not that a mask was built. It is that a position's
output changes when a token *after* it changes, which is false for the causal
backbone and must be true after the conversion. Everything else in step 2 is
worthless if this is wrong, so it is checked against a real Qwen3 forward pass
rather than against a mask tensor.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3Model  # noqa: E402

from model.convert.bidirectional import (  # noqa: E402
    FULL_ATTENTION,
    bidirectional_masks,
    document_ids_for,
)

VOCAB = 64
SEP = 63


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    return Qwen3Model(config).eval()


def hidden(model, ids, masks=None):
    with torch.no_grad():
        return model(input_ids=ids, attention_mask=masks, use_cache=False).last_hidden_state


def test_the_backbone_is_causal_before_anything_is_done_to_it(model):
    ids = torch.randint(0, VOCAB - 1, (1, 8))
    changed = ids.clone()
    changed[0, -1] = (ids[0, -1] + 1) % (VOCAB - 1)
    before, after = hidden(model, ids), hidden(model, changed)
    # Changing the last token leaves every earlier position untouched.
    assert torch.allclose(before[:, :-1], after[:, :-1], atol=1e-6)


def test_after_the_conversion_an_earlier_position_sees_a_later_token(model):
    ids = torch.randint(0, VOCAB - 1, (1, 8))
    changed = ids.clone()
    changed[0, -1] = (ids[0, -1] + 1) % (VOCAB - 1)
    embeds = model.embed_tokens(ids)
    masks = bidirectional_masks(model, embeds)
    before = hidden(model, ids, masks)
    after = hidden(model, changed, masks)
    # This is the whole point of the conversion: the future is now visible.
    assert not torch.allclose(before[:, 0], after[:, 0], atol=1e-6)


def test_padding_is_still_hidden_when_reading_both_directions(model):
    ids = torch.randint(0, VOCAB - 1, (1, 8))
    padded = torch.cat([ids, torch.full((1, 4), 5)], dim=1)
    pad_mask = torch.cat([torch.ones(1, 8), torch.zeros(1, 4)], dim=1).long()
    embeds = model.embed_tokens(padded)
    masks = bidirectional_masks(model, embeds, pad_mask)
    with_pad = hidden(model, padded, masks)[:, :8]

    other = padded.clone()
    other[0, 8:] = 7  # different padding content, same padding positions
    with_other = hidden(model, other, masks)[:, :8]
    # What sits in the padded positions must not reach the real ones.
    assert torch.allclose(with_pad, with_other, atol=1e-6)


def test_packed_documents_do_not_read_each_other(model):
    # Two documents in one window, separated by SEP.
    ids = torch.tensor([[1, 2, 3, SEP, 10, 11, 12, SEP]])
    docs = document_ids_for(ids, SEP)
    assert docs.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]

    embeds = model.embed_tokens(ids)
    masks = bidirectional_masks(model, embeds, document_ids=docs)
    before = hidden(model, ids, masks)

    changed = ids.clone()
    changed[0, 5] = 20  # a token inside the second document only
    after = hidden(model, changed, masks)
    # The first document must be unaffected by the second.
    assert torch.allclose(before[:, :4], after[:, :4], atol=1e-6)
    # And the second document must notice its own change.
    assert not torch.allclose(before[:, 4:], after[:, 4:], atol=1e-6)


def test_without_document_ids_a_packed_window_does_leak_across_the_boundary(model):
    # The counterpart of the test above: this is what we are choosing to avoid.
    ids = torch.tensor([[1, 2, 3, SEP, 10, 11, 12, SEP]])
    embeds = model.embed_tokens(ids)
    masks = bidirectional_masks(model, embeds)
    before = hidden(model, ids, masks)
    changed = ids.clone()
    changed[0, 5] = 20
    after = hidden(model, changed, masks)
    assert not torch.allclose(before[:, :4], after[:, :4], atol=1e-6)


def test_the_mask_mapping_names_the_layer_types_the_model_asks_for(model):
    embeds = model.embed_tokens(torch.randint(0, VOCAB - 1, (1, 8)))
    masks = bidirectional_masks(model, embeds)
    assert FULL_ATTENTION in masks
    assert ("sliding_attention" in masks) == bool(getattr(model, "has_sliding_layers", False))


def test_document_ids_count_from_zero_and_keep_the_separator_with_its_own_text():
    ids = torch.tensor([[5, SEP, 6, 7, SEP, 8]])
    assert document_ids_for(ids, SEP).tolist() == [[0, 0, 1, 1, 1, 2]]

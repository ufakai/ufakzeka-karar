"""The hand-written network and the library class agree, weight for weight."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from model.convert.bidirectional import bidirectional_masks, document_ids_for  # noqa: E402
from model.convert.native import (  # noqa: E402
    NativeConfig,
    NativeModel,
    dense_document_mask,
    document_index,
)

SEP = 3


def library_model():
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=97,
        hidden_size=48,
        intermediate_size=96,
        num_hidden_layers=3,
        num_attention_heads=6,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 100000.0, "rope_type": "default"},
        tie_word_embeddings=True,
        attn_implementation="sdpa",
    )
    model = Qwen3ForCausalLM(config).eval()
    # Norm weights start at one, which would hide a norm wired to the wrong place.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.add_(torch.randn_like(parameter) * 0.1)
    return model


def native_from(library):
    native = NativeModel(NativeConfig.from_hf(library.config, softcap=0.0)).eval()
    native.load_hf_state(library.state_dict())
    return native


def ids_with_documents():
    torch.manual_seed(1)
    ids = torch.randint(4, 97, (2, 24))
    ids[0, 7] = ids[0, 15] = ids[1, 11] = SEP
    return ids


def test_the_published_weights_give_the_published_outputs_when_read_causally():
    library = library_model()
    native = native_from(library)
    ids = ids_with_documents()
    with torch.no_grad():
        expected = library(input_ids=ids).logits
        got = native.logits(native.hidden(ids, causal=True))
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()


def test_both_implementations_read_bidirectionally_with_documents_the_same_way():
    library = library_model()
    native = native_from(library)
    ids = ids_with_documents()
    with torch.no_grad():
        embeds = library.get_input_embeddings()(ids)
        masks = bidirectional_masks(library.model, embeds, document_ids=document_ids_for(ids, SEP))
        expected = library.model(inputs_embeds=embeds, attention_mask=masks).last_hidden_state
        got = native.hidden(ids, mask=dense_document_mask(ids, SEP))
    assert torch.allclose(got, expected, atol=1e-5), (got - expected).abs().max()
    # And with no documents to separate, the unmasked fast path agrees too.
    plain = torch.randint(4, 97, (2, 24))
    with torch.no_grad():
        embeds = library.get_input_embeddings()(plain)
        expected = library.model(
            inputs_embeds=embeds, attention_mask=bidirectional_masks(library.model, embeds)
        ).last_hidden_state
        got = native.hidden(plain)
    assert torch.allclose(got, expected, atol=1e-5)


def test_the_two_document_numberings_agree():
    ids = ids_with_documents()
    assert torch.equal(document_index(ids, SEP), document_ids_for(ids, SEP))


def test_weights_go_back_into_the_library_class_unchanged():
    library = library_model()
    native = native_from(library)
    with torch.no_grad():
        native.body.layers[0].mlp.up_proj.weight.mul_(1.5)
    fresh = Qwen3ForCausalLM(library.config)
    missing, unexpected = fresh.load_state_dict(native.hf_state(), strict=False)
    assert [key for key in missing if key != "lm_head.weight"] == [] and unexpected == []
    assert torch.equal(
        fresh.model.layers[0].mlp.up_proj.weight, native.body.layers[0].mlp.up_proj.weight
    )
    # Tied, so the head follows the embedding without being in the state.
    assert fresh.lm_head.weight.data_ptr() == fresh.model.embed_tokens.weight.data_ptr()


def test_a_checkpoint_that_does_not_fit_is_refused_not_half_loaded():
    library = library_model()
    native = NativeModel(NativeConfig.from_hf(library.config))
    state = dict(library.state_dict())
    state.pop("model.layers.1.self_attn.q_norm.weight")
    with pytest.raises(ValueError, match="missing"):
        native.load_hf_state(state)


def test_the_soft_cap_is_applied_and_bounds_the_logits():
    library = library_model()
    capped = NativeModel(NativeConfig.from_hf(library.config, softcap=30.0)).eval()
    capped.load_hf_state(library.state_dict())
    hidden = torch.randn(2, 5, 48) * 500
    assert capped.logits(hidden).abs().max() <= 30.0


def test_explicit_positions_reproduce_the_default_and_shift_like_rope():
    library = library_model()
    native = native_from(library)
    ids = ids_with_documents()
    default = torch.arange(ids.shape[1]).expand(ids.shape[0], -1)
    with torch.no_grad():
        plain = native.hidden(ids)
        explicit = native.hidden(ids, positions=default)
        # Rotary attention depends on relative offsets only, so shifting every
        # position by the same amount leaves unmasked outputs unchanged.
        shifted = native.hidden(ids, positions=default + 7)
    assert torch.equal(plain, explicit)
    assert torch.allclose(plain, shifted, atol=1e-4)

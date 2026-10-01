"""The continuation corpus: overlap with evaluation text is dropped, shards are well formed."""

import json

import numpy as np

from data.continuation import evaluation_texts, overlap_reason, write_tier
from data.decontam.ngrams import MinHashLSH, NgramIndex

EVAL = "Kargom üç gündür gelmedi, siparişimin nerede olduğunu nasıl öğrenebilirim acaba?"


def checks():
    index = NgramIndex.build([("eval:1", EVAL)])
    lsh = MinHashLSH()
    return [("eval", index, lsh)]


def test_a_document_carrying_an_evaluation_text_is_dropped():
    long = "Mahkeme kararı. " * 40 + EVAL + " Karar kesindir. " * 40
    assert overlap_reason(long, checks()) == "eval:covers"
    assert overlap_reason(EVAL, checks()) == "eval:ngram"
    assert overlap_reason("Tamamen başka bir karar metni ve başka konular. " * 10, checks()) is None


def test_tiers_are_uint16_shards_with_a_separator_after_each_document(tmp_path):
    docs = [("a", "bir iki üç"), ("b", EVAL), ("c", "dört beş")]

    def encode_batch(texts):
        return [[len(w) for w in t.split()] for t in texts]

    counts = write_tier(docs, encode_batch, 99, tmp_path, target_tokens=100, checks=checks(),
                        batch=2)  # fmt: skip
    assert counts["seen"] == 3 and counts["kept"] == 2 and counts["dropped:eval:ngram"] == 1
    tokens = np.fromfile(tmp_path / "part-000.bin", dtype=np.uint16)
    assert tokens.tolist() == [3, 3, 2, 99, 4, 3, 99]
    assert counts["tokens"] == len(tokens)


def test_evaluation_texts_come_from_validation_and_held_out_files_only(tmp_path):
    for rel in ("typed/x/validation.jsonl", "typed/x/train.jsonl", "sss/heldout_task.jsonl"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"row_id": rel, "state": rel}) + "\n")
    got = sorted(t for _, t in evaluation_texts(tmp_path))
    assert got == ["sss/heldout_task.jsonl", "typed/x/validation.jsonl"]


def test_worker_processes_give_the_same_shards_as_one(tmp_path):
    docs = [(str(i), f"belge {i} " + "kelime " * (i % 7 + 1)) for i in range(200)]
    docs[50] = ("50", EVAL)

    def encode_batch(texts):
        return [[len(w) for w in t.split()] for t in texts]

    one = write_tier(docs, encode_batch, 99, tmp_path / "one", 10_000, checks(), batch=8)
    two = write_tier(docs, encode_batch, 99, tmp_path / "two", 10_000, checks(), batch=8,
                     workers=2)  # fmt: skip
    assert one == two and one["dropped:eval:ngram"] == 1
    shard = "part-000.bin"
    assert (tmp_path / "one" / shard).read_bytes() == (tmp_path / "two" / shard).read_bytes()


def test_a_payload_the_filters_drop_is_counted_and_not_written(tmp_path):
    docs = [("a", {"t": "bir iki"}), ("b", {"t": None}), ("c", {"t": "üç"})]

    def encode_batch(texts):
        return [[len(w) for w in t.split()] for t in texts]

    counts = write_tier(docs, encode_batch, 99, tmp_path, 100, checks(), batch=1,
                        prepare=lambda row: row["t"])  # fmt: skip
    assert counts["kept"] == 2 and counts["dropped:filtered"] == 1
    assert np.fromfile(tmp_path / "part-000.bin", dtype=np.uint16).tolist() == [3, 3, 99, 2, 99]


def test_one_minhash_signature_per_document(monkeypatch):
    import data.decontam.ngrams as ngrams

    calls = []
    real = ngrams.minhash_signature
    monkeypatch.setattr(ngrams, "minhash_signature", lambda text: calls.append(text) or real(text))
    text = "Tamamen başka bir karar metni ve başka konular. " * 10
    assert overlap_reason(text, checks() + checks()) is None
    assert len(calls) == 1
